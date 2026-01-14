#!/usr/bin/env python3
"""
Pure classical L-SHADE (Tanabe & Fukunaga, 2014)

Implements SHADE 1.1 + Linear Population Size Reduction (LPSR), i.e. L-SHADE:
- Parameter sampling CR_i, F_i from historical memory (Eq. (1)-(2))
- current-to-pbest/1 mutation (Eq. (3))
- bound handling (Eq. (4))
- binomial crossover (Eq. (5))
- selection (Eq. (6))
- external archive (Sec. II-D)
- memory update via weighted Lehmer means (Algorithm 1, Eq. (7)-(9))
- linear population size reduction (Eq. (10), Algorithm 2 lines 21–24)
"""

import numpy as np
from scipy.stats import cauchy


class LSHADE:
    def __init__(
        self,
        func,
        bounds,
        popsize=None,
        N_min=4,
        max_evals=None,
        memory_size=6,
        sigma_f=0.1,
        sigma_cr=0.1,
        p_best_rate=0.11,
        arc_rate=2.6,
        atol=0.0,
        seed=None,
        disp=False,
    ):
        # Objective function f(x) and box constraints [l,u]^D
        self.func = func
        self.bounds = np.array(bounds, float)
        self.dim = len(bounds)

        # L-SHADE uses N_init = r_Ninit * D (paper tunes r_Ninit=18), and N_min = 4
        # N_min=4 is required by current-to-pbest/1 (needs distinct indices i, pbest, r1, r2)
        self.N_init = popsize if popsize is not None else 18 * self.dim
        self.N_min = N_min
        self.popsize = self.N_init

        # CEC-style budget in paper: MAX_NFE = 10000 * D
        self.max_evals = max_evals if max_evals is not None else 10000 * self.dim

        # Historical memory size H (paper’s tuned H=6); initialize M_CR and M_F to 0.5
        self.memory_size = memory_size
        self.memory_f = np.full(memory_size, 0.5)
        # M_CR stores either a real value in [0,1] or terminal value ⊥ (here represented via None)
        self.memory_cr = np.full(memory_size, 0.5, dtype=object)
        self.memory_pos = 0

        # Parameter sampling noise:
        # - F uses Cauchy centered at M_F with scale 0.1 (Eq. (2) uses randc_i(MF, 0.1))
        # - CR uses Normal centered at M_CR with std 0.1 (Eq. (1) uses randn_i(MCR, 0.1))
        self.sigma_f = sigma_f
        self.sigma_cr = sigma_cr
        self.p_best_rate = p_best_rate  # p in current-to-pbest/1 (paper tuned p=0.11)
        self.arc_rate = arc_rate        # archive size |A| = round(arc_rate * N_init) (paper tuned 2.6)

        # External archive A (Sec. II-D) for diversity; capacity resizes with population
        self.archive_capacity = int(round(self.arc_rate * self.N_init))
        self.archive = []

        self.atol = atol
        self.disp = disp

        # Reproducible RNG (PCG64); paper uses randomness per run
        self.rng = np.random.Generator(np.random.PCG64(seed))

        # Counters: number of evaluations (NFE), number of generations (G)
        self.nfev = 0
        self.nit = 0

        # Best-so-far tracking
        self.best_fitness = np.inf
        self.best_individual = None
        self.convergence = []

        # Rich per-generation logs for evaluation/diagnostics/validation of theory events
        self.history = {
            "memory_f": [],
            "memory_cr": [],
            "pop_size": [],
            "archive_size": [],
            "positions": [],
            "fitness": [],
            "x_best": [],
            "f_best": [],
            "all_F": [],
            "all_CR": [],
            "successful_F": [],
            "successful_CR": [],
            "delta_f": [],
            "trial_fitness_best": [],  # NEW: f(u_{t,b}) for validation (b = best index at gen start)
        }

    def _initialize_population(self):
        """
        Initialization phase (Algorithm 2, line 2):
        P^1 = (x_1,...,x_N) randomly initialized within bounds.
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return self.rng.uniform(lower, upper, (self.N_init, self.dim))

    def _compute_target_pop_size(self):
        """
        Linear Population Size Reduction (LPSR), Eq. (10):
        N_{G+1} = round( (N_min - N_init)/MAX_NFE * NFE + N_init )

        Your 'ratio' formulation is algebraically equivalent and yields a linear schedule
        from N_init down to N_min as NFE approaches MAX_NFE.
        """
        ratio = (self.max_evals - self.nfev) / self.max_evals
        ratio = np.clip(ratio, 0.0, 1.0)
        N_target = self.N_min + (self.N_init - self.N_min) * ratio
        return max(self.N_min, int(round(N_target)))

    def _shrink_population(self, pop, fit):
        """
        LPSR application (Algorithm 2, lines 21–24):
        - compute target NG+1
        - if NG+1 < NG: delete worst (NG - NG+1) individuals (by fitness)
        - resize archive capacity proportional to new population size
        """
        N_target = self._compute_target_pop_size()
        current = len(pop)

        if N_target < current:
            n_remove = current - N_target
            worst = np.argsort(fit)[-n_remove:]
            mask = np.ones(current, dtype=bool)
            mask[worst] = False
            pop = pop[mask]
            fit = fit[mask]

            # Archive resizing after population reduction (Algorithm 2, line 24)
            self.archive_capacity = int(round(self.arc_rate * len(pop)))
            if len(self.archive) > self.archive_capacity:
                idx = self.rng.choice(len(self.archive), self.archive_capacity, replace=False)
                self.archive = [self.archive[i] for i in idx]

        return pop, fit

    def _bound_constrain(self, mutant, parent):
        """
        Bound handling (Eq. (4)):
        If mutant component violates bounds, replace by midpoint between violated bound and parent.
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        result = mutant.copy()

        for j in range(self.dim):
            if result[j] < lower[j]:
                result[j] = 0.5 * (lower[j] + parent[j])
            elif result[j] > upper[j]:
                result[j] = 0.5 * (upper[j] + parent[j])

        return result

    def _update_archive(self, individual):
        """
        External archive A (Sec. II-D):
        - store parent x_i if trial strictly improves it (f(u)<f(x))
        - if |A| exceeds capacity, delete random element to maintain size
        """
        self.archive.append(individual)
        if len(self.archive) > self.archive_capacity:
            idx = self.rng.integers(0, len(self.archive))
            del self.archive[idx]

    def _update_memory(self, successful_f, successful_cr, delta_f):
        """
        Historical memory update (Algorithm 1, Eq. (7)-(9)):

        successful_f  -> S_F
        successful_cr -> S_CR
        delta_f       -> Δf_k = |f(u_k) - f(x_k)| (here it's fit[i] - f_trial when improved)

        Memory slot k = memory_pos updated by weighted Lehmer mean:
          mean_WL(S) = sum w_k S_k^2 / sum w_k S_k
          w_k = Δf_k / sum Δf

        Terminal CR value ⊥:
        - paper: if MCR,k is ⊥ or max(SCR)=0 then set MCR,k <- ⊥ permanently
        - here: represent ⊥ by None and preserve it (see the 'or self.memory_cr[k] is None' branch)
        """
        if len(successful_f) == 0:
            return

        k = self.memory_pos

        s_f = np.asarray(successful_f)
        s_cr = np.asarray(successful_cr)
        delta = np.asarray(delta_f)

        total_delta = delta.sum()
        if total_delta <= 0:
            return

        w = delta / total_delta

        # MF update via weighted Lehmer mean (Eq. (7))
        mean_f = np.sum(w * (s_f ** 2)) / (np.sum(w * s_f) + 1e-12)
        mean_f = float(np.clip(mean_f, 0.0, 1.0))
        self.memory_f[k] = mean_f

        # MCR update and terminal rule (Algorithm 1, line 2)
        if self.memory_cr[k] is None or np.max(s_cr) == 0.0:
            self.memory_cr[k] = None
        else:
            mean_cr = np.sum(w * (s_cr ** 2)) / (np.sum(w * s_cr) + 1e-12)
            self.memory_cr[k] = float(np.clip(mean_cr, 0.0, 1.0))

        # Advance cyclic memory pointer k (Algorithm 1, lines 7-8)
        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def _log_state(self, pop, fit, all_F, all_CR, successful_f, successful_cr, delta_f, trial_fitness_best):
        """
        Logging after a generation:
        - memory snapshots
        - population/archive sizes
        - full population positions/fitness (heavy, but enables deep post-hoc analysis)
        - per-individual sampled (F, CR) and successful subsets, plus improvements Δf
        - trial_fitness_best: f(u_{t,b}) for the individual b that was best at generation start
        """
        self.history["memory_f"].append(self.memory_f.copy())
        self.history["memory_cr"].append(
            np.array([m if m is not None else np.nan for m in self.memory_cr])
        )
        self.history["pop_size"].append(len(pop))
        self.history["archive_size"].append(len(self.archive))
        self.history["positions"].append(pop.copy())
        self.history["fitness"].append(fit.copy())
        best_idx = np.argmin(fit)
        self.history["x_best"].append(pop[best_idx].copy())
        self.history["f_best"].append(float(fit[best_idx]))
        self.history["all_F"].append(np.array(all_F))
        self.history["all_CR"].append(np.array(all_CR))
        self.history["successful_F"].append(np.array(successful_f))
        self.history["successful_CR"].append(np.array(successful_cr))
        self.history["delta_f"].append(np.array(delta_f))
        self.history["trial_fitness_best"].append(trial_fitness_best)

    def _log_initial_state(self, pop, fit):
        """
        Logging at generation start (Algorithm 2 initialization phase):
        No (F,CR) yet at t=0, so store empty arrays and NaN trial fitness.
        """
        self.history["memory_f"].append(self.memory_f.copy())
        self.history["memory_cr"].append(
            np.array([m if m is not None else np.nan for m in self.memory_cr])
        )
        self.history["pop_size"].append(len(pop))
        self.history["archive_size"].append(len(self.archive))
        self.history["positions"].append(pop.copy())
        self.history["fitness"].append(fit.copy())
        best_idx = np.argmin(fit)
        self.history["x_best"].append(pop[best_idx].copy())
        self.history["f_best"].append(float(fit[best_idx]))
        self.history["all_F"].append(np.array([]))
        self.history["all_CR"].append(np.array([]))
        self.history["successful_F"].append(np.array([]))
        self.history["successful_CR"].append(np.array([]))
        self.history["delta_f"].append(np.array([]))
        self.history["trial_fitness_best"].append(np.nan)

    def _lshade_generation(self, pop, fit):
        """
        One generation of L-SHADE (Algorithm 2 main loop, lines 5–25):

        For each i:
        - sample r_i uniformly from {1,...,H} (Algorithm 2 line 7)
        - sample CR_i from Normal(MCR[r_i], 0.1) or 0 if ⊥ (Eq. (1))
        - sample F_i  from Cauchy(MF[r_i], 0.1) with resampling until F>0 (Eq. (2))
        - current-to-pbest/1 mutation (Eq. (3)), with archive option for r2 (Sec. II-D)
        - bound correction (Eq. (4))
        - binomial crossover (Eq. (5))
        - selection (Eq. (6))
        - collect successful (CR,F) pairs and Δf for memory update (Algorithm 1 / Eq. (7)-(9))
        - after loop: update memory, shrink population by LPSR, log state
        """
        NP = len(pop)

        # Identify best at start of generation (useful for hazard/witness validations)
        best_idx_start = int(np.argmin(fit))
        trial_fitness_best = np.nan

        new_pop = []
        new_fit = []

        successful_f = []
        successful_cr = []
        delta_f = []
        all_F = []
        all_CR = []

        for i in range(NP):
            # Budget stop: if max evals reached mid-generation, carry over remaining individuals unchanged
            if self.nfev >= self.max_evals:
                for j in range(i, NP):
                    new_pop.append(pop[j])
                    new_fit.append(fit[j])
                    all_F.append(np.nan)
                    all_CR.append(np.nan)
                break

            # r_i ~ Uniform{0,...,H-1} corresponds to selecting a memory index (Algorithm 2 line 7)
            r = self.rng.integers(0, self.memory_size)

            # F_i sampling: Cauchy centered at MF[r], truncate at 1, resample until >0 (Eq. (2))
            while True:
                F = cauchy.rvs(
                    loc=self.memory_f[r],
                    scale=self.sigma_f,
                    random_state=self.rng,
                )
                if F > 1:
                    F = 1
                if F > 0:
                    break

            # CR_i sampling: if memory CR is terminal ⊥ -> CR=0 (Eq. (1))
            mu = self.memory_cr[r]
            if mu is None:
                CR = 0.0
            else:
                CR = float(self.rng.normal(mu, self.sigma_cr))
                CR = float(np.clip(CR, 0.0, 1.0))

            all_F.append(F)
            all_CR.append(CR)

            # p-best selection: choose from top ceil(p * NP) by fitness (Eq. (3) description)
            p_num = max(2, int(np.ceil(self.p_best_rate * NP)))
            pbest_pool = np.argsort(fit)[:p_num]
            pbest_idx = self.rng.choice(pbest_pool)

            # r1 drawn from population indices excluding i (Eq. (3) requires distinct indices)
            r1 = self.rng.choice([j for j in range(NP) if j != i])

            # r2 drawn from P ∪ A excluding i and r1 (Sec. II-D)
            total = NP + len(self.archive)
            candidates = [j for j in range(total) if j != i and j != r1]
            if not candidates:
                new_pop.append(pop[i])
                new_fit.append(fit[i])
                continue

            r2 = self.rng.choice(candidates)
            x_r2 = pop[r2] if r2 < NP else self.archive[r2 - NP]

            # current-to-pbest/1 mutation (Eq. (3))
            mutant = (
                pop[i]
                + F * (pop[pbest_idx] - pop[i])
                + F * (pop[r1] - x_r2)
            )

            # Bound correction (Eq. (4))
            mutant = self._bound_constrain(mutant, pop[i])

            # Binomial crossover (Eq. (5))
            trial = pop[i].copy()
            j_rand = self.rng.integers(0, self.dim)
            for j in range(self.dim):
                if self.rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Evaluate trial
            f_trial = float(self.func(trial))
            self.nfev += 1

            # Extra logging: trial fitness for best-at-start index
            if i == best_idx_start:
                trial_fitness_best = f_trial

            # Selection (Eq. (6))
            if f_trial <= fit[i]:
                new_pop.append(trial)
                new_fit.append(f_trial)

                # Strict improvement -> archive + success-history lists (Algorithm 2 lines 16–18)
                if f_trial < fit[i]:
                    self._update_archive(pop[i].copy())
                    successful_f.append(F)
                    successful_cr.append(CR)
                    delta_f.append(fit[i] - f_trial)
            else:
                new_pop.append(pop[i])
                new_fit.append(fit[i])

        # Memory update at generation end (Algorithm 2 line 20)
        self._update_memory(successful_f, successful_cr, delta_f)

        new_pop = np.asarray(new_pop)
        new_fit = np.asarray(new_fit)

        # LPSR population reduction and archive resizing (Algorithm 2 lines 21–24)
        new_pop, new_fit = self._shrink_population(new_pop, new_fit)

        # Log full state
        self._log_state(new_pop, new_fit, all_F, all_CR, successful_f, successful_cr, delta_f, trial_fitness_best)

        return new_pop, new_fit

    def solve(self):
        """
        Full L-SHADE run (Algorithm 2):

        - initialize population (line 2)
        - initialize memories to 0.5 (line 3)
        - loop until termination (budget or atol):
            - generate trials via current-to-pbest/1/bin, selection, archive updates, memory updates
            - apply LPSR population reduction
        """
        pop = self._initialize_population()
        fit = np.array([self.func(x) for x in pop])
        self.nfev += len(fit)

        best_idx = np.argmin(fit)
        self.best_fitness = float(fit[best_idx])
        self.best_individual = pop[best_idx].copy()
        self.convergence.append(self.best_fitness)

        self._log_initial_state(pop, fit)

        if self.disp:
            print(f"Initial best = {self.best_fitness:.6e}")

        while self.nfev < self.max_evals:
            self.nit += 1
            pop, fit = self._lshade_generation(pop, fit)

            best_idx = np.argmin(fit)
            best_now = float(fit[best_idx])
            if best_now < self.best_fitness:
                self.best_fitness = best_now
                self.best_individual = pop[best_idx].copy()

            self.convergence.append(self.best_fitness)

            # Optional early termination threshold (not part of paper’s benchmarking protocol)
            if self.atol > 0 and self.best_fitness <= self.atol:
                break

        return type(
            "Result",
            (),
            dict(
                x=self.best_individual,
                fun=self.best_fitness,
                nit=self.nit,
                nfev=self.nfev,
                convergence=self.convergence,
                final_pop_size=len(pop),
                history=self.history,
            ),
        )()
