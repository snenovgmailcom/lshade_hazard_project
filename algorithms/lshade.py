#!/usr/bin/env python3
"""
Pure classical L-SHADE (Tanabe & Fukunaga, 2014)
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
        self.func = func
        self.bounds = np.array(bounds, float)
        self.dim = len(bounds)

        self.N_init = popsize if popsize is not None else 18 * self.dim
        self.N_min = N_min
        self.popsize = self.N_init

        self.max_evals = max_evals if max_evals is not None else 10000 * self.dim

        self.memory_size = memory_size
        self.memory_f = np.full(memory_size, 0.5)
        self.memory_cr = np.full(memory_size, 0.5, dtype=object)
        self.memory_pos = 0

        self.sigma_f = sigma_f
        self.sigma_cr = sigma_cr
        self.p_best_rate = p_best_rate
        self.arc_rate = arc_rate

        self.archive_capacity = int(round(self.arc_rate * self.N_init))
        self.archive = []

        self.atol = atol
        self.disp = disp

        self.rng = np.random.Generator(np.random.PCG64(seed))

        self.nfev = 0
        self.nit = 0

        self.best_fitness = np.inf
        self.best_individual = None
        self.convergence = []

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
            "trial_fitness_best": [],  # NEW: f(u_{t,b}) for validation
        }

    def _initialize_population(self):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return self.rng.uniform(lower, upper, (self.N_init, self.dim))

    def _compute_target_pop_size(self):
        ratio = (self.max_evals - self.nfev) / self.max_evals
        ratio = np.clip(ratio, 0.0, 1.0)
        N_target = self.N_min + (self.N_init - self.N_min) * ratio
        return max(self.N_min, int(round(N_target)))

    def _shrink_population(self, pop, fit):
        N_target = self._compute_target_pop_size()
        current = len(pop)

        if N_target < current:
            n_remove = current - N_target
            worst = np.argsort(fit)[-n_remove:]
            mask = np.ones(current, dtype=bool)
            mask[worst] = False
            pop = pop[mask]
            fit = fit[mask]

            self.archive_capacity = int(round(self.arc_rate * len(pop)))
            if len(self.archive) > self.archive_capacity:
                idx = self.rng.choice(len(self.archive), self.archive_capacity, replace=False)
                self.archive = [self.archive[i] for i in idx]

        return pop, fit

    def _bound_constrain(self, mutant, parent):
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
        self.archive.append(individual)
        if len(self.archive) > self.archive_capacity:
            idx = self.rng.integers(0, len(self.archive))
            del self.archive[idx]

    def _update_memory(self, successful_f, successful_cr, delta_f):
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

        mean_f = np.sum(w * (s_f ** 2)) / (np.sum(w * s_f) + 1e-12)
        mean_f = float(np.clip(mean_f, 0.0, 1.0))
        self.memory_f[k] = mean_f

        if self.memory_cr[k] is None or np.max(s_cr) == 0.0:
            self.memory_cr[k] = None
        else:
            mean_cr = np.sum(w * (s_cr ** 2)) / (np.sum(w * s_cr) + 1e-12)
            self.memory_cr[k] = float(np.clip(mean_cr, 0.0, 1.0))

        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def _log_state(self, pop, fit, all_F, all_CR, successful_f, successful_cr, delta_f, trial_fitness_best):
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
        self.history["trial_fitness_best"].append(trial_fitness_best)  # NEW

    def _log_initial_state(self, pop, fit):
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
        self.history["trial_fitness_best"].append(np.nan)  # NEW: no trial at t=0

    def _lshade_generation(self, pop, fit):
        NP = len(pop)

        # NEW: identify best individual at start of generation
        best_idx_start = int(np.argmin(fit))
        trial_fitness_best = np.nan  # will be set when i == best_idx_start

        new_pop = []
        new_fit = []

        successful_f = []
        successful_cr = []
        delta_f = []
        all_F = []
        all_CR = []

        for i in range(NP):
            if self.nfev >= self.max_evals:
                for j in range(i, NP):
                    new_pop.append(pop[j])
                    new_fit.append(fit[j])
                    all_F.append(np.nan)
                    all_CR.append(np.nan)
                break

            r = self.rng.integers(0, self.memory_size)

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

            mu = self.memory_cr[r]

            if mu is None:
                CR = 0.0
            else:
                CR = float(self.rng.normal(mu, self.sigma_cr))
                CR = float(np.clip(CR, 0.0, 1.0))

            all_F.append(F)
            all_CR.append(CR)

            p_num = max(2, int(np.ceil(self.p_best_rate * NP)))
            pbest_pool = np.argsort(fit)[:p_num]
            pbest_idx = self.rng.choice(pbest_pool)

            r1 = self.rng.choice([j for j in range(NP) if j != i])

            total = NP + len(self.archive)
            candidates = [j for j in range(total) if j != i and j != r1]
            if not candidates:
                new_pop.append(pop[i])
                new_fit.append(fit[i])
                continue

            r2 = self.rng.choice(candidates)
            x_r2 = pop[r2] if r2 < NP else self.archive[r2 - NP]

            mutant = (
                pop[i]
                + F * (pop[pbest_idx] - pop[i])
                + F * (pop[r1] - x_r2)
            )

            mutant = self._bound_constrain(mutant, pop[i])

            trial = pop[i].copy()
            j_rand = self.rng.integers(0, self.dim)
            for j in range(self.dim):
                if self.rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            f_trial = float(self.func(trial))
            self.nfev += 1

            # NEW: record trial fitness for best individual
            if i == best_idx_start:
                trial_fitness_best = f_trial

            if f_trial <= fit[i]:
                new_pop.append(trial)
                new_fit.append(f_trial)

                if f_trial < fit[i]:
                    self._update_archive(pop[i].copy())
                    successful_f.append(F)
                    successful_cr.append(CR)
                    delta_f.append(fit[i] - f_trial)
            else:
                new_pop.append(pop[i])
                new_fit.append(fit[i])

        self._update_memory(successful_f, successful_cr, delta_f)

        new_pop = np.asarray(new_pop)
        new_fit = np.asarray(new_fit)

        new_pop, new_fit = self._shrink_population(new_pop, new_fit)

        self._log_state(new_pop, new_fit, all_F, all_CR, successful_f, successful_cr, delta_f, trial_fitness_best)

        return new_pop, new_fit

    def solve(self):
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
