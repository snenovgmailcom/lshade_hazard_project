#!/usr/bin/env python3
"""
Validation experiment v2: Larger epsilon for more data points.
"""

import numpy as np
from scipy.stats import cauchy, norm
from pathlib import Path
import pickle
from collections import defaultdict

# ============================================================
# Benchmark
# ============================================================

class SphereBenchmark:
    def __init__(self, dim, shift=None):
        self.dim = dim
        if shift is None:
            np.random.seed(0)
            self.shift = np.random.uniform(-80, 80, dim)
        else:
            self.shift = shift
        self.f_star = 0.0
        self.bounds = (-100, 100)
    
    def __call__(self, x):
        z = x - self.shift
        return np.sum(z**2) + self.f_star
    
    def get_optimal(self):
        return self.shift.copy()


# ============================================================
# L-SHADE with logging (simplified)
# ============================================================

class LSHADELogged:
    def __init__(self, func, dim, max_evals, p_best_rate=0.11, arc_rate=2.6, 
                 memory_size=6, n_init_multiplier=18, n_min=4):
        self.func = func
        self.dim = dim
        self.max_evals = max_evals
        self.p_best_rate = p_best_rate
        self.arc_rate = arc_rate
        self.memory_size = memory_size
        self.n_init = n_init_multiplier * dim
        self.n_min = n_min
        self.logs = []
        
    def run(self, log_every=1):
        # Initialize
        lb, ub = self.func.bounds
        pop = np.random.uniform(lb, ub, (self.n_init, self.dim))
        fitness = np.array([self.func(x) for x in pop])
        archive = []
        memory_F = np.full(self.memory_size, 0.5)
        memory_CR = np.full(self.memory_size, 0.5)
        memory_idx = 0
        n_evals = self.n_init
        gen = 0
        
        self.logs = []
        
        while n_evals < self.max_evals:
            gen += 1
            NP = len(pop)
            
            # Log
            if gen % log_every == 0:
                self.logs.append({
                    'generation': gen,
                    'n_evals': n_evals,
                    'population': pop.copy(),
                    'fitness': fitness.copy(),
                    'archive': [a.copy() for a in archive],
                    'memory_F': memory_F.copy(),
                    'memory_CR': memory_CR.copy(),
                    'best_fitness': np.min(fitness),
                })
            
            S_F, S_CR, S_delta = [], [], []
            trials = []
            
            n_pbest = max(1, int(self.p_best_rate * NP))
            pbest_idx = np.argsort(fitness)[:n_pbest]
            
            for i in range(NP):
                if n_evals >= self.max_evals:
                    break
                
                # Sample parameters
                r = np.random.randint(self.memory_size)
                F = self._sample_F(memory_F[r])
                CR = self._sample_CR(memory_CR[r])
                
                # Mutation: current-to-pbest/1
                b = np.random.choice(pbest_idx)
                r1 = np.random.choice([x for x in range(NP) if x not in [i, b]])
                
                pool = list(range(NP)) + list(range(NP, NP + len(archive)))
                pool = [x for x in pool if x not in [i, b, r1]]
                r2_idx = np.random.choice(pool)
                x_r2 = archive[r2_idx - NP] if r2_idx >= NP else pop[r2_idx]
                
                v = pop[i] + F * (pop[b] - pop[i]) + F * (pop[r1] - x_r2)
                
                # Boundary handling
                for j in range(self.dim):
                    if v[j] < lb:
                        v[j] = (lb + pop[i, j]) / 2
                    elif v[j] > ub:
                        v[j] = (ub + pop[i, j]) / 2
                
                # Crossover
                u = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() <= CR or j == j_rand:
                        u[j] = v[j]
                
                f_u = self.func(u)
                n_evals += 1
                trials.append((i, u, f_u, F, CR))
            
            # Selection
            for i, u, f_u, F, CR in trials:
                if f_u <= fitness[i]:
                    if f_u < fitness[i]:
                        S_F.append(F)
                        S_CR.append(CR)
                        S_delta.append(fitness[i] - f_u)
                        if len(archive) < int(self.arc_rate * self.n_init):
                            archive.append(pop[i].copy())
                        else:
                            archive[np.random.randint(len(archive))] = pop[i].copy()
                    pop[i] = u
                    fitness[i] = f_u
            
            # Update memory
            if len(S_F) > 0:
                w = np.array(S_delta) / np.sum(S_delta)
                memory_F[memory_idx] = np.sum(w * np.array(S_F)**2) / np.sum(w * np.array(S_F))
                if np.sum(w * np.array(S_CR)) > 0:
                    memory_CR[memory_idx] = np.sum(w * np.array(S_CR)**2) / np.sum(w * np.array(S_CR))
                memory_idx = (memory_idx + 1) % self.memory_size
            
            # LPSR
            target_NP = max(self.n_min, round(self.n_init + (self.n_min - self.n_init) * n_evals / self.max_evals))
            if target_NP < NP:
                keep = np.argsort(fitness)[:target_NP]
                pop = pop[keep]
                fitness = fitness[keep]
        
        return {
            'best_fitness': np.min(fitness),
            'logs': self.logs,
        }
    
    def _sample_F(self, mu):
        while True:
            F = cauchy.rvs(loc=mu, scale=0.1)
            if F > 0:
                return min(F, 1.0)
    
    def _sample_CR(self, mu):
        return np.clip(norm.rvs(loc=mu, scale=0.1), 0, 1)


# ============================================================
# Analysis functions
# ============================================================

def compute_ell_and_crossover(log, func, target, dim, 
                               n_tuples=500, n_F_samples=50, n_cross_samples=100):
    """
    Compute γ_t, ℓ, and crossover success probability.
    """
    pop = log['population']
    fitness = log['fitness']
    archive = log['archive']
    NP = len(pop)
    
    if NP < 4:
        return {'gamma': 0, 'ell_mean': 0, 'cross_prob': 0, 'tuples_with_ell': 0}
    
    n_pbest = max(1, int(0.11 * NP))
    pbest_idx = np.argsort(fitness)[:n_pbest]
    pool_size = NP + len(archive)
    
    CR = np.mean(log['memory_CR'])
    
    tuples_with_ell = 0
    ell_values = []
    cross_probs = []
    
    for _ in range(n_tuples):
        # Sample tuple
        i = np.random.randint(NP)
        b = np.random.choice(pbest_idx)
        r1_cand = [x for x in range(NP) if x not in [i, b]]
        if not r1_cand:
            continue
        r1 = np.random.choice(r1_cand)
        r2_cand = [x for x in range(pool_size) if x not in [i, b, r1]]
        if not r2_cand:
            continue
        r2_idx = np.random.choice(r2_cand)
        
        x_i = pop[i]
        x_b = pop[b]
        x_r1 = pop[r1]
        x_r2 = archive[r2_idx - NP] if r2_idx >= NP else pop[r2_idx]
        
        # Direction vector
        d = (x_b - x_i) + (x_r1 - x_r2)
        
        # Find successful F values
        F_values = np.linspace(0.1, 1.0, n_F_samples)
        successful_Fs = []
        
        for F in F_values:
            v = x_i + F * d
            # Boundary check
            lb, ub = func.bounds
            v_clipped = np.clip(v, lb, ub)
            if func(v_clipped) <= target:
                successful_Fs.append(F)
        
        # Compute ℓ
        if successful_Fs:
            ell = len(successful_Fs) / n_F_samples * 0.9  # Scale by F range
            ell_values.append(ell)
            tuples_with_ell += 1
            
            # Compute crossover success for a successful mutant
            F_good = successful_Fs[len(successful_Fs)//2]  # Use middle successful F
            v_good = x_i + F_good * d
            v_good = np.clip(v_good, lb, ub)
            
            # Sample crossover
            n_cross_success = 0
            for _ in range(n_cross_samples):
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) <= CR
                mask[j_rand] = True
                u = np.where(mask, v_good, x_i)
                if func(u) <= target:
                    n_cross_success += 1
            
            cross_probs.append(n_cross_success / n_cross_samples)
    
    gamma = tuples_with_ell / n_tuples if n_tuples > 0 else 0
    ell_mean = np.mean(ell_values) if ell_values else 0
    cross_prob = np.mean(cross_probs) if cross_probs else 0
    
    return {
        'gamma': gamma,
        'ell_mean': ell_mean,
        'cross_prob': cross_prob,
        'tuples_with_ell': tuples_with_ell,
        'n_tuples': n_tuples,
        'CR': CR,
    }


def compute_a_t(gamma, ell, cross_prob, NP, archive_size, dim, g_F=0.1, q_CR=0.5):
    """
    Compute both old and new bounds.
    """
    m_t = max(1, int(0.11 * NP))
    s = NP + archive_size - 2
    
    if s < 2 or ell <= 0:
        return 0, 0
    
    # Common factor
    base = (g_F * ell) / (m_t * s * (s - 1)) * q_CR
    
    # Old bound: c_0^{d-1}
    c0 = 0.5
    a_old = base * (c0 ** (dim - 1))
    
    # New bound: empirical crossover probability
    a_new = base * cross_prob
    
    return a_old, a_new


# ============================================================
# Main experiment
# ============================================================

def run_experiment(n_runs=30, dim=10, max_evals=100000, epsilon=10.0):
    """
    Run validation with larger epsilon for more data.
    """
    print("=" * 80)
    print("VALIDATION: a_t * γ_t vs ĥ_t")
    print("=" * 80)
    print(f"Dimension: {dim}")
    print(f"Max evals: {max_evals}")
    print(f"Epsilon: {epsilon}")
    print(f"Runs: {n_runs}")
    
    func = SphereBenchmark(dim)
    target = func.f_star + epsilon
    print(f"Target: f* + ε = {func.f_star} + {epsilon} = {target}")
    print()
    
    all_hit_times = []
    all_analysis = []
    
    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...", end=" ")
        np.random.seed(42 + run)
        
        lshade = LSHADELogged(func, dim, max_evals)
        result = lshade.run(log_every=2)  # Log every 2 generations
        
        # Find hit time
        hit_gen = None
        for log in result['logs']:
            if log['best_fitness'] <= target:
                hit_gen = log['generation']
                break
        
        all_hit_times.append(hit_gen if hit_gen else np.inf)
        print(f"hit at gen {hit_gen}" if hit_gen else "no hit")
        
        # Analyze each logged generation before hit
        for log in result['logs']:
            if log['best_fitness'] <= target:
                break
            
            stats = compute_ell_and_crossover(log, func, target, dim,
                                               n_tuples=300, n_cross_samples=50)
            
            NP = len(log['population'])
            arc_size = len(log['archive'])
            
            a_old, a_new = compute_a_t(
                stats['gamma'], stats['ell_mean'], stats['cross_prob'],
                NP, arc_size, dim
            )
            
            all_analysis.append({
                'run': run,
                'gen': log['generation'],
                'best_f': log['best_fitness'],
                'gap': log['best_fitness'] - target,
                'gamma': stats['gamma'],
                'ell': stats['ell_mean'],
                'cross_prob': stats['cross_prob'],
                'a_old': a_old,
                'a_new': a_new,
                'a_gamma_old': a_old * stats['gamma'],
                'a_gamma_new': a_new * stats['gamma'],
            })
    
    # Compute empirical hazard
    print("\n" + "=" * 80)
    print("EMPIRICAL HAZARD")
    print("=" * 80)
    
    hit_times = np.array(all_hit_times)
    finite_hits = hit_times[hit_times < np.inf]
    
    print(f"Hits: {len(finite_hits)}/{n_runs}")
    if len(finite_hits) > 0:
        print(f"Hit time: min={min(finite_hits)}, median={np.median(finite_hits):.0f}, max={max(finite_hits)}")
    
    # Hazard at each time
    hazards = {}
    if len(finite_hits) > 0:
        for t in range(int(max(finite_hits)) + 1):
            n_t = np.sum(hit_times >= t)
            d_t = np.sum(hit_times == t)
            if n_t > 0:
                hazards[t] = d_t / n_t
    
    # Aggregate analysis by generation
    print("\n" + "=" * 80)
    print("COMPARISON BY GENERATION")
    print("=" * 80)
    
    by_gen = defaultdict(list)
    for a in all_analysis:
        by_gen[a['gen']].append(a)
    
    print(f"\n{'Gen':>5} {'Gap':>8} {'γ':>8} {'ℓ':>10} {'P(cross)':>10} "
          f"{'a_old*γ':>12} {'a_new*γ':>12} {'ĥ_t':>10} {'new/old':>8}")
    print("-" * 100)
    
    significant_gens = []
    
    for gen in sorted(by_gen.keys()):
        records = by_gen[gen]
        
        avg_gap = np.mean([r['gap'] for r in records])
        avg_gamma = np.mean([r['gamma'] for r in records])
        avg_ell = np.mean([r['ell'] for r in records])
        avg_cross = np.mean([r['cross_prob'] for r in records])
        avg_a_old = np.mean([r['a_gamma_old'] for r in records])
        avg_a_new = np.mean([r['a_gamma_new'] for r in records])
        
        h_hat = hazards.get(gen, 0)
        ratio = avg_a_new / avg_a_old if avg_a_old > 0 else 0
        
        # Only print generations with γ > 0
        if avg_gamma > 0.001:
            print(f"{gen:>5} {avg_gap:>8.2f} {avg_gamma:>8.4f} {avg_ell:>10.6f} "
                  f"{avg_cross:>10.4f} {avg_a_old:>12.2e} {avg_a_new:>12.2e} "
                  f"{h_hat:>10.4f} {ratio:>8.1f}x")
            
            significant_gens.append({
                'gen': gen,
                'gap': avg_gap,
                'gamma': avg_gamma,
                'ell': avg_ell,
                'cross_prob': avg_cross,
                'a_gamma_old': avg_a_old,
                'a_gamma_new': avg_a_new,
                'h_hat': h_hat,
                'ratio': ratio,
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    c0_term = 0.5 ** (dim - 1)
    print(f"\nOld bound factor: c_0^(d-1) = 0.5^{dim-1} = {c0_term:.2e}")
    
    if significant_gens:
        avg_cross_all = np.mean([g['cross_prob'] for g in significant_gens])
        avg_ratio = np.mean([g['ratio'] for g in significant_gens if g['ratio'] > 0])
        
        print(f"New bound factor: avg P(crossover) = {avg_cross_all:.4f}")
        print(f"Improvement: {avg_cross_all / c0_term:.1f}x tighter")
        
        # Compare with empirical hazard
        gens_with_events = [g for g in significant_gens if g['h_hat'] > 0]
        if gens_with_events:
            print(f"\nGenerations with events (h_hat > 0):")
            for g in gens_with_events:
                ratio_to_h_old = g['h_hat'] / g['a_gamma_old'] if g['a_gamma_old'] > 0 else np.inf
                ratio_to_h_new = g['h_hat'] / g['a_gamma_new'] if g['a_gamma_new'] > 0 else np.inf
                print(f"  Gen {g['gen']}: ĥ={g['h_hat']:.4f}, "
                      f"a_old*γ={g['a_gamma_old']:.2e} (ĥ/{g['a_gamma_old']:.2e}={ratio_to_h_old:.1f}x), "
                      f"a_new*γ={g['a_gamma_new']:.2e} (ĥ/{g['a_gamma_new']:.2e}={ratio_to_h_new:.1f}x)")
    
    # Save
    output = {
        'hit_times': all_hit_times,
        'analysis': all_analysis,
        'hazards': hazards,
        'significant_gens': significant_gens,
        'params': {'dim': dim, 'epsilon': epsilon, 'n_runs': n_runs},
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/validation_v2.pkl', 'wb') as f:
        pickle.dump(output, f)
    print("\nSaved: results/validation_v2.pkl")
    
    return output


if __name__ == "__main__":
    # Use larger epsilon to see more generations with γ > 0
    run_experiment(
        n_runs=30,
        dim=10,
        max_evals=100000,
        epsilon=10.0,  # Larger target region
    )
