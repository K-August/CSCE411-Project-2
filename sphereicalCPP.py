import pickle
import time
import numpy as np
import random
import os
import fast_evaluator
from concurrent.futures import ProcessPoolExecutor

# Keep your global cache in Python for maximum speed
mheight_cache = {}

def clear_cache():
    global mheight_cache
    mheight_cache = {}

# ==========================================
# NEW: Top-level worker for Multiprocessing
# ==========================================
def eval_worker(args):
    # This must be at the top level so the ProcessPoolExecutor can pickle it
    import fast_evaluator
    n, k, m, mat, threshold = args
    return fast_evaluator.calc_mHeight_efficient(n, k, m, mat, threshold)

def get_cached_mHeight(n, k, m, P, threshold=0.95):
    cache_key = (n, k, m, tuple(P.flatten()))
    if cache_key in mheight_cache:
        return mheight_cache[cache_key]
    
    height = fast_evaluator.calc_mHeight_efficient(n, k, m, P, threshold)
    mheight_cache[cache_key] = height
    return height

# ==========================================
# NEW: Hypersphere Initialization
# ==========================================
def generate_hypersphere_matrix(k, cols, radius_min=2, radius_max=8):
    # Generate normal Gaussian floats (naturally spherically symmetric)
    rand_floats = np.random.randn(k, cols)
    
    # Normalize each column to sit exactly on the unit hypersphere surface
    norms = np.linalg.norm(rand_floats, axis=0)
    norms[norms == 0] = 1e-10 # Prevent division by zero
    unit_sphere_vectors = rand_floats / norms
    
    # Scale by a random radius favoring the small numbers
    radius = np.random.uniform(radius_min, radius_max)
    
    # Round to integers
    return np.round(unit_sphere_vectors * radius).astype(int)

# On the Height Profile of Analog Error-Correcting Codes
def genetic_search(n, k, m, target_height, current_best_matrix=None, starters=100, track=10, generations=50, perturbations=20): 
    cols = n - k
    proxy_m = m
    population = []
    
    if current_best_matrix is not None:
        best_proxy_score = get_cached_mHeight(n, k, proxy_m, current_best_matrix)
        population.append((current_best_matrix.copy(), best_proxy_score))
    
    count = 0
    # 1. Generate starting matrices using the Hypersphere method
    for _ in range(starters - 1):
        G = generate_hypersphere_matrix(k, cols, radius_min=2, radius_max=9)
            
        if np.any(np.sum(np.abs(G), axis=0) == 0):
            continue
            
        height = get_cached_mHeight(n, k, proxy_m, G, 0.940)
        
        if height == float('inf') or height == 6969.0:
            count += 1
            continue 

        population.append((G, height))
        
    print(f"Generated {len(population)} valid starting matrices (discarded {count} candidates)")
    
    if len(population) == 0:
        print("WARNING: Could not find any valid matrices to start with! Returning infinity.")
        return None, float('inf')
    
    population.sort(key=lambda x: x[1])
    S = [item[0] for item in population[:track]]
    best_proxy_height = population[0][1]
    
    print(f"Starting evolutionary part")
    
    for gen in range(generations):  
        Snew = []
        
        # --- Crossover Phase ---
        crossover_count = perturbations // 2
        for _ in range(crossover_count):
            if len(S) < 2:
                break
            parent1, parent2 = random.sample(S, 2)
            child = np.zeros_like(parent1)
            for c in range(cols):
                if random.random() > 0.5:
                    child[:, c] = parent1[:, c]
                else:
                    child[:, c] = parent2[:, c]
            if np.any(np.sum(np.abs(child), axis=0) == 0):
                continue
            Snew.append(child)

        # --- Mutation & Greedy Slide Phase ---
        threshold = 0.935
        for _ in range(perturbations):
            G = random.choice(S).copy()
            current_height = get_cached_mHeight(n, k, proxy_m, G, threshold)
            
            P = np.zeros((k, cols), dtype=int)
            row = random.randint(0, k - 1)
            col = random.randint(0, cols - 1)
            P[row, col] = random.choice([-1, 1]) 
            
            # NEW: Catastrophic Mutation (10% chance to jump big)
            if random.random() > 0.1:
                sigma = 1
            else:
                sigma = random.choice([2, 3, 4]) 
                
            Gnew = np.clip(G + sigma * P, -15, 15)
            
            if np.any(np.sum(np.abs(Gnew), axis=0) == 0):
                continue
                
            new_height = get_cached_mHeight(n, k, proxy_m, Gnew, threshold)
            
            # Greedy slide
            while new_height < current_height:
                true_height_check = get_cached_mHeight(n, k, m, Gnew, threshold)
                if true_height_check == float('inf'):
                    break 
                
                Snew.append(Gnew.copy())
                G = Gnew.copy()
                current_height = new_height
                
                Gnew = np.clip(G + sigma * P, -15, 15)
                if np.any(np.sum(np.abs(Gnew), axis=0) == 0):
                    break
                new_height = get_cached_mHeight(n, k, proxy_m, Gnew, threshold)

        # --- NEW: Diversity Injector (Fresh Blood) ---
        fresh_blood_count = track // 10
        for _ in range(fresh_blood_count):
            fresh_G = generate_hypersphere_matrix(k, cols, radius_min=2, radius_max=15)
            if not np.any(np.sum(np.abs(fresh_G), axis=0) == 0):
                Snew.append(fresh_G)

        # --- NEW: Multiprocessing Evaluation ---
        combined = S + Snew
        
        # Deduplicate to save evaluating identical matrices
        unique_combined = []
        seen = set()
        for mat in combined:
            mat_tuple = tuple(mat.flatten())
            if mat_tuple not in seen:
                seen.add(mat_tuple)
                unique_combined.append(mat)
                
        scored = []
        to_evaluate = []
        
        for mat in unique_combined:
            cache_key = (n, k, proxy_m, tuple(mat.flatten()))
            if cache_key in mheight_cache:
                scored.append((mat, mheight_cache[cache_key]))
            else:
                to_evaluate.append((n, k, proxy_m, mat, threshold))
                
        # Evaluate all unseen matrices in parallel across CPU cores
        if to_evaluate:
            with ProcessPoolExecutor() as executor:
                results = executor.map(eval_worker, to_evaluate)
                for args, h in zip(to_evaluate, results):
                    mat = args[3]
                    cache_key = (n, k, proxy_m, tuple(mat.flatten()))
                    mheight_cache[cache_key] = h # Update local cache
                    scored.append((mat, h))
            
        scored.sort(key=lambda x: x[1])
        S = [item[0] for item in scored[:track]]
        
        if scored[0][1] < best_proxy_height:
            best_proxy_height = scored[0][1]
            print(f"Gen {gen}: height pushed down to {best_proxy_height:.4f}")

    best_G = S[0]
    final_target_height = get_cached_mHeight(n, k, m, best_G, threshold)
    print(f"Final m={m} height: {final_target_height:.4f} (Target: {target_height})")
    
    return best_G, final_target_height


if __name__ == '__main__':
    ideal_targets = {
        (9, 4, 2): 1.17, (9, 4, 3): 1.62, (9, 4, 4): 2.72, (9, 4, 5): 22.15,
        (9, 5, 2): 1.58, (9, 5, 3): 3.07, (9, 5, 4): 6.14,
        (9, 6, 2): 2.59, (9, 6, 3): 18.13
    }

    focus_parameters = [
        (9,4,4),
        (9,4,5),
        (9,5,2),
        (9,5,3),
        (9,5,4),
        (9,6,2),
        (9,6,3)
    ]

    best_matrices = {}
    best_heights = {}

    os.makedirs('data', exist_ok=True)

    try:
        with open('data/spherical-gm', 'rb') as f:
            best_matrices = pickle.load(f)
        with open('data/spherical-mh', 'rb') as f:
            best_heights = pickle.load(f)
    except FileNotFoundError:
        print("No previous data found. Starting fresh.")

    while True:
        try:
            for (n, k, m) in focus_parameters:
                clear_cache()
                start_time = time.perf_counter()
                target = ideal_targets.get((n, k, m), float('inf'))
                print(f"\nStarting with  n={n}, k={k}, m={m}; target: {target}")

                if (n, k, m) in best_heights and (n, k, m) in best_matrices:
                    current_best_height = best_heights[(n, k, m)]
                    current_best_P = best_matrices[(n, k, m)]
                    print(f"Starting height: {current_best_height:.4f}")

                    if current_best_height <= target + .0001:
                        print(f"Already hit best mheight")
                        continue
                else:
                    current_best_height = float('inf')
                    current_best_P = None

                print(f"Running genetic search")
                historical_champion = best_matrices.get((n, k, m), None)

                es_best_P, es_best_height = genetic_search(
                    n, k, m,
                    target_height=target,
                    current_best_matrix=historical_champion,           
                    starters=7000,
                    track=100,
                    generations=750,
                    perturbations=200
                )

                if es_best_height < current_best_height:
                    best_matrices[(n, k, m)] = es_best_P
                    best_heights[(n, k, m)] = es_best_height
                    print(f"  SUCCESS! Evolutionary Strategy improved m-height to: {es_best_height:.4f}")
                    
                    try:
                        with open('data/spherical-gm2', 'wb') as f:
                            pickle.dump(best_matrices, f)
                        with open('data/spherical-mh2', 'wb') as f:
                            pickle.dump(best_heights, f)
                        print("  [Checkpoint automatically saved to disk]")
                    except Exception as e:
                        print(f"  [Warning: Failed to save checkpoint: {e}]")

                else:
                    print(f"  Evolutionary Strategy finished. No better matrix found (Best remains {current_best_height:.4f}).")

                end_time = time.perf_counter()
                print(f"  Time elapsed for this case: {(end_time - start_time)/60:.2f} mins\n")
            
            print("#" * 26)
            print("COMPLETED FULL ROUND")
            print("#" * 26)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught! Saving and exiting safely...")
            break

        finally:
            with open('data/spherical-gm2', 'wb') as f:
                pickle.dump(best_matrices, f)
            with open('data/spherical-mh2', 'wb') as f:
                pickle.dump(best_heights, f)