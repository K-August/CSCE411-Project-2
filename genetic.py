import pickle
import time
import numpy as np
import itertools
import random
from scipy.optimize import linprog
import os

def is_spherically_spread(G, threshold=0.95):
    """
    Fast geometric check. Returns False if any two columns are too close 
    to being parallel or anti-parallel.
    """
    # 1. Normalize the columns of G to project them onto the unit hypersphere
    norms = np.linalg.norm(G, axis=0)
    # Prevent division by zero if a column is all zeros
    if np.any(norms == 0):
        return False 
    G_norm = G / norms
    
    # 2. Compute the dot products of all pairs of columns (Cosine similarities)
    # This yields an n x n matrix of similarities
    similarities = np.abs(np.dot(G_norm.T, G_norm))
    
    # 3. Ignore the diagonal (a column compared to itself is always 1.0)
    np.fill_diagonal(similarities, 0)
    
    # 4. If the maximum similarity is above the threshold, the columns are too close!
    if np.max(similarities) > threshold:
        return False
        
    return True

# On the Height Profile of Analog Error-Correcting Codes (https://doi.org/10.48550/arXiv.2602.20366)
def calc_mHeight_efficient(n, k, m, P, threshold=0.95):
    # G = [I | P]
    I = np.identity(k)
    G = np.concatenate((I, P), axis=1)

    if not is_spherically_spread(G, threshold):
        return 6969.0

    all_columns = set(range(n))
    h_max = 0.0
    
    for S in itertools.combinations(range(n), m):
        S_bar = list(all_columns - set(S))
    
        # Constraints
        A_all = G[:, S_bar].T 
        A_ub = np.vstack([A_all, -A_all])
        b_ub = np.ones(2 * (n - m))
        
        for i in S:
            c = (-1.0 * G[:, i]).flatten()
        
            res = linprog(
                c, 
                A_ub=A_ub, 
                b_ub=b_ub, 
                bounds=(None, None),
                method='highs',
                options={'disp': False}
            )
            
            if res.success:
                z = -res.fun
                if z > h_max:
                    h_max = z
                    
            elif res.status == 3: 
                return float('inf')
                
    return h_max

# Global cache to store previously evaluated matrices
mheight_cache = {}

def clear_cache():
    global mheight_cache
    mheight_cache = {}

def get_cached_mHeight(n, k, m, P, threshold=0.95):
    # Create a hashable key from the parameters and matrix
    # Flattening the matrix makes it easy to store as a tuple
    cache_key = (n, k, m, tuple(P.flatten()))
    
    if cache_key in mheight_cache:
        return mheight_cache[cache_key]
    
    # If not in cache, calculate it and store it
    height = calc_mHeight_efficient(n, k, m, P, threshold)
    mheight_cache[cache_key] = height
    return height

# On the Height Profile of Analog Error-Correcting Codes (https://doi.org/10.48550/arXiv.2602.20366)
def genetic_search(n, k, m, target_height, current_best_matrix=None, starters=100, track=10, generations=50, perturbations=20):
    cols = n - k
    
    # Proxy for starting out, disable once fintuning
    # proxy_m = m - 1 if m >= 4 else m
    proxy_m = m
    
    population = []
    
    if current_best_matrix is not None:
        # UPDATED: Use cached evaluator
        best_proxy_score = get_cached_mHeight(n, k, proxy_m, current_best_matrix)
        population.append((current_best_matrix.copy(), best_proxy_score)) # seed with current best
    
    count = 0
    # Generate lots of starting matrices with elements [-100, 100]
    for _ in range(starters - 1):
        base_col = np.random.randint(-100, 101, size=k)
        G = np.zeros((k, cols), dtype=int)
        for j in range(cols):
            G[:, j] = np.random.permutation(base_col)
            
        # discard if any column is zero vect
        if np.any(np.sum(np.abs(G), axis=0) == 0):
            continue
            
        # UPDATED: Use cached evaluator
        height = get_cached_mHeight(n, k, proxy_m, G, 0.93)
        if height == float('inf') or height == 6969.0:
            count += 1
            continue 

        population.append((G, height))
        
    print(f"Generated {len(population)} valid starting matrices (discarded {count} candidates)")
    
    # Sort and keep only the absolute best N matrices
    population.sort(key=lambda x: x[1])
    S = [item[0] for item in population[:track]]
    best_proxy_height = population[0][1]
    
    print(f"Starting evolutionary part")
    
    for gen in range(generations):  
        Snew = []
        
        # ==========================================
        # NEW: Crossover Phase
        # ==========================================
        crossover_count = perturbations // 2
        for _ in range(crossover_count):
            if len(S) < 2:
                break
                
            # Pick two random parents from the best matrices
            parent1, parent2 = random.sample(S, 2)
            child = np.zeros_like(parent1)
            
            # Uniform crossover by column
            for c in range(cols):
                if random.random() > 0.5:
                    child[:, c] = parent1[:, c]
                else:
                    child[:, c] = parent2[:, c]
            
            # Discard if any column is zero vect
            if np.any(np.sum(np.abs(child), axis=0) == 0):
                continue
                
            # Add child to Snew (it will be evaluated and scored at the end of the generation)
            Snew.append(child)

        # ==========================================
        # EXISTING: Mutation & Greedy Slide Phase
        # ==========================================
        threshold = 0.93
        for _ in range(perturbations):
            # Step 1: Uniformly randomly select a generator matrix G from S
            G = random.choice(S).copy()
            
            # UPDATED: Use cached evaluator
            current_height = get_cached_mHeight(n, k, proxy_m, G, threshold)
            
            # Step 2: Generate a random perturbation direction (P)
            P = np.zeros((k, cols), dtype=int)
            row = random.randint(0, k - 1)
            col = random.randint(0, cols - 1)
            P[row, col] = random.choice([-1, 1]) 
            
            sigma = 1
            Gnew = np.clip(G + sigma * P, -5, 5)
            
            if np.any(np.sum(np.abs(Gnew), axis=0) == 0):
                continue
                
            # UPDATED: Use cached evaluator
            new_height = get_cached_mHeight(n, k, proxy_m, Gnew, threshold)
            
            # greedy slide
            while new_height < current_height:
                # UPDATED: Use cached evaluator
                true_height_check = get_cached_mHeight(n, k, m, Gnew, threshold)
                if true_height_check == float('inf'):
                    break 
                
                Snew.append(Gnew.copy())
                G = Gnew.copy()
                current_height = new_height
                
                Gnew = np.clip(G + sigma * P, -100, 100)
                
                if np.any(np.sum(np.abs(Gnew), axis=0) == 0):
                    break
                    
                # UPDATED: Use cached evaluator
                new_height = get_cached_mHeight(n, k, proxy_m, Gnew, threshold)

        # Combine the old best matrices with the new mutated/crossover ones
        combined = S + Snew
        scored = []
        for mat in combined:
            # UPDATED: Use cached evaluator
            h = get_cached_mHeight(n, k, proxy_m, mat, threshold)
            scored.append((mat, h))
            
        # Sort and keep the top N
        scored.sort(key=lambda x: x[1])
        
        # Remove duplicates
        unique_S = []
        seen = []
        for mat, h in scored:
            mat_tuple = tuple(mat.flatten())
            if mat_tuple not in seen:
                seen.append(mat_tuple)
                unique_S.append(mat)
            if len(unique_S) == track:
                break
                
        S = unique_S
        
        if scored[0][1] < best_proxy_height:
            best_proxy_height = scored[0][1]
            print(f"Gen {gen}: height pushed down to {best_proxy_height:.4f}")
            

    best_G = S[0]
    
    # UPDATED: Use cached evaluator
    final_target_height = get_cached_mHeight(n, k, m, best_G, threshold)
    print(f"Final m={m} height: {final_target_height:.4f} (Target: {target_height})")
    
    return best_G, final_target_height


if __name__ == '__main__':
    target_parameters = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3)
    ]

    # min mHeights to aim for
    ideal_targets = {
        (9, 4, 2): 1.17,
        (9, 4, 3): 1.62,  
        (9, 4, 4): 2.72,  
        (9, 4, 5): 22.15,
        (9, 5, 2): 1.58,
        (9, 5, 3): 3.07,
        (9, 5, 4): 6.14,
        (9, 6, 2): 2.59,
        (9, 6, 3): 18.13
    }

    # perform algorithm on these
    focus_parameters = [
        (9,4,3),
        (9,4,4),
        (9,5,2),
        (9,5,3),
        (9,5,4),
        (9,6,2),
        (9,6,3),
    ]

    best_matrices = {}
    best_heights = {}

    # Make sure the data directory actually exists before trying to read/write to it!
    os.makedirs('data', exist_ok=True)

    # Load existing data if available
    try:
        with open('out/1m', 'rb') as f:
            best_matrices = pickle.load(f)
        with open('out/1g', 'rb') as f:
            best_heights = pickle.load(f)
    except FileNotFoundError:
        print("No previous data found. Starting fresh.")

    # infinte loop to run over night, ctrl+c stops and saves
    while True:
        try:
            for (n, k, m) in target_parameters:
                clear_cache()
                start_time = time.perf_counter()
                target = ideal_targets.get((n, k, m), float('inf'))
                print(f"\nStarting with  n={n}, k={k}, m={m}; target: {target}")

                # Pull the baseline from the loaded dictionary (if it exists)
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
                    starters=8000,
                    track=60,
                    generations=225,
                    perturbations=80
                )

                # After the ES finishes, check if it actually beat our saved checkpoint
                if es_best_height < current_best_height:
                    best_matrices[(n, k, m)] = es_best_P
                    best_heights[(n, k, m)] = es_best_height
                    print(f"  SUCCESS! Evolutionary Strategy improved m-height to: {es_best_height:.4f}")
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
            with open('out/3m', 'wb') as f:
                pickle.dump(best_matrices, f)

            with open('out/3g', 'wb') as f:
                pickle.dump(best_heights, f)