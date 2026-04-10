import pickle
import time
import numpy as np
import random
import math
import os
import fast_evaluator

# ==========================================
# CACHING & FAST EVALUATOR
# ==========================================
mheight_cache = {}

def clear_cache():
    global mheight_cache
    mheight_cache = {}

def get_cached_mHeight(n, k, m, P, threshold=0.935):
    cache_key = (n, k, m, tuple(P.flatten()))
    if cache_key in mheight_cache:
        return mheight_cache[cache_key]
    
    height = fast_evaluator.calc_mHeight_efficient(n, k, m, P, threshold)
    mheight_cache[cache_key] = height
    return height

# ==========================================
# SIMULATED ANNEALING SEARCH
# ==========================================
def simulated_annealing_search_integer(n, k, m, target_height, initial_P=None, initial_height=float('inf'), max_iterations=100000, initial_temp=5.0, cooling_rate=0.99):
    if initial_P is not None:
        current_P = initial_P.copy()
        current_height = initial_height
    else:
        # Generate a targeted random starting matrix
        if (n, k, m) == (9, 6, 3):
            current_P = np.random.randint(-3, 4, size=(k, n - k))
        else:
            current_P = np.random.randint(-10, 11, size=(k, n - k))
            
        if (n, k, m) in [(9, 4, 4), (9, 6, 3)]:
            current_P[current_P == 0] = random.choice([-1, 1])
            
        current_height = get_cached_mHeight(n, k, m, current_P)
        
    best_P = current_P.copy()
    best_height = current_height
    temperature = initial_temp
    
    # --- NEW: CALCULATE PERFECT THERMODYNAMICS ---
    # We want the temperature to slowly decay to 0.01 exactly as the loop finishes
    target_final_temp = 0.01
    dynamic_cooling_rate = (target_final_temp / initial_temp) ** (1 / max_iterations)
    
    for i in range(max_iterations):
        neighbor_P = current_P.copy()
        
        # --- NEW: MULTI-CELL MUTATION ---
        # 15% chance to tweak TWO cells at once to leap out of tight craters
        num_tweaks = 2 if random.random() < 0.15 else 1
        
        for _ in range(num_tweaks):
            row = random.randint(0, k - 1)
            col = random.randint(0, (n - k) - 1)
            
            # Tweak by +1, -1, +2, or -2
            change = random.choice([-2, -1, 1, 2])
            
            # Apply the tweak with slightly relaxed bounds for breathing room
            if (n, k, m) == (9, 6, 3):
                neighbor_P[row, col] = np.clip(neighbor_P[row, col] + change, -4, 4) 
            else:
                neighbor_P[row, col] = np.clip(neighbor_P[row, col] + change, -15, 15)
                
            # Apply Zero-Ban
            if (n, k, m) in [(9, 4, 4), (9, 6, 3)]:
                if neighbor_P[row, col] == 0:
                    neighbor_P[row, col] = random.choice([-1, 1])
        
        # Ensure no all-zero columns exist
        if np.any(np.sum(np.abs(neighbor_P), axis=0) == 0):
            continue 
            
        neighbor_height = get_cached_mHeight(n, k, m, neighbor_P)
        
        if neighbor_height < current_height:
            current_P = neighbor_P
            current_height = neighbor_height
            
            if current_height < best_height:
                best_height = current_height
                best_P = current_P.copy()

                if best_height <= target_height + 1e-5:
                    print(f"      *** PERFECT SCORE REACHED at iteration {i}! Early Stopping. ***")
                    break 
        else:
            try:
                diff = current_height - neighbor_height
                
                # --- NEW: CRASH PREVENTION ---
                # Safely floor the temperature so it NEVER divides by true 0.0
                safe_temp = max(temperature, 1e-10) 
                probability_of_acceptance = math.exp(diff / safe_temp)
                
            except (OverflowError, ZeroDivisionError):
                probability_of_acceptance = 0.0
                
            if random.random() < probability_of_acceptance:
                current_P = neighbor_P
                current_height = neighbor_height
                
        # Apply the mathematically perfect cooling rate
        temperature *= dynamic_cooling_rate
        
    return best_P, best_height
        

# ==========================================
# MAIN EXECUTION ENGINE
# ==========================================
if __name__ == '__main__':
    ideal_targets = {
        (9, 4, 2): 1.17, (9, 4, 3): 1.62, (9, 4, 4): 2.72, (9, 4, 5): 22.15,
        (9, 5, 2): 1.58, (9, 5, 3): 3.07, (9, 5, 4): 6.14,
        (9, 6, 2): 2.59, (9, 6, 3): 18.13
    }

    # Focus parameters for the weekend run
    focus_parameters = [
        (9, 4, 4), (9, 4, 5), (9, 5, 2), (9, 5, 3), (9, 5, 4), (9, 6, 2), (9, 6, 3)
    ]

    best_matrices = {}
    best_heights = {}

    os.makedirs('data', exist_ok=True)

    print("Starting the Modern Annealing Engine...\n")

    # Load existing data if available
    try:
        with open('data/spherical-gm3', 'rb') as f:
            best_matrices = pickle.load(f)
        with open('data/spherical-mh3', 'rb') as f:
            best_heights = pickle.load(f)
        print("Successfully loaded previous data!\n")
    except FileNotFoundError:
        print("No previous data found. Starting from scratch...\n")

    # Infinite Loop (Ctrl+C to stop)
    while True:
        try:
            for (n, k, m) in focus_parameters:
                clear_cache() # Prevent memory bloat over the weekend
                start_time = time.perf_counter()
                target = ideal_targets[(n, k, m)]
                print(f"--- Optimizing matrix for n={n}, k={k}, m={m} | TARGET: {target} ---")
                
                if (n, k, m) in best_heights and (n, k, m) in best_matrices:
                    current_best_height = best_heights[(n, k, m)]
                    current_best_P = best_matrices[(n, k, m)]
                    print(f"  Loaded baseline. Starting height: {current_best_height:.4f}")

                    if current_best_height <= target + 1e-5:
                        print(f"  Target already achieved! Skipping to next case.\n")
                        continue 
                else:
                    current_best_height = float('inf')
                    current_best_P = None
                    print("  No previous baseline. Starting from scratch.")

                # Monte Carlo Drop for stuck matrices
                if current_best_height > 30.0:
                    print(f"  Baseline is rough ({current_best_height:.4f}). Performing Fast Monte Carlo Drop...")
                    best_random_height = float('inf')
                    best_random_P = None
                    
                    for _ in range(1000):
                        if (n, k, m) == (9, 6, 3):
                            test_P = np.random.randint(-3, 4, size=(k, n - k))
                        else:
                            test_P = np.random.randint(-6, 7, size=(k, n - k))
                            
                        if (n, k, m) in [(9, 4, 4), (9, 6, 3)]:
                            test_P[test_P == 0] = random.choice([-1, 1])
                            
                        # Ensure no zero columns
                        if np.any(np.sum(np.abs(test_P), axis=0) == 0):
                            continue
                            
                        test_height = get_cached_mHeight(n, k, m, test_P)
                        
                        if test_height < best_random_height:
                            best_random_height = test_height
                            best_random_P = test_P.copy()
                    
                    if best_random_height < current_best_height:
                        current_best_height = best_random_height
                        current_best_P = best_random_P
                        print(f"  Monte Carlo Drop succeeded! Best starting point: {current_best_height:.4f}")

                # Dynamic Temperature Setup
                error_distance = current_best_height - target
                if error_distance > 10.0 or current_best_height == float('inf'):
                    dynamic_temp = 5  
                elif error_distance > 3.0:
                    dynamic_temp = 2.5   
                else:
                    dynamic_temp = 1   

                print(f"  Running Simulated Annealing (Temp: {dynamic_temp})...")
                
                new_best_P, new_best_height = simulated_annealing_search_integer(
                    n, k, m,
                    target_height=target,           
                    initial_P=current_best_P,
                    initial_height=current_best_height,
                    max_iterations=200000,  # Huge iterations because C++ is so fast
                    initial_temp=dynamic_temp,      
                    cooling_rate=0.999,
                )

                if new_best_height < current_best_height:
                    best_matrices[(n, k, m)] = new_best_P
                    best_heights[(n, k, m)] = new_best_height
                    print(f"  SUCCESS! Annealing improved m-height to: {new_best_height:.4f}")
                    
                    # --- FAILSAFE DISK SAVE ---
                    try:
                        with open('data/annealing-gm', 'wb') as f:
                            pickle.dump(best_matrices, f)
                        with open('data/annealing-mh', 'wb') as f:
                            pickle.dump(best_heights, f)
                        print("  [Checkpoint automatically saved to disk]")
                    except Exception as e:
                        print(f"  [Warning: Failed to save checkpoint: {e}]")
                else:
                    print(f"  Annealing finished. No better matrix found (Best remains {current_best_height:.4f}).")

                end_time = time.perf_counter()
                print(f"  Time elapsed for this case: {end_time - start_time:.2f} seconds\n")
            
            print("#" * 35)
            print("COMPLETED FULL ROUND - RESTARTING")
            print("#" * 35 + "\n")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught! Saving and exiting safely...")
            break

        finally:
            with open('data/annealing-gm', 'wb') as f:
                pickle.dump(best_matrices, f)
            with open('data/annealing-mh', 'wb') as f:
                pickle.dump(best_heights, f)