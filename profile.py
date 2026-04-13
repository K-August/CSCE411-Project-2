import numpy as np
import pickle
import time
import fast_evaluator # This imports your new C++ .so file!

# Keep your global cache in Python for maximum speed
mheight_cache = {}

def clear_cache():
    global mheight_cache
    mheight_cache = {}

def get_cached_mHeight(n, k, m, P, threshold=0.95):
    # Flatten the matrix to make it a hashable tuple key
    cache_key = (n, k, m, tuple(P.flatten()))
    
    if cache_key in mheight_cache:
        return mheight_cache[cache_key]
    
    # CALL THE C++ FUNCTION
    # Pybind11 automatically converts your Numpy array 'P' into an Eigen Matrix
    height = fast_evaluator.calc_mHeight_efficient(n, k, m, P, threshold)
    
    mheight_cache[cache_key] = height
    return height

def main():
    print("Loading test files...")
    
    # 1. Open and unpack the pickle files
    try:
        with open('test/sample-n_k_m_P', 'rb') as f:
            inputs = pickle.load(f)
            
        with open('test/sample-mHeights', 'rb') as f:
            expected_mHeights = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure you are running this from the directory containing the 'test' folder.")
        return

    # Ensure the files have the same number of items
    num_cases = min(len(inputs), len(expected_mHeights))
    print(f"Loaded {num_cases} test cases. Starting evaluation...\n")

    passed = 0
    failed = 0
    
    count = 0

    # 2. Start the timer
    start_time = time.perf_counter()
    
    timer_start = time.perf_counter()
    # 3. Iterate through the unpacked data
    for i in range(num_cases):
        count += 1
        if count % 1000 == 0:
            checkpoint_end = time.perf_counter()
            print(f"Processed {count} cases... ({checkpoint_end - timer_start:.2f} seconds)")
            timer_start = time.perf_counter()
        # Unpack the specific inputs
        n, k, m, P = inputs[i]
        expected_h = expected_mHeights[i]
        
        # Ensure P is a numpy array (just in case pickle loaded it as a nested list/tuple)
        P = np.array(P)
        
        # 4. Calculate the mHeight using the C++ module
        # We pass 1.01 for the threshold to strictly disable the spherical spread filter 
        # so we can directly compare the LP solver's raw output.
        calculated_h = fast_evaluator.calc_mHeight_efficient(n, k, m, P, 1.01)
        
        # 5. Compare the output (using np.isclose to handle tiny floating point rounding differences)
        if np.isclose(calculated_h, expected_h, atol=1e-5):
            passed += 1
        else:
            failed += 1
            # Print the first few failures to help with debugging
            if failed <= 5:
                print(f"Mismatch at index {i}: Expected {expected_h:.6f}, Got {calculated_h:.6f} (n={n}, k={k}, m={m})")


    # 6. Stop the timer
    end_time = time.perf_counter()
    
    # Calculate stats
    total_time = end_time - start_time
    evals_per_sec = num_cases / total_time if total_time > 0 else 0
    
    print("\n" + "="*30)
    print("       TEST RESULTS")
    print("="*30)
    print(f"Total Cases : {num_cases}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print("-" * 30)
    print(f"Total Time  : {total_time:.4f} seconds")
    print(f"Speed       : {evals_per_sec:.2f} matrices / second")
    print("="*30)

if __name__ == "__main__":
    main()