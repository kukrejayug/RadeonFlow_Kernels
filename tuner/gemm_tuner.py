from __future__ import annotations
import os
import subprocess
import re
import threading
import datetime
import time
import logging
import csv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Dict, List, Optional

tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_dir')
output_dir = os.path.join(os.path.dirname(__file__), 'output_dir')
# Create tmp_dir if it doesn't exist
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Setup a function to get a logger for a specific M,N,K configuration
def get_logger(M=None, N=None, K=None):
    # Clear any existing handlers from the logger
    logger = logging.getLogger('gemm_tuner')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up the logger
    logger.setLevel(logging.DEBUG)
    
    # Define the log file path based on M,N,K if provided
    if M is not None and N is not None and K is not None:
        log_file_path = os.path.join(output_dir, f'gemm_{M}_{N}_{K}.log')
    else:
        log_file_path = os.path.join(output_dir, 'gemm_tuner.log')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add only file handler to logger (no console handler)
    logger.addHandler(file_handler)
    
    # Prevent logs from propagating to the root logger
    logger.propagate = False
    
    return logger

# Initialize a default logger
logger = get_logger()

# Function to log operation with parameters
def log_operation(operation_type: str, params: Dict, messages: List[str], success: bool = True):
    """
    Log an operation (compilation or execution) with its parameters and results
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the parameter string
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    
    # Create header and footer
    header = f"===== {operation_type} {param_str} ===== [{timestamp}]"
    status = "SUCCESS" if success else "FAILED"
    footer = f"===== {operation_type} {status} ====="
    
    # Log header
    logger.info(header)
    
    # Log each message
    for msg in messages:
        if msg:  # Only log non-empty messages
            logger.info(msg)
    
    # Log footer
    logger.info(footer)
    logger.info("")  # Empty line for readability

def get_gemm_kernel(M: int, N: int, K: int, BM: int, BN: int, BK: int, QUANT_SIZE: int, BLOCK_SIZE: int,
                    WARP_M: int, WARP_N: int, SPLITK_FACTOR: int, LOAD_BATCH_SIZE: int) -> str:
    """Generates and compiles a GEMM kernel with the specified parameters"""
    
    # Collect logs for this operation
    log_messages = []
    log_messages.append(f"Starting GEMM kernel generation and compilation")
    
    # Parameters for logging
    params = {
        'M': M, 'N': N, 'K': K, 
        'BM': BM, 'BN': BN, 'BK': BK,
        'QUANT_SIZE': QUANT_SIZE, 'BLOCK_SIZE': BLOCK_SIZE,
        'WARP_M': WARP_M, 'WARP_N': WARP_N, 
        'SPLITK_FACTOR': SPLITK_FACTOR, 'LOAD_BATCH_SIZE': LOAD_BATCH_SIZE
    }
    
    # Get the full path to the template file
    template_path = os.path.join(os.path.dirname(__file__), 'template', 'gemm_perf.cpp')
    log_messages.append(f"Using template file: {template_path}")
    
    try:
        with open(template_path) as f:
            template_lines = f.readlines()
    except Exception as e:
        log_messages.append(f"Failed to read template file: {e}")
        log_operation("Compile", params, log_messages, success=False)
        return None
    
    # Find the begin and end of parameterization section
    begin_idx = -1
    end_idx = -1
    for i, line in enumerate(template_lines):
        if "// Begin parameterization" in line:
            begin_idx = i
        elif "// End parameterization" in line:
            end_idx = i
            break
    
    if begin_idx == -1 or end_idx == -1:
        log_messages.append("Could not find parameterization markers in template file")
        log_operation("Compile", params, log_messages, success=False)
        return None
    
    # Create new parameter block
    new_params = [
        "// Begin parameterization\n",
        f"constexpr int M = {M} /* param M*/, N = {N} /* param N*/ , K = {K} /* param K */;\n",
        f"constexpr int BM = {BM} /* param BM */, BN = {BN} /* param BN */, BK = {BK} /* param BK */;\n",
        f"constexpr int QUANT_SIZE = {QUANT_SIZE} /* param QUANT_SIZE */, BLOCK_SIZE = {BLOCK_SIZE} /* param BLOCK_SIZE */;\n",
        f"constexpr int SPLITK_FACTOR = {SPLITK_FACTOR} /* param SPLITK_FACTOR */;\n",
        f"constexpr int LOAD_BATCH_SIZE = {LOAD_BATCH_SIZE} /* param LOAD_BATCH_SIZE */;\n",
        "#ifdef TEST_ON_RDNA4 // RDNA4, WAVE_SIZE = 32\n",
        f"constexpr int WARP_M = {WARP_M} /* param RNDA4_WARP_M */, WARP_N = {WARP_N} /* param WARP_N */;\n",
        "#else // CDNA3, WAVE_SIZE = 64\n",
        f"constexpr int WARP_M = {WARP_M} /* param CDNA3_WARP_M */, WARP_N = {WARP_N} /* param CDNA_WARP_N */;\n",
        "#endif\n",
        "// End parameterization\n"
    ]
    
    # Replace the parameterization section
    modified_template = template_lines[:begin_idx] + new_params + template_lines[end_idx+1:]
    
    # Write the modified template to the correct path
    cpp_path = os.path.join(tmp_dir, f'gemm_perf_{M}_{N}_{K}_{BM}_{BN}_{BK}_{QUANT_SIZE}_{BLOCK_SIZE}_{WARP_M}_{WARP_N}_{SPLITK_FACTOR}_{LOAD_BATCH_SIZE}.cpp')
    log_messages.append(f"Writing modified template to: {cpp_path}")
    
    try:
        with open(cpp_path, 'w') as f:
            f.writelines(modified_template)
    except Exception as e:
        log_messages.append(f"Failed to write to file: {e}")
        log_operation("Compile", params, log_messages, success=False)
        return None
    
    # Compile the generated file
    binary_path = os.path.join(tmp_dir, f'gemm_perf_{M}_{N}_{K}_{BM}_{BN}_{BK}_{QUANT_SIZE}_{BLOCK_SIZE}_{WARP_M}_{WARP_N}_{SPLITK_FACTOR}_{LOAD_BATCH_SIZE}')
    # compile_cmd = f"hipcc -std=c++17 -O3 {cpp_path} -DTEST_ON_HIP -DTEST_ON_RDNA4=1 --offload-arch=gfx1201 -o {binary_path}"
    compile_cmd = f"hipcc -std=c++17 -O3 {cpp_path} -DTEST_ON_HIP --offload-arch=gfx942 -o {binary_path}"
    
    log_messages.append(f"Compiling with command: {compile_cmd}")
    
    try:
        process = subprocess.run(compile_cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True)
        log_messages.append("Compilation stdout:")
        log_messages.append(process.stdout)
        log_messages.append("Compilation stderr:")
        log_messages.append(process.stderr)
        log_messages.append(f"Successfully compiled: {binary_path}")
        
        # Delete the source file after successful compilation
        try:
            os.remove(cpp_path)
            log_messages.append(f"Deleted source file: {cpp_path}")
        except Exception as e:
            log_messages.append(f"Warning: Failed to delete source file {cpp_path}: {e}")
        log_operation("Compile", params, log_messages, success=True)
            
    except subprocess.CalledProcessError as e:
        log_messages.append(f"Compilation failed: {e}")
        log_messages.append(f"Stdout: {e.stdout}")
        log_messages.append(f"Stderr: {e.stderr}")
        log_operation("Compile", params, log_messages, success=False)
        return None
    
    return binary_path

def execute_gemm_kernel(binary_path: str, params: Dict = None) -> float:
    """
    Execute the compiled GEMM kernel binary and return its execution time in milliseconds.
    
    Args:
        binary_path: Path to the compiled binary
        params: Dictionary of parameters for logging purposes
        
    Returns:
        float: Execution time in milliseconds, or None if execution failed
    """
    log_messages = []
    log_messages.append(f"Executing GEMM kernel: {binary_path}")
    
    if params is None:
        # Try to extract parameters from binary name
        try:
            basename = os.path.basename(binary_path)
            parts = basename.replace('gemm_perf_', '').split('_')
            if len(parts) >= 10:
                params = {
                    'M': parts[0], 'N': parts[1], 'K': parts[2],
                    'BM': parts[3], 'BN': parts[4], 'BK': parts[5],
                    'QUANT_SIZE': parts[6], 'BLOCK_SIZE': parts[7],
                    'WARP_M': parts[8], 'WARP_N': parts[9], 'SPLITK_FACTOR': parts[10]
                }
            else:
                params = {'binary': basename}
        except:
            params = {'binary': os.path.basename(binary_path)}
    
    if not os.path.exists(binary_path):
        log_messages.append(f"Error: Binary file {binary_path} does not exist")
        log_operation("Execute", params, log_messages, success=False)
        return None
    
    try:
        # Run the compiled binary and capture stderr where the time is written
        log_messages.append(f"Running command: {binary_path}")
        
        start_time = time.time()
        result = subprocess.run(binary_path, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True)
        end_time = time.time()
        
        log_messages.append(f"Total process execution time: {(end_time - start_time)*1000:.2f} ms")
        log_messages.append("Process stdout:")
        log_messages.append(result.stdout)
        
        # The execution time is printed to stderr as a float
        if result.stderr:
            time_str = result.stderr.strip()
            # Don't log stderr when successful, just extract the time
            try:
                execution_time = float(time_str)
                log_messages.append(f"Parsed execution time: {execution_time} ms")
                log_operation("Execute", params, log_messages, success=True)
                
                # Delete the binary file after successful execution
                try:
                    os.remove(binary_path)
                    log_messages.append(f"Deleted binary file: {binary_path}")
                except Exception as e:
                    log_messages.append(f"Warning: Failed to delete binary file {binary_path}: {e}")
                
                return execution_time  # This is already in milliseconds
            except ValueError:
                # Only log stderr when parsing fails
                log_messages.append(f"Failed to parse execution time from stderr: {time_str}")
                log_operation("Execute", params, log_messages, success=False)
                return None
        else:
            log_messages.append("No output captured from stderr")
            log_operation("Execute", params, log_messages, success=False)
            return None
            
    except subprocess.CalledProcessError as e:
        log_messages.append(f"Execution failed with error code {e.returncode}: {e}")
        log_messages.append(f"Output: {e.stdout}")
        log_messages.append(f"Error: {e.stderr}")
        log_operation("Execute", params, log_messages, success=False)
        
        # In case of execution failure, also try to clean up the binary
        try:
            os.remove(binary_path)
            log_messages.append(f"Deleted binary file after execution failure: {binary_path}")
        except Exception as clean_error:
            log_messages.append(f"Warning: Failed to delete binary file {binary_path}: {clean_error}")
            
        return None

def get_valid_powers_of_two(min_val, max_val, upper_limit=None):
    """Get all valid powers of 2 within the specified range and not exceeding upper_limit."""
    powers = []
    power = 1
    while power <= max_val:
        if power >= min_val and (upper_limit is None or power <= upper_limit):
            powers.append(power)
        power *= 2
    return powers

def compile_kernel_worker(config):
    """Worker function for parallel compilation of kernels"""
    M = config['M']
    N = config['N']
    K = config['K']
    BM = config['BM']
    BN = config['BN']
    BK = config['BK']
    QUANT_SIZE = config['QUANT_SIZE']
    BLOCK_SIZE = config['BLOCK_SIZE']
    WARP_M = config['WARP_M']
    WARP_N = config['WARP_N']
    SPLITK_FACTOR = config['SPLITK_FACTOR']
    LOAD_BATCH_SIZE = config['LOAD_BATCH_SIZE']
    
    # Compile the kernel
    binary_path = get_gemm_kernel(M, N, K, BM, BN, BK, QUANT_SIZE, BLOCK_SIZE, WARP_M, WARP_N, SPLITK_FACTOR, LOAD_BATCH_SIZE)
    
    # Return the config and binary path if successful, None otherwise
    if binary_path:
        return (config, binary_path)
    return (config, None)

def tune_gemm_kernel(M, N, K, batch_size=100, max_workers=min(os.cpu_count(), 40)):
    """
    Tune GEMM kernel parameters for best performance with given M, N, K.
    Uses parallel compilation for batches of configurations.
    
    Args:
        M, N, K: Matrix dimensions
        batch_size: Number of configurations to compile in parallel
        max_workers: Maximum number of worker threads (None = auto)
        
    Returns:
        tuple: Best configuration parameters and execution time
    """
    # Get a logger specific to this M,N,K configuration
    global logger
    logger = get_logger(M, N, K)
    logger.info(f"Starting GEMM kernel tuning for M={M}, N={N}, K={K}")
    
    # Fixed values
    QUANT_SIZE = 128
    WARP_SIZE = 64
    # Use WMMA_M and WMMA_N as 32 as specified
    WMMA_M = 32
    WMMA_N = 32
    
    # Generate valid parameter values according to constraints
    valid_BM = get_valid_powers_of_two(128,256, M)
    valid_BN = get_valid_powers_of_two(128,128, N)
    valid_BK = get_valid_powers_of_two(64, 256, 256)
    valid_BLOCK_SIZE = get_valid_powers_of_two(512, 512, 1024)
    # Only include 8 and 16 for LOAD_BATCH_SIZE
    valid_LOAD_BATCH_SIZE = [16]
    
    # if K == 256 or K == 2304:
    #     valid_SPLITK_FACTOR = [1, 2,]
    # else:
    #     valid_SPLITK_FACTOR = [1, 2, 4, 8]
    valid_SPLITK_FACTOR = [1, 2, 4, 8]  # Allowing up to 16 for more flexibility
    
    # First, get all WARP_M, WARP_N combinations for each BLOCK_SIZE
    block_size_to_warp_combinations = {}
    for BLOCK_SIZE in valid_BLOCK_SIZE:
        valid_warp_combinations = []
        for WARP_M in get_valid_powers_of_two(1, 16):
            for WARP_N in get_valid_powers_of_two(1, 16):
                # Constraint 1: WARP_M * WARP_N * WARP_SIZE == BLOCK_SIZE
                if WARP_M * WARP_N * WARP_SIZE == BLOCK_SIZE:
                    valid_warp_combinations.append((WARP_M, WARP_N))
        
        if valid_warp_combinations:  # Only store block sizes with valid combinations
            block_size_to_warp_combinations[BLOCK_SIZE] = valid_warp_combinations
    
    # Generate all possible combinations
    all_combinations = []
    for BLOCK_SIZE, warp_combinations in block_size_to_warp_combinations.items():
        for BM in valid_BM:
            for BN in valid_BN:
                for BK in valid_BK:
                    for WARP_M, WARP_N in warp_combinations:
                        for SPLITK_FACTOR in valid_SPLITK_FACTOR:
                            for LOAD_BATCH_SIZE in valid_LOAD_BATCH_SIZE:
                                all_combinations.append({
                                    'M': M, 'N': N, 'K': K,
                                    'BM': BM, 'BN': BN, 'BK': BK,
                                    'QUANT_SIZE': QUANT_SIZE, 'BLOCK_SIZE': BLOCK_SIZE,
                                    'WARP_M': WARP_M, 'WARP_N': WARP_N,
                                    'SPLITK_FACTOR': SPLITK_FACTOR,
                                    'LOAD_BATCH_SIZE': LOAD_BATCH_SIZE
                                })
    
    # Now filter the combinations based on our constraints
    valid_combinations = []
    for config in all_combinations:
        BM = config['BM']
        BN = config['BN']
        BK = config['BK']
        BLOCK_SIZE = config['BLOCK_SIZE']
        WARP_M = config['WARP_M']
        WARP_N = config['WARP_N']
        LOAD_BATCH_SIZE = config['LOAD_BATCH_SIZE']
        
        # Check constraint 2: (BK * BM) % (BLOCK_SIZE * LOAD_BATCH_SIZE) == 0
        if (BK * BM) % (BLOCK_SIZE * LOAD_BATCH_SIZE) != 0:
            continue
        
        # Check constraint 3: BK % LOAD_BATCH_SIZE == 0
        if BK % LOAD_BATCH_SIZE != 0:
            continue
        
        # Check constraint 4: BM % (WMMA_M * WARP_M) == 0
        if BM % (WMMA_M * WARP_M) != 0:
            continue
            
        # Check constraint 5: BN % (WMMA_N * WARP_N) == 0
        if BN % (WMMA_N * WARP_N) != 0:
            continue
            
        # All constraints passed, this is a valid configuration
        valid_combinations.append(config)
    
    # Log the filtering results
    logger.info(f"Generated {len(all_combinations)} total configurations")
    logger.info(f"Found {len(valid_combinations)} valid configurations after filtering")
    print(f"Generated {len(all_combinations)} total configurations")
    print(f"Found {len(valid_combinations)} valid configurations after filtering")
    
    configs = valid_combinations
    
    # Setup CSV file for recording results
    csv_file_path = os.path.join(output_dir, f'gemm_{M}_{N}_{K}.csv')
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Define CSV columns
        fieldnames = ['M', 'N', 'K', 'BM', 'BN', 'BK', 'QUANT_SIZE', 'BLOCK_SIZE', 
                      'WARP_M', 'WARP_N', 'SPLITK_FACTOR', 'LOAD_BATCH_SIZE', 'execution_time_us', 'tflops']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        # Store best configuration and its execution time
        best_config = None
        best_time = float('inf')
        all_results = []
        total_configs = len(configs)
        configs_tested = 0
        
        # Split configs into batches
        num_batches = ceil(len(configs) / batch_size)
        print(f"Starting GEMM tuning for M={M}, N={N}, K={K}")
        print(f"Total configurations to try: {total_configs}")
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(configs))
            batch = configs[start_idx:end_idx]
            
            # Log batch info to file, not console
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch)} configurations")
            
            # Update console with batch progress (using carriage return)
            print(f"\rProcessing batch {batch_idx+1}/{num_batches}... ", end="")
            
            # Compile all kernels in the batch in parallel
            compiled_kernels = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {executor.submit(compile_kernel_worker, config): config for config in batch}
                for future in concurrent.futures.as_completed(future_to_config):
                    config, binary_path = future.result()
                    if binary_path:
                        compiled_kernels.append((config, binary_path))
            
            # Log compilation results to file
            logger.info(f"Successfully compiled {len(compiled_kernels)}/{len(batch)} kernels in batch {batch_idx+1}")
            
            # Execute each successfully compiled kernel sequentially
            for config, binary_path in compiled_kernels:
                # Extract parameters for readability
                BM = config['BM']
                BN = config['BN']
                BK = config['BK']
                BLOCK_SIZE = config['BLOCK_SIZE']
                WARP_M = config['WARP_M']
                WARP_N = config['WARP_N']
                SPLITK_FACTOR = config['SPLITK_FACTOR']
                
                execution_time_ms = execute_gemm_kernel(binary_path, config)
                configs_tested += 1
                
                if execution_time_ms is not None:
                    # Convert from milliseconds to microseconds
                    execution_time_us = execution_time_ms * 1000
                    
                    # Calculate TFLOPS: (2*M*N*K) / (execution_time_seconds * 10^12)
                    # First convert milliseconds to seconds
                    execution_time_s = execution_time_ms / 1000
                    flops = 2 * M * N * K  # Number of floating-point operations
                    tflops = flops / (execution_time_s * 10**12)
                    
                    config_str = f"BM={BM}, BN={BN}, BK={BK}, BLOCK_SIZE={BLOCK_SIZE}, WARP_M={WARP_M}, WARP_N={WARP_N}, SPLITK_FACTOR={SPLITK_FACTOR}"
                    
                    # Log detailed result to file only
                    logger.info(f"Config: {config_str} - Time: {execution_time_us:.2f} µs, TFLOPS: {tflops:.2f}")
                    
                    # Update console with progress and current best (using carriage return)
                    best_info = f"Best: {best_time:.2f}µs ({best_config['tflops']:.2f} TFLOPS)" if best_config else "No valid config yet"
                    print(f"\rProgress: {configs_tested}/{total_configs} configs tested. {best_info}", end="")
                    
                    # Store result and write to CSV
                    result = {
                        **config,
                        'execution_time_us': execution_time_us,
                        'tflops': tflops
                    }
                    all_results.append(result)
                    csv_writer.writerow(result)
                    csv_file.flush()  # Ensure data is written immediately
                    
                    # Update best configuration if this is better
                    if execution_time_ms < best_time:
                        best_time = execution_time_ms
                        best_config = config.copy()
                        best_config['execution_time_us'] = execution_time_us
                        best_config['tflops'] = tflops
                        
                        # Log new best to file
                        logger.info(f"New best: {config_str} - Time: {execution_time_us:.2f} µs, TFLOPS: {tflops:.2f}")
                        
                        # Update console with new best (using carriage return)
                        print(f"\rNew best config: {config_str} - Time: {execution_time_us:.2f} µs, TFLOPS: {tflops:.2f}", end="")
                        # Wait a moment to ensure the best config is visible
                        time.sleep(0.5)
                        # Return to progress display
                        print(f"\rProgress: {configs_tested}/{total_configs} configs tested. Best: {execution_time_us:.2f}µs ({tflops:.2f} TFLOPS)", end="")
        
        # Save all results to a JSON file for additional processing if needed
        import json
        results_file = os.path.join(output_dir, f'gemm_tuner_results_M{M}_N{N}_K{K}.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print final newline to finish the carriage return line
        print()
        
        if best_config:
            print("\nBest configuration found:")
            config_str = f"BM={best_config['BM']}, BN={best_config['BN']}, BK={best_config['BK']}, BLOCK_SIZE={best_config['BLOCK_SIZE']}, WARP_M={best_config['WARP_M']}, WARP_N={best_config['WARP_N']}, SPLITK_FACTOR={best_config['SPLITK_FACTOR']}, LOAD_BATCH_SIZE={best_config['LOAD_BATCH_SIZE']}"
            print(f"  {config_str}")
            print(f"  Execution time: {best_config['execution_time_us']:.2f} µs")
            print(f"  Performance: {best_config['tflops']:.2f} TFLOPS")
            print(f"Results saved to: {csv_file_path}")
            return best_config
        else:
            print("No valid configurations found.")
            return None

def main():
    # User can specify M, N, K values as command-line arguments
    # 1024, 576, 7168 [44.10 us, 191.74 TFLOPS], Slowest: [213.24 us, 39.65 TFLOPS]
    import argparse
    parser = argparse.ArgumentParser(description='GEMM Kernel Tuner')
    parser.add_argument('--M', type=int, default=1024, help='M dimension')
    parser.add_argument('--N', type=int, default=576, help='N dimension')
    parser.add_argument('--K', type=int, default=7168, help='K dimension')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of configurations to compile in parallel')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker threads')
    args = parser.parse_args()
    
    # Run the tuner
    best_config = tune_gemm_kernel(args.M, args.N, args.K, args.batch_size, args.max_workers)
    
if __name__ == '__main__':
    main()
