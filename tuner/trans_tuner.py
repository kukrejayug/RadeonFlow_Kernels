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

# Setup a function to get a logger for a specific M,N configuration
def get_logger(M=None, N=None):
    # Clear any existing handlers from the logger
    logger = logging.getLogger('trans_tuner')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up the logger
    logger.setLevel(logging.DEBUG)
    
    # Define the log file path based on M,N if provided
    if M is not None and N is not None:
        log_file_path = os.path.join(output_dir, f'trans_{M}_{N}.log')
    else:
        log_file_path = os.path.join(output_dir, 'trans_tuner.log')
    
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

def get_transpose_kernel(M: int, N: int, TILE_DIM: int, BLOCK_SIZE: int, VEC_SIZE: int, elem_type: str = "float") -> str:
    """Generates and compiles a transpose kernel with the specified parameters"""
    
    # Collect logs for this operation
    log_messages = []
    log_messages.append(f"Starting transpose kernel generation and compilation")
    
    # Parameters for logging
    params = {
        'M': M, 'N': N, 
        'TILE_DIM': TILE_DIM, 'BLOCK_SIZE': BLOCK_SIZE, 'VEC_SIZE': VEC_SIZE,
        'elem_type': elem_type
    }
    
    # Get the full path to the template file
    template_path = os.path.join(os.path.dirname(__file__), 'template', 'trans_perf.cpp')
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
        f"using Elem = {elem_type} /* param Elem */;\n",
        f"constexpr int M = {M} /* param M */, N = {N} /* param N */;\n",
        f"constexpr int TILE_DIM = {TILE_DIM} /* param TILE_DIM */, BLOCK_SIZE = {BLOCK_SIZE} /* param BLOCK_SIZE */, VEC_SIZE = {VEC_SIZE} /* param VEC_SIZE */;\n",
        "// End parameterization\n"
    ]
    
    # Replace the parameterization section
    modified_template = template_lines[:begin_idx] + new_params + template_lines[end_idx+1:]
    
    # Write the modified template to the correct path
    cpp_path = os.path.join(tmp_dir, f'trans_perf_{M}_{N}_{TILE_DIM}_{BLOCK_SIZE}_{VEC_SIZE}_{elem_type}.cpp')
    log_messages.append(f"Writing modified template to: {cpp_path}")
    
    try:
        with open(cpp_path, 'w') as f:
            f.writelines(modified_template)
    except Exception as e:
        log_messages.append(f"Failed to write to file: {e}")
        log_operation("Compile", params, log_messages, success=False)
        return None
    
    # Compile the generated file
    binary_path = os.path.join(tmp_dir, f'trans_perf_{M}_{N}_{TILE_DIM}_{BLOCK_SIZE}_{VEC_SIZE}_{elem_type}')
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

def execute_transpose_kernel(binary_path: str, params: Dict = None) -> float:
    """
    Execute the compiled transpose kernel binary and return its execution time in milliseconds.
    
    Args:
        binary_path: Path to the compiled binary
        params: Dictionary of parameters for logging purposes
        
    Returns:
        float: Execution time in milliseconds, or None if execution failed
    """
    log_messages = []
    log_messages.append(f"Executing transpose kernel: {binary_path}")
    
    if params is None:
        # Try to extract parameters from binary name
        try:
            basename = os.path.basename(binary_path)
            parts = basename.replace('trans_perf_', '').split('_')
            if len(parts) >= 5:
                params = {
                    'M': parts[0], 'N': parts[1],
                    'TILE_DIM': parts[2], 'BLOCK_SIZE': parts[3], 'VEC_SIZE': parts[4]
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
    TILE_DIM = config['TILE_DIM']
    BLOCK_SIZE = config['BLOCK_SIZE']
    VEC_SIZE = config['VEC_SIZE']
    elem_type = config['elem_type']
    
    # Compile the kernel
    binary_path = get_transpose_kernel(M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE, elem_type)
    
    # Return the config and binary path if successful, None otherwise
    if binary_path:
        return (config, binary_path)
    return (config, None)

def tune_transpose_kernel(M, N, elem_type="float", batch_size=100, max_workers=min(os.cpu_count(), 40)):
    """
    Tune transpose kernel parameters for best performance with given M, N.
    Uses parallel compilation for batches of configurations.
    
    Args:
        M, N: Matrix dimensions
        elem_type: Element type (float, double, etc.)
        batch_size: Number of configurations to compile in parallel
        max_workers: Maximum number of worker threads (None = auto)
        
    Returns:
        tuple: Best configuration parameters and execution time
    """
    # Get a logger specific to this M,N configuration
    global logger
    logger = get_logger(M, N)
    logger.info(f"Starting transpose kernel tuning for M={M}, N={N}, elem_type={elem_type}")
    
    # Generate valid parameter values according to constraints
    valid_TILE_DIM = get_valid_powers_of_two(32, 192, 10000)  # Common tile dimensions
    valid_BLOCK_SIZE = get_valid_powers_of_two(256, 1024, 10000)  # Common block sizes
    valid_VEC_SIZE = [4, 8, 16, 32]  # Vector sizes
    
    # Generate all possible combinations
    all_combinations = []
    for TILE_DIM in valid_TILE_DIM:
        for BLOCK_SIZE in valid_BLOCK_SIZE:
            for VEC_SIZE in valid_VEC_SIZE:
                # Check basic constraints
                if TILE_DIM % VEC_SIZE != 0:
                    continue
                TBLOCK_X = TILE_DIM // VEC_SIZE
                if BLOCK_SIZE % TBLOCK_X != 0:
                    continue
                if M % TILE_DIM != 0 or N % TILE_DIM != 0:
                    continue
                
                all_combinations.append({
                    'M': M, 'N': N,
                    'TILE_DIM': TILE_DIM, 'BLOCK_SIZE': BLOCK_SIZE, 'VEC_SIZE': VEC_SIZE,
                    'elem_type': elem_type
                })
    
    # Log the filtering results
    logger.info(f"Found {len(all_combinations)} valid configurations")
    print(f"Found {len(all_combinations)} valid configurations")
    
    configs = all_combinations
    
    # Setup CSV file for recording results
    csv_file_path = os.path.join(output_dir, f'trans_{M}_{N}_{elem_type}.csv')
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Define CSV columns
        fieldnames = ['M', 'N', 'TILE_DIM', 'BLOCK_SIZE', 'VEC_SIZE', 'elem_type', 
                      'execution_time_us', 'bandwidth_gb_s']
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
        print(f"Starting transpose tuning for M={M}, N={N}, elem_type={elem_type}")
        print(f"Total configurations to try: {total_configs}")
        
        # Get element size in bytes
        elem_size_map = {'float': 4, 'double': 8, '__hip_bfloat16': 2, 'half': 2, '__hip_fp8_e4m3_fnuz': 1}
        elem_size = elem_size_map.get(elem_type, 4)
        
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
                TILE_DIM = config['TILE_DIM']
                BLOCK_SIZE = config['BLOCK_SIZE']
                VEC_SIZE = config['VEC_SIZE']
                
                execution_time_ms = execute_transpose_kernel(binary_path, config)
                configs_tested += 1
                
                if execution_time_ms is not None:
                    # Convert from milliseconds to microseconds
                    execution_time_us = execution_time_ms * 1000
                    
                    # Calculate memory bandwidth: (read + write) bytes / time in seconds
                    # First convert milliseconds to seconds
                    execution_time_s = execution_time_ms / 1000
                    bytes_transferred = 2 * M * N * elem_size  # read + write
                    bandwidth_gb_s = bytes_transferred / (execution_time_s * 10**9)
                    
                    config_str = f"TILE_DIM={TILE_DIM}, BLOCK_SIZE={BLOCK_SIZE}, VEC_SIZE={VEC_SIZE}"
                    
                    # Log detailed result to file only
                    logger.info(f"Config: {config_str} - Time: {execution_time_us:.2f} µs, Bandwidth: {bandwidth_gb_s:.2f} GB/s")
                    
                    # Update console with progress and current best (using carriage return)
                    best_info = f"Best: {best_time:.2f}µs ({best_config['bandwidth_gb_s']:.2f} GB/s)" if best_config else "No valid config yet"
                    print(f"\rProgress: {configs_tested}/{total_configs} configs tested. {best_info}", end="")
                    
                    # Store result and write to CSV
                    result = {
                        **config,
                        'execution_time_us': execution_time_us,
                        'bandwidth_gb_s': bandwidth_gb_s
                    }
                    all_results.append(result)
                    csv_writer.writerow(result)
                    csv_file.flush()  # Ensure data is written immediately
                    
                    # Update best configuration if this is better
                    if execution_time_ms < best_time:
                        best_time = execution_time_ms
                        best_config = config.copy()
                        best_config['execution_time_us'] = execution_time_us
                        best_config['bandwidth_gb_s'] = bandwidth_gb_s
                        
                        # Log new best to file
                        logger.info(f"New best: {config_str} - Time: {execution_time_us:.2f} µs, Bandwidth: {bandwidth_gb_s:.2f} GB/s")
                        
                        # Update console with new best (using carriage return)
                        print(f"\rNew best config: {config_str} - Time: {execution_time_us:.2f} µs, Bandwidth: {bandwidth_gb_s:.2f} GB/s", end="")
                        # Wait a moment to ensure the best config is visible
                        time.sleep(0.5)
                        # Return to progress display
                        print(f"\rProgress: {configs_tested}/{total_configs} configs tested. Best: {execution_time_us:.2f}µs ({bandwidth_gb_s:.2f} GB/s)", end="")
        
        # Save all results to a JSON file for additional processing if needed
        import json
        results_file = os.path.join(output_dir, f'trans_tuner_results_M{M}_N{N}_{elem_type}.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print final newline to finish the carriage return line
        print()
        
        if best_config:
            print("\nBest configuration found:")
            config_str = f"TILE_DIM={best_config['TILE_DIM']}, BLOCK_SIZE={best_config['BLOCK_SIZE']}, VEC_SIZE={best_config['VEC_SIZE']}"
            print(f"  {config_str}")
            print(f"  Execution time: {best_config['execution_time_us']:.2f} µs")
            print(f"  Memory Bandwidth: {best_config['bandwidth_gb_s']:.2f} GB/s")
            print(f"Results saved to: {csv_file_path}")
            return best_config
        else:
            print("No valid configurations found.")
            return None

def main():
    # User can specify M, N values as command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Transpose Kernel Tuner')
    parser.add_argument('--M', type=int, default=4096, help='M dimension')
    parser.add_argument('--N', type=int, default=4096, help='N dimension')
    parser.add_argument('--elem-type', type=str, default='__hip_fp8_e4m3_fnuz', help='Element type')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of configurations to compile in parallel')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker threads')
    args = parser.parse_args()
    
    # Run the tuner
    best_config = tune_transpose_kernel(args.M, args.N, args.elem_type, args.batch_size, args.max_workers)
    
if __name__ == '__main__':
    main()
