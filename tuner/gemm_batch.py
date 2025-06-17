import os
from typing import List, Dict, Optional
import time  # Add import for time module


output_dir = os.path.join(os.path.dirname(__file__), 'output_dir')
os.makedirs(output_dir, exist_ok=True)
TUNE_KERNEL = '''
        DISPATCH_GEMM(1024,   1536,   7168,    128,    128,     32,      2,      2,    128,     4);
        DISPATCH_GEMM(1024,   3072,   1536,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(1024,    576,   7168,    128,     64,     32,      4,      1,    128,     4);
        DISPATCH_GEMM(1024,   7168,    256,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(1024,   7168,   2048,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(1024,   4608,   7168,    128,    128,     32,      2,      2,    128,     1);
        DISPATCH_GEMM(1024,   7168,   2304,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(1024,    512,   7168,     64,    128,     32,      2,      2,    128,     4);
        DISPATCH_GEMM(1024,   4096,    512,    128,    256,     32,      2,      4,    256,     1);
        DISPATCH_GEMM(6144,   1536,   7168,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,   3072,   1536,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,    576,   7168,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,   7168,    256,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,   7168,   2048,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,   4608,   7168,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,   7168,   2304,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,    512,   7168,    256,    128,     32,      4,      2,    256,     1);
        DISPATCH_GEMM(6144,   4096,    512,    256,    128,     32,      4,      2,    256,     1);
'''

#!/usr/bin/env python
"""
Batch GEMM tuner - processes multiple GEMM configurations and 
finds optimal parameters for each one.
"""

import os
import re
import argparse
import datetime
from gemm_tuner import tune_gemm_kernel

def parse_dispatch_gemm(tune_kernel_str: str) -> List[Dict[str, int]]:
    """Parse DISPATCH_GEMM entries from the input string."""
    pattern = r'DISPATCH_GEMM\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    lines = tune_kernel_str.strip().splitlines()
    
    configs = []
    for line in lines:
        line = line.strip()
        if line.startswith("//") or not re.match(pattern, line):
            # Preserve comments or non-matching lines
            configs.append({'raw': line})
        else:
            match = re.match(pattern, line)
            if match:
                M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, SPLITK_FACTOR = map(int, match.groups())
                configs.append({
                    'M': M, 'N': N, 'K': K,
                    'BM': BM, 'BN': BN, 'BK': BK,
                    'WARP_M': WARP_M, 'WARP_N': WARP_N,
                    'BLOCK_SIZE': BLOCK_SIZE,
                    'SPLITK_FACTOR': SPLITK_FACTOR
                })
    
    return configs

def format_dispatch_gemm(config: Dict[str, int], comment: str = "", tflops: Optional[float] = None) -> str:
    """Format a config as a DISPATCH_GEMM entry."""
    if tflops is not None:
        comment = f"{comment} ({tflops:.2f} TFlops)"
    
    return f"DISPATCH_GEMM({config['M']:6}, {config['N']:6}, {config['K']:6}, " \
           f"{config['BM']:6}, {config['BN']:6}, {config['BK']:6}, " \
           f"{config['WARP_M']:6}, {config['WARP_N']:6}, {config['BLOCK_SIZE']:6}, {config['SPLITK_FACTOR']}); {comment}"

def calculate_tflops(m: int, n: int, k: int, time_ms: float) -> float:
    """Calculate TFlops for a GEMM operation.
    
    Args:
        m, n, k: Matrix dimensions
        time_ms: Execution time in milliseconds
        
    Returns:
        TFlops value
    """
    if time_ms <= 0:
        return 0.0
    
    # 2 * M * N * K operations, convert ms to seconds (divide by 1000)
    return (2 * m * n * k) / (time_ms * 1e-3 * 1e12)

def batch_tune(
    configs: List[Dict[str, int]], 
    batch_size: int = 100, 
    max_workers: Optional[int] = None, 
    output_file: Optional[str] = None
) -> List[Dict[str, Optional[float]]]:
    """Tune multiple GEMM configurations and find optimal parameters for each."""
    results = []
    start_time = time.time()  # Record the start time
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file if specified
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"tuned_gemm_configs_{timestamp}.cpp")
    
    print(f"Starting batch tuning of {len(configs)} GEMM configurations")
    print(f"Results will be saved to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("// GEMM configurations tuned on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("// Format: DISPATCH_GEMM(M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE)\n\n")
        
        for i, config in enumerate(configs):
            elapsed_time = time.time() - start_time
            avg_time_per_config = elapsed_time / (i + 1)
            remaining_time = avg_time_per_config * (len(configs) - (i + 1))
            
            if 'raw' in config:
                # Handle raw lines (e.g., comments or invalid lines)
                print(f"\n[{i+1}/{len(configs)}]) Skipping raw line: {config['raw']}")
                f.write(config['raw'] + "\n")
                continue
            
            print(f"\n[{i+1}/{len(configs)}]) Tuning GEMM {config['M']}x{config['N']}x{config['K']}")
            print(f"Estimated time remaining: {remaining_time:.2f} seconds")
            
            # Run tuner to find best parameters
            best_config = tune_gemm_kernel(
                config['M'], config['N'], config['K'],
                batch_size=batch_size,
                max_workers=max_workers
            )
            
            # Original config info with TFlops if original execution time is available
            original_tflops = None
            if 'execution_time' in config and config['execution_time'] > 0:
                original_tflops = calculate_tflops(
                    config['M'], config['N'], config['K'], 
                    config['execution_time']
                )
                original_comment = f"// Original: {config['execution_time']:.2f} ms"
            else:
                original_comment = "// Original configuration"
            
            original_config_line = format_dispatch_gemm(config, original_comment, original_tflops)
            f.write(original_config_line + "\n")
            
            if best_config:
                # Format the optimized configuration with TFlops
                best_time = best_config.get('execution_time', 0)
                optimized_tflops = None
                
                if best_time > 0:
                    # Use the same M, N, K for TFlops calculation
                    optimized_tflops = calculate_tflops(
                        config['M'], config['N'], config['K'],
                        best_time
                    )
                    optimized_comment = f"// Optimized: {best_time:.2f} ms"
                else:
                    optimized_comment = "// Optimized configuration (no timing data)"
                
                optimized_config_line = format_dispatch_gemm(
                    best_config, 
                    optimized_comment,
                    optimized_tflops
                )
                f.write(optimized_config_line + "\n\n")  # Write immediately to the file
                
                # Save the result
                results.append({
                    'original': config,
                    'optimized': best_config,
                    'original_tflops': original_tflops,
                    'optimized_tflops': optimized_tflops,
                    'improvement': (config.get('execution_time', float('inf')) - best_time) / config.get('execution_time', float('inf')) * 100 if 'execution_time' in config else None
                })
            else:
                f.write("// Failed to find optimized configuration\n\n")  # Write failure immediately
    
    print(f"\nBatch tuning completed. Results saved to: {output_file}")
    return results

def main() -> None:
    """Main function to parse arguments and run batch tuning."""
    parser = argparse.ArgumentParser(description='Batch GEMM Kernel Tuner')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of configurations to compile in parallel')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker threads')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()
    
    # Parse the configurations from TUNE_KERNEL
    configs = parse_dispatch_gemm(TUNE_KERNEL)
    
    # Run batch tuning
    batch_tune(configs, args.batch_size, args.max_workers, args.output)

if __name__ == '__main__':
    main()
