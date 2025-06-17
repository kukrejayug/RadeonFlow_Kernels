import os
from typing import List, Dict, Optional
import time
import re
import argparse
import datetime
from trans_tuner import tune_transpose_kernel

output_dir = os.path.join(os.path.dirname(__file__), 'output_dir')
os.makedirs(output_dir, exist_ok=True)

# Transpose shapes derived from GEMM matrix dimensions
# Matrix A shapes [M,K] and Matrix B shapes [N,K]
        # DISPATCH_TRANSPOSE(1024, 7168, 32, 256, 4);  // Matrix A: M=1024, K=7168
        # DISPATCH_TRANSPOSE(1024, 1536, 32, 256, 4);  // Matrix A: M=1024, K=1536
        # DISPATCH_TRANSPOSE(1024, 256, 32, 256, 4);   // Matrix A: M=1024, K=256
        # DISPATCH_TRANSPOSE(1024, 2048, 32, 256, 4);  // Matrix A: M=1024, K=2048
        # DISPATCH_TRANSPOSE(1024, 512, 32, 256, 4);   // Matrix A: M=1024, K=512
        # DISPATCH_TRANSPOSE(6144, 7168, 32, 256, 4);  // Matrix A: M=6144, K=7168
        # DISPATCH_TRANSPOSE(6144, 1536, 32, 256, 4);  // Matrix A: M=6144, K=1536
        # DISPATCH_TRANSPOSE(6144, 256, 32, 256, 4);   // Matrix A: M=6144, K=256
        # DISPATCH_TRANSPOSE(6144, 2048, 32, 256, 4);  // Matrix A: M=6144, K=2048
        # DISPATCH_TRANSPOSE(6144, 512, 32, 256, 4);   // Matrix A: M=6144, K=512
        # DISPATCH_TRANSPOSE(1536, 7168, 32, 256, 4);  // Matrix B: N=1536, K=7168
        # DISPATCH_TRANSPOSE(3072, 1536, 32, 256, 4);  // Matrix B: N=3072, K=1536
        # DISPATCH_TRANSPOSE(576, 7168, 32, 256, 4);   // Matrix B: N=576, K=7168
        # DISPATCH_TRANSPOSE(7168, 256, 32, 256, 4);   // Matrix B: N=7168, K=256
        # DISPATCH_TRANSPOSE(7168, 2048, 32, 256, 4);  // Matrix B: N=7168, K=2048
        # DISPATCH_TRANSPOSE(4608, 7168, 32, 256, 4);  // Matrix B: N=4608, K=7168
        # DISPATCH_TRANSPOSE(7168, 2304, 32, 256, 4);  // Matrix B: N=7168, K=2304
        # DISPATCH_TRANSPOSE(512, 7168, 32, 256, 4);   // Matrix B: N=512, K=7168
        # DISPATCH_TRANSPOSE(4096, 512, 32, 256, 4);   // Matrix B: N=4096, K=512
TUNE_TRANSPOSE = '''

        // Additional configurations needed by GEMM operations
    DISPATCH_TRANSPOSE(256, 1024, 32, 256, 4); // transpose_fp8<256, 1024>
    DISPATCH_TRANSPOSE(256, 6144, 32, 256, 4); // transpose_fp8<256, 6144>
    DISPATCH_TRANSPOSE(256, 7168, 32, 256, 4); // transpose_fp8<256, 7168>
    DISPATCH_TRANSPOSE(512, 1024, 32, 256, 4); // transpose_fp8<512, 1024>
    DISPATCH_TRANSPOSE(512, 4096, 32, 256, 4); // transpose_fp8<512, 4096>
    DISPATCH_TRANSPOSE(512, 6144, 32, 256, 4); // transpose_fp8<512, 6144>
    DISPATCH_TRANSPOSE(1536, 1024, 32, 256, 4); // transpose_fp8<1536, 1024>
    DISPATCH_TRANSPOSE(1536, 3072, 32, 256, 4); // transpose_fp8<1536, 3072>
    DISPATCH_TRANSPOSE(1536, 6144, 32, 256, 4); // transpose_fp8<1536, 6144>
    DISPATCH_TRANSPOSE(2048, 1024, 32, 256, 4); // transpose_fp8<2048, 1024>
    DISPATCH_TRANSPOSE(2048, 6144, 32, 256, 4); // transpose_fp8<2048, 6144>
    DISPATCH_TRANSPOSE(2048, 7168, 32, 256, 4); // transpose_fp8<2048, 7168>
    DISPATCH_TRANSPOSE(2304, 1024, 32, 256, 4); // transpose_fp8<2304, 1024>
    DISPATCH_TRANSPOSE(2304, 6144, 32, 256, 4); // transpose_fp8<2304, 6144>
    DISPATCH_TRANSPOSE(2304, 7168, 32, 256, 4); // transpose_fp8<2304, 7168>
    DISPATCH_TRANSPOSE(7168, 512, 32, 256, 4); // transpose_fp8<7168, 512>
    DISPATCH_TRANSPOSE(7168, 576, 32, 256, 4); // transpose_fp8<7168, 576>
    DISPATCH_TRANSPOSE(7168, 1024, 32, 256, 4); // transpose_fp8<7168, 1024>
    DISPATCH_TRANSPOSE(7168, 1536, 32, 256, 4); // transpose_fp8<7168, 1536>
    DISPATCH_TRANSPOSE(7168, 4608, 32, 256, 4); // transpose_fp8<7168, 4608>
    DISPATCH_TRANSPOSE(7168, 6144, 32, 256, 4); // transpose_fp8<7168, 6144>
'''

def parse_dispatch_transpose(tune_transpose_str: str) -> List[Dict[str, int]]:
    """Parse DISPATCH_TRANSPOSE entries from the input string."""
    pattern = r'DISPATCH_TRANSPOSE\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    lines = tune_transpose_str.strip().splitlines()
    
    configs = []
    for line in lines:
        line = line.strip()
        if line.startswith("//") or not re.match(pattern, line):
            # Preserve comments or non-matching lines
            configs.append({'raw': line})
        else:
            match = re.match(pattern, line)
            if match:
                M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE = map(int, match.groups())
                configs.append({
                    'M': M, 'N': N,
                    'TILE_DIM': TILE_DIM, 'BLOCK_SIZE': BLOCK_SIZE, 'VEC_SIZE': VEC_SIZE
                })
    
    return configs

def format_dispatch_transpose(config: Dict[str, int], comment: str = "", bandwidth: Optional[float] = None) -> str:
    """Format a config as a DISPATCH_TRANSPOSE entry."""
    if bandwidth is not None:
        comment = f"{comment} ({bandwidth:.2f} GB/s)"
    
    return f"DISPATCH_TRANSPOSE({config['M']:6}, {config['N']:6}, " \
           f"{config['TILE_DIM']:6}, {config['BLOCK_SIZE']:6}, {config['VEC_SIZE']}); {comment}"

def calculate_bandwidth(m: int, n: int, time_us: float, elem_size: int = 2) -> float:
    """Calculate memory bandwidth for a transpose operation.
    
    Args:
        m, n: Matrix dimensions
        time_us: Execution time in microseconds
        elem_size: Element size in bytes (default 2 for half)
        
    Returns:
        Bandwidth in GB/s
    """
    if time_us <= 0:
        return 0.0
    
    # Transpose involves reading and writing the entire matrix (2 * M * N * elem_size bytes)
    bytes_transferred = 2 * m * n * elem_size
    time_s = time_us * 1e-6  # Convert microseconds to seconds
    return bytes_transferred / (time_s * 1e9)  # Convert to GB/s

def batch_tune_transpose(
    configs: List[Dict[str, int]], 
    elem_type: str = "__hip_fp8_e4m3_fnuz",
    batch_size: int = 100, 
    max_workers: Optional[int] = None, 
    output_file: Optional[str] = None
) -> List[Dict[str, Optional[float]]]:
    """Tune multiple transpose configurations and find optimal parameters for each."""
    results = []
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file if specified
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"tuned_transpose_configs_{timestamp}.cpp")
    
    print(f"Starting batch tuning of {len(configs)} transpose configurations")
    print(f"Element type: {elem_type}")
    print(f"Results will be saved to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("// Transpose configurations tuned on " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write(f"// Element type: {elem_type}\n")
        f.write("// Format: DISPATCH_TRANSPOSE(M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE)\n\n")
        
        for i, config in enumerate(configs):
            elapsed_time = time.time() - start_time
            avg_time_per_config = elapsed_time / (i + 1)
            remaining_time = avg_time_per_config * (len(configs) - (i + 1))
            
            if 'raw' in config:
                # Handle raw lines (e.g., comments or invalid lines)
                print(f"\n[{i+1}/{len(configs)}]) Skipping raw line: {config['raw']}")
                f.write(config['raw'] + "\n")
                continue
            
            print(f"\n[{i+1}/{len(configs)}]) Tuning Transpose {config['M']}x{config['N']}")
            print(f"Estimated time remaining: {remaining_time:.2f} seconds")
            
            # Run tuner to find best parameters
            best_config = tune_transpose_kernel(
                config['M'], config['N'], 
                elem_type=elem_type,
                batch_size=batch_size,
                max_workers=max_workers
            )
            
            # Original config info
            original_comment = "// Original configuration"
            original_config_line = format_dispatch_transpose(config, original_comment)
            f.write(original_config_line + "\n")
            
            if best_config:
                # Format the optimized configuration with bandwidth
                best_time_us = best_config.get('execution_time_us', 0)
                optimized_bandwidth = best_config.get('bandwidth_gb_s', 0)
                
                if best_time_us > 0:
                    optimized_comment = f"// Optimized: {best_time_us:.2f} µs"
                else:
                    optimized_comment = "// Optimized configuration (no timing data)"
                
                optimized_config_line = format_dispatch_transpose(
                    best_config, 
                    optimized_comment,
                    optimized_bandwidth
                )
                f.write(optimized_config_line + "\n\n")
                
                # Save the result
                results.append({
                    'original': config,
                    'optimized': best_config,
                    'optimized_bandwidth': optimized_bandwidth,
                    'optimized_time_us': best_time_us
                })
                
                print(f"Best config found: TILE_DIM={best_config['TILE_DIM']}, "
                      f"BLOCK_SIZE={best_config['BLOCK_SIZE']}, VEC_SIZE={best_config['VEC_SIZE']}")
                print(f"Performance: {best_time_us:.2f} µs, {optimized_bandwidth:.2f} GB/s")
            else:
                f.write("// Failed to find optimized configuration\n\n")
                print("Failed to find valid configuration")
                results.append({
                    'original': config,
                    'optimized': None,
                    'optimized_bandwidth': None,
                    'optimized_time_us': None
                })
            
            f.flush()  # Ensure data is written immediately
    
    print(f"\nBatch transpose tuning completed. Results saved to: {output_file}")
    return results

def main() -> None:
    """Main function to parse arguments and run batch transpose tuning."""
    parser = argparse.ArgumentParser(description='Batch Transpose Kernel Tuner')
    parser.add_argument('--elem-type', type=str, default='__hip_fp8_e4m3_fnuz', help='Element type')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of configurations to compile in parallel')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker threads')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()
    
    # Parse the configurations from TUNE_TRANSPOSE
    configs = parse_dispatch_transpose(TUNE_TRANSPOSE)
    
    # Run batch tuning
    batch_tune_transpose(configs, args.elem_type, args.batch_size, args.max_workers, args.output)

if __name__ == '__main__':
    main()
