#!/usr/bin/env python3
import json
import os
import re
import subprocess
import itertools
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time

@dataclass
class TuneResult:
    case: int
    subcase: int
    params: Dict[str, int]
    time_us: float
    is_valid: bool

class MoETuner:
    def __init__(self, param_file: str, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        
        # Read parameter configuration
        with open(param_file, 'r') as f:
            self.config = json.load(f)
        
        self.cases = self.config['cases']
        self.subcase_indexes = self.config['subcase_indexes']
        self.tunables = self.config['tunables']
        
        # Create workspace directory
        os.makedirs(self.workspace_root, exist_ok=True)
        
        # Get source code root directory
        self.src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def get_tunable_params_for_subcase(self, subcase_idx: int) -> Dict[str, Dict]:
        """Get tunable parameters for a specific subcase"""
        applicable_params = {}
        
        for param_name, param_config in self.tunables.items():
            # Check if there are case_specific restrictions
            if 'case_specific' not in param_config:
                # No restrictions, applicable to all subcases
                applicable_params[param_name] = param_config
            elif subcase_idx in param_config['case_specific']:
                # Has restrictions, and current subcase is in the restriction list
                applicable_params[param_name] = param_config
                
        return applicable_params
    
    def generate_param_combinations(self, subcase_idx: int) -> List[Dict[str, int]]:
        """Generate all parameter combinations for a specific subcase"""
        applicable_params = self.get_tunable_params_for_subcase(subcase_idx)
        
        if not applicable_params:
            return [{}]  # No parameters need tuning
        
        # Generate all possible values for each parameter
        param_values = {}
        for param_name, param_config in applicable_params.items():
            values = list(range(
                param_config['min'],
                param_config['max'] + 1,
                param_config['inc']
            ))
            param_values[param_name] = values
        
        # Generate all combinations
        param_names = list(param_values.keys())
        value_lists = [param_values[name] for name in param_names]
        
        combinations = []
        for values in itertools.product(*value_lists):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
            
        return combinations
    
    def modify_moe_cpp(self, src_file: str, dst_file: str, case_idx: int, params: Dict[str, int]):
        """Modify DISPATCH_MOE parameters in moe.cpp file"""
        with open(src_file, 'r') as f:
            content = f.read()
        
        # Find actual DISPATCH_MOE calls (not macro definitions and comments)
        lines = content.split('\n')
        dispatch_calls = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip comment lines, macro definitions and empty lines
            if (line.startswith('//') or line.startswith('#define') or 
                line.startswith('/*') or not line):
                i += 1
                continue
            
            # Look for DISPATCH_MOE calls (not in comments)
            if 'DISPATCH_MOE(' in line and not line.startswith('#define'):
                # Found the start of a call
                start_line = i
                call_lines = [line]
                
                # If this line doesn't end with semicolon, continue collecting subsequent lines
                while not call_lines[-1].rstrip().endswith(');'):
                    i += 1
                    if i < len(lines):
                        next_line = lines[i].strip()
                        # Skip comment lines
                        if not next_line.startswith('//'):
                            call_lines.append(lines[i])
                    else:
                        break
                
                # Merge all lines
                full_call = ' '.join(call_lines)
                
                # Check if it's an actual call (contains numeric parameters and not in comments)
                if re.search(r'DISPATCH_MOE\s*\(\s*\d+', full_call) and not full_call.strip().startswith('//'):
                    dispatch_calls.append({
                        'text': full_call,
                        'start_line': start_line,
                        'end_line': i,
                        'lines': call_lines
                    })
            
            i += 1
        
        if case_idx >= len(dispatch_calls):
            raise ValueError(f"Case index {case_idx} out of range, found {len(dispatch_calls)} DISPATCH_MOE calls")
        
        # Get corresponding DISPATCH_MOE call
        target_call = dispatch_calls[case_idx]
        original_dispatch = target_call['text']
        
        # Parse parameters - extract content within parentheses
        start_paren = original_dispatch.find('(')
        end_paren = original_dispatch.rfind(')')
        params_str = original_dispatch[start_paren + 1:end_paren]
        
        # Clean parameter string, remove extra whitespace and newlines
        params_str = re.sub(r'\s+', ' ', params_str.strip())
        
        # Split parameters
        params_list = [p.strip() for p in params_str.split(',')]
        
        # Modify parameters at specified indices
        for param_name, param_value in params.items():
            param_idx = self.tunables[param_name]['index']
            if param_idx < len(params_list):
                params_list[param_idx] = str(param_value)
        
        # Reassemble DISPATCH_MOE, maintain original multi-line format
        if len(params_list) > 6:
            # Split into two lines, maintain original indentation style
            first_line_params = ', '.join(params_list[:6])
            second_line_params = ', '.join([f"{p:>12}" for p in params_list[6:]])
            new_dispatch_lines = [
                f"    DISPATCH_MOE({first_line_params},",
                f"                                            {second_line_params});"
            ]
        else:
            new_dispatch_lines = [f"    DISPATCH_MOE({', '.join(params_list)});"]
        
        # Replace content
        new_lines = lines[:target_call['start_line']] + new_dispatch_lines + lines[target_call['end_line'] + 1:]
        new_content = '\n'.join(new_lines)
        
        # Write to new file
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with open(dst_file, 'w') as f:
            f.write(new_content)
    
    def build_variant(self, build_dir: str, modified_src: str) -> bool:
        """Build a variant"""
        try:
            # Create build directory
            os.makedirs(build_dir, exist_ok=True)
            
            # Copy source files to temporary location, excluding unnecessary folders
            temp_src_dir = os.path.join(build_dir, 'src_copy')
            if os.path.exists(temp_src_dir):
                shutil.rmtree(temp_src_dir)
            
            # Define folders and files to ignore
            def ignore_patterns(dir, files):
                ignore_dirs = {'.cache', 'build', '.vscode', '.git', '__pycache__', 
                              '.pytest_cache', '.mypy_cache', 'CMakeFiles'}
                return [f for f in files if f in ignore_dirs or f.endswith('.pyc')]
            
            shutil.copytree(self.src_root, temp_src_dir, ignore=ignore_patterns)
            
            # Replace moe.cpp
            moe_cpp_path = os.path.join(temp_src_dir, 'src/moe/moe.cpp')
            shutil.copy2(modified_src, moe_cpp_path)
            
            # Run cmake
            cmake_cmd = [
                'cmake',
                '-S', temp_src_dir,
                '-B', build_dir,
                '-DCMAKE_BUILD_TYPE=Release',
                '-G', 'Ninja'
            ]
            subprocess.run(cmake_cmd, check=True, capture_output=True)
            
            # Build moe shared library and moe_topk_checker
            build_cmd = ['cmake', '--build', build_dir, '--target', 'moe', 'moe_topk_checker', '-j']
            subprocess.run(build_cmd, check=True, capture_output=True)
            
            return True
            
        except subprocess.CalledProcessError:
            return False
    
    def run_test(self, executable: str, case: int, subcase: int) -> Optional[float]:
        """Run test and get results"""
        try:
            # Set library path
            build_dir = os.path.dirname(executable)
            env = os.environ.copy()
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{build_dir}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = build_dir
            
            cmd = [executable, '-c', str(case), '-t', str(subcase)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            
            # Parse output time value
            output = result.stdout.strip()
            if output:
                return float(output)
            else:
                return None
                
        except subprocess.CalledProcessError:
            return None
    
    def prepare_builds(self) -> Dict[Tuple[int, Tuple[Tuple[str, int], ...]], str]:
        """Prepare all required builds"""
        builds = {}  # (case, params_tuple) -> build_dir
        build_tasks = []
        
        # Collect all required build tasks
        for case in self.cases:
            for subcase in self.subcase_indexes:
                combinations = self.generate_param_combinations(subcase)
                
                for params in combinations:
                    # Create parameter tuple as key
                    params_tuple = tuple(sorted(params.items()))
                    key = (case, params_tuple)
                    
                    if key not in builds:
                        # Create unique build directory
                        build_id = f"case{case}_" + "_".join([f"{k}{v}" for k, v in params.items()])
                        build_dir = os.path.join(self.workspace_root, 'builds', build_id)
                        builds[key] = build_dir
                        
                        # Add build task
                        modified_src = os.path.join(self.workspace_root, 'modified_src', f'{build_id}.cpp')
                        build_tasks.append((case, params, modified_src, build_dir))
        
        # Prepare modified source files in parallel
        print("Preparing modified source files...")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            
            for case, params, modified_src, build_dir in build_tasks:
                future = executor.submit(
                    self.modify_moe_cpp,
                    os.path.join(self.src_root, 'src/moe/moe.cpp'),
                    modified_src,
                    case,
                    params
                )
                futures.append((future, modified_src, build_dir))
            
            for future, _, _ in futures:
                future.result()
        
        # Parallel compilation
        print(f"Starting compilation of {len(build_tasks)} variants...")
        successful_builds = {}
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            
            for case, params, modified_src, build_dir in tqdm(build_tasks, desc="Submitting build tasks"):
                future = executor.submit(self.build_variant, build_dir, modified_src)
                futures.append((future, case, params, build_dir))
            
            for future, case, params, build_dir in tqdm(futures, desc="Build progress"):
                success = future.result()
                if success:
                    params_tuple = tuple(sorted(params.items()))
                    key = (case, params_tuple)
                    successful_builds[key] = build_dir
                else:
                    print(f"Build failed: case={case}, params={params}")
        
        return successful_builds
    
    def run_tuning(self):
        """Run complete tuning process"""
        # Prepare all builds
        builds = self.prepare_builds()
        
        if not builds:
            print("No successfully compiled variants!")
            return
        
        # Run all tests
        print("\nStarting test runs...")
        results = []
        test_count = 0
        
        for case in self.cases:
            for subcase in self.subcase_indexes:
                combinations = self.generate_param_combinations(subcase)
                test_count += len(combinations)
        
        with tqdm(total=test_count, desc="Test progress") as pbar:
            for case in self.cases:
                for subcase in self.subcase_indexes:
                    combinations = self.generate_param_combinations(subcase)
                    
                    for params in combinations:
                        params_tuple = tuple(sorted(params.items()))
                        key = (case, params_tuple)
                        
                        if key in builds:
                            build_dir = builds[key]
                            executable = os.path.join(build_dir, 'moe_topk_checker')
                            
                            # Run test
                            time_us = self.run_test(executable, case, subcase)
                            
                            result = TuneResult(
                                case=case,
                                subcase=subcase,
                                params=params,
                                time_us=time_us if time_us is not None else float('inf'),
                                is_valid=time_us is not None
                            )
                            results.append(result)
                        
                        pbar.update(1)
        
        # Analyze results
        self.analyze_results(results)
    
    def analyze_results(self, results: List[TuneResult]):
        """Analyze tuning results and output report"""
        print("\n" + "="*80)
        print("Tuning Results Report")
        print("="*80 + "\n")
        
        # Group by case and subcase
        grouped_results = {}
        for result in results:
            key = (result.case, result.subcase)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Output best and worst results for each group
        for (case, subcase), group_results in sorted(grouped_results.items()):
            print(f"Case {case}, Subcase {subcase}:")
            print("-" * 40)
            
            # Filter valid results
            valid_results = [r for r in group_results if r.is_valid]
            
            if not valid_results:
                print("  No valid results\n")
                continue
            
            # Find best and worst
            best = min(valid_results, key=lambda r: r.time_us)
            worst = max(valid_results, key=lambda r: r.time_us)
            
            print(f"  Best time: {best.time_us:.2f} us")
            print(f"  Best params: {best.params}")
            print(f"  Worst time: {worst.time_us:.2f} us")
            print(f"  Worst params: {worst.params}")
            print(f"  Performance improvement: {(worst.time_us / best.time_us - 1) * 100:.1f}%")
            print(f"  Valid combinations: {len(valid_results)}/{len(group_results)}")
            print()
        
        # Save detailed results to file
        output_file = os.path.join(self.workspace_root, 'tuning_results.json')
        
        results_data = []
        for result in results:
            results_data.append({
                'case': result.case,
                'subcase': result.subcase,
                'params': result.params,
                'time_us': result.time_us if result.is_valid else None,
                'is_valid': result.is_valid
            })
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MOE Kernel Parameter Tuning Tool')
    parser.add_argument('--param-file', default='tuner/moe_param.json', 
                        help='Parameter configuration file path')
    parser.add_argument('--workspace', default='/tmp/moe_tuning',
                        help='Workspace directory path')
    args = parser.parse_args()
    
    tuner = MoETuner(args.param_file, args.workspace)
    tuner.run_tuning()

if __name__ == '__main__':
    main()
