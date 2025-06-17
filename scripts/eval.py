import dataclasses
import multiprocessing
import re
import time
import os
import sys
import math
import yaml
import argparse
import importlib.util
from pathlib import Path
from typing import Any, Optional
import torch.cuda

from utils import set_seed


class PopcornOutput:
    def __init__(self):
        self.file = sys.stdout
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.file, flush=True)
    
    def log(self, key, value):
        self.print(f"{key}: {value}")


@dataclasses.dataclass
class TestCase:
    args: dict
    spec: str


def _combine(a: int, b: int) -> int:
    # combine two integers into one:
    # we need this to generate a secret seed based on the test-level seed and
    # the global secret seed.
    # the test-level seeds are public knowledge, and typically relatively small numbers,
    # so we need to make sure they don't provide any useful info for the full seed.
    # This Cantor construction ensures that if the secret seed is a large number,
    # then so is the overall seed.
    return int(a + (a+b)*(a+b+1)//2)


def get_test_cases_from_yaml(mode: str, seed: Optional[int], problem_dir: Path) -> list[TestCase]:
    try:
        yml_path = problem_dir / "task.yml"
        with open(yml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
    except Exception as E:
        print(f"Could not open task.yml file at {yml_path}: {E}", file=sys.stderr)
        exit(113)

    tests = []
    if mode == "test":
        cases = yaml_data.get("tests", [])
    elif mode == "benchmark" or mode == "leaderboard":
        cases = yaml_data.get("benchmarks", [])
    else:
        cases = []

    for case in cases:
        spec_parts = []
        for key, val in case.items():
            spec_parts.append(f"{key}: {val}")
        spec = "; ".join(spec_parts)
        tests.append(TestCase(spec=spec, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(test.args["seed"], seed)

    return tests


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
    runs = len(durations)
    if runs <= 1: # Avoid division by zero or sqrt of negative in std calculation if runs <= 1
       return Stats(runs=runs, mean=sum(durations)/runs if runs > 0 else 0, std=0, err=0,
                    best=min(durations) if runs > 0 else 0, worst=max(durations) if runs > 0 else 0)

    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg)**2, durations)) / (runs - 1)
    std = math.sqrt(variance)
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best),
                 worst=float(worst))


def _clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def wrap_check_implementation(data, submission_output, reference_module):
    # Old version returned just a single string, new version
    # returns (bool, str); this function ensures compatibility with old
    # problem definitions.
    result = reference_module.check_implementation(data, submission_output)
    if isinstance(result, tuple):
        return result
    else:
        # Assuming non-empty string means error
        is_error = bool(result)
        return not is_error, str(result)


def _run_single_test(test: TestCase, problem_dir: str):
    """
    Runs a single test case. Do not call directly
    """
    # Load the modules in the child process
    problem_dir_path = Path(problem_dir)
    sys.path.insert(0, str(problem_dir_path))
    
    # Import modules in the child process
    task_path = problem_dir_path / "task.py"
    reference_path = problem_dir_path / "reference.py"
    
    task_spec = importlib.util.spec_from_file_location("task", task_path)
    task_module = importlib.util.module_from_spec(task_spec)
    sys.modules['task'] = task_module
    task_spec.loader.exec_module(task_module)
    
    reference_spec = importlib.util.spec_from_file_location("reference", reference_path)
    reference_module = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference_module)
    
    # It's better to import submission within the function if it might be generated/modified
    try:
        from submission import custom_kernel
    except ImportError as e:
        print(f"Error: Could not import custom_kernel from submission.py. Did you generate it? {e}", file=sys.stderr)
        # Return an error state recognizable by the caller
        return False, "ImportError: custom_kernel not found in submission.py"
    
    data = reference_module.generate_input(**test.args)
    # Ensure data tensors are on the correct device before cloning/passing to kernel
    data = _move_to_cuda(data)
    torch.cuda.synchronize()
    submission_output = custom_kernel(_clone_data(data))
    torch.cuda.synchronize()
    # Ensure reference data is also on CUDA if check_implementation requires it
    data_cuda = _move_to_cuda(data)
    return wrap_check_implementation(data_cuda, submission_output, reference_module)


def _move_to_cuda(data):
    """Recursively moves tensors in data structure to CUDA."""
    if isinstance(data, tuple):
        return tuple(_move_to_cuda(x) for x in data)
    elif isinstance(data, list):
        return [_move_to_cuda(x) for x in data]
    elif isinstance(data, dict):
        return {k: _move_to_cuda(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def run_single_test(pool: multiprocessing.Pool, test: TestCase, problem_dir: str):
    """
    Runs a single test in another process.
    """
    # Consider using apply_async for better error handling and timeouts
    result = pool.apply_async(_run_single_test, (test, problem_dir))
    try:
        # Add a timeout maybe?
        return result.get()
    except Exception as e:
        print(f"Error running test case {test.spec}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Return a consistent error format
        return False, f"Exception during test execution: {e}"


def run_testing(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase], problem_dir: str):
    """
    Executes the actual test case code and checks for correctness.

    @param logger: A PopcornOutput object used for logging test results.
    @param pool: Process pool for running tests.
    @param tests: A list of TestCase objects representing the test cases to be executed.
    @param problem_dir: Path to the problem directory.
    @return: An integer representing the exit status: 0 if all tests pass, otherwise 112.
    """
    passed = True
    logger.log("test-count", len(tests))
    for idx, test in enumerate(tests):
        logger.log(f"test.{idx}.spec", test.spec)
        good, message = run_single_test(pool, test, str(problem_dir))
        if not good:
            logger.log(f"test.{idx}.status", "fail")
            logger.log(f"test.{idx}.error", message)
            passed = False
        else:
            logger.log(f"test.{idx}.status", "pass")
            if message: # Log message even on pass if provided (e.g., warnings)
                logger.log(f"test.{idx}.message", message)

    if passed:
        logger.log("check", "pass")
        return 0
    else:
        logger.log("check", "fail")
        return 112


def _run_single_benchmark(test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float, problem_dir: str) -> Stats | Any:
    """
    Runs one benchmark. Do not call directly.
    """
    # Load the modules in the child process
    problem_dir_path = Path(problem_dir)
    sys.path.insert(0, str(problem_dir_path))
    
    # Import modules in the child process
    task_path = problem_dir_path / "task.py"
    reference_path = problem_dir_path / "reference.py"
    
    task_spec = importlib.util.spec_from_file_location("task", task_path)
    task_module = importlib.util.module_from_spec(task_spec)
    sys.modules['task'] = task_module
    task_spec.loader.exec_module(task_module)
    
    reference_spec = importlib.util.spec_from_file_location("reference", reference_path)
    reference_module = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference_module)
    
    # Import submission here as well
    try:
        from submission import custom_kernel
    except ImportError as e:
        print(f"Error: Could not import custom_kernel from submission.py. Did you generate it? {e}", file=sys.stderr)
        return "ImportError: custom_kernel not found in submission.py"


    durations = []
    # generate input data once
    data = reference_module.generate_input(**test.args)
    data = _move_to_cuda(data) # Move to GPU
    check_copy = _clone_data(data)

    # Correctness check before timing runs
    try:
        output = custom_kernel(_clone_data(data)) # Use a clone for the check run
        torch.cuda.synchronize()
        good, message = wrap_check_implementation(check_copy, output, reference_module)
        del output # Free memory
        if not good:
            return message # Return error message if check fails
    except Exception as e:
        print(f"Error during initial correctness check for {test.spec}: {e}", file=sys.stderr)
        return f"Exception during correctness check: {e}"


    # Timing runs
    total_time_measured_ns = 0
    for i in range(max_repeats):
        run_data = _clone_data(data) # Use fresh clone for each timing run
        if recheck:
             # If rechecking, generate new data based on potentially updated seed
             if "seed" in test.args:
                 test.args["seed"] += 13 # Increment seed for variability
             data = reference_module.generate_input(**test.args)
             data = _move_to_cuda(data) # Move new data to GPU
             run_data = _clone_data(data) # Use clone of new data
             check_copy_recheck = _clone_data(data) # Clone for recheck

        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = custom_kernel(run_data)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        run_duration = end - start
        durations.append(run_duration)
        total_time_measured_ns += run_duration

        if recheck:
            try:
                good, message = wrap_check_implementation(check_copy_recheck, output, reference_module)
                if not good:
                     del output
                     return f"Recheck failed on iteration {i}: {message}"
            except Exception as e:
                del output
                print(f"Error during recheck for {test.spec} iter {i}: {e}", file=sys.stderr)
                return f"Exception during recheck iter {i}: {e}"

        del output # Free memory

        # Check exit conditions for timing loop
        if len(durations) >= 3: # Need at least 3 runs for meaningful stats
            stats = calculate_stats(durations)
            # Exit if relative error is small OR total time exceeds budget
            # Ensure stats.mean is not zero before dividing
            relative_error = (stats.err / stats.mean) if stats.mean > 0 else 0
            if relative_error < 0.01 or total_time_measured_ns > max_time_ns:
                 break
        elif len(durations) >= 1 and total_time_measured_ns > max_time_ns: # Exit if even 1 run exceeds time limit
             break


    if not durations:
        return "Error: No timing runs completed."

    return calculate_stats(durations)


def run_single_benchmark(pool: multiprocessing.Pool, test: TestCase, recheck: bool, max_repeats: int, max_time_ns: float, problem_dir: str):
    """
    For a particular test case, check correctness (if applicable) and grab runtime results.

    @param pool: Process pool for running benchmarks.
    @param test: TestCase object.
    @param recheck: Flag for whether to explicitly check functional correctness on each run.
    @param max_repeats: Maximum number of timing trials.
    @param max_time_ns: Maximum total time allowed for timing runs in nanoseconds.
    @param problem_dir: Path to the problem directory.
    @return: A Stats object for this benchmark case or an error message if it fails.
    """
    # Using apply_async allows for better error capture from the subprocess
    result = pool.apply_async(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns, problem_dir))
    try:
        # Consider adding a timeout to get()
        return result.get()
    except Exception as e:
        print(f"Error running benchmark case {test.spec}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return f"Exception during benchmark execution: {e}"


def run_benchmarking(logger: PopcornOutput, pool: multiprocessing.Pool, tests: list[TestCase], problem_dir: str):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process pool for running benchmarks.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @param problem_dir: Path to the problem directory.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    if not tests:
        logger.log("check", "pass") # No benchmarks to run
        print("Warning: No benchmark cases found.", file=sys.stderr)
        return 0

    # Warm up with the first test case, minimum 3 runs, short time limit
    logger.print("Running warmup...")
    warmup_result = run_single_benchmark(pool, tests[0], False, 10, 1e8, str(problem_dir)) # 100ms time limit for warmup
    if isinstance(warmup_result, Stats):
         logger.print(f"Warmup complete. Best time: {warmup_result.best / 1e6:.3f} ms")
    else:
         logger.print(f"Warning: Warmup failed or produced an error: {warmup_result}")
         # Decide if we should proceed or exit? For now, proceed but log warning.


    passed = True
    logger.log("benchmark-count", len(tests))
    results = [] # Store results for potential aggregation later

    for idx, test in enumerate(tests):
        logger.log(f"benchmark.{idx}.spec", test.spec)
        # Use recheck=False for benchmark mode, longer time limit (e.g., 10s)
        result = run_single_benchmark(pool, test, False, 100, 10e9, str(problem_dir))
        if isinstance(result, Stats):
            results.append(result)
            for field in dataclasses.fields(Stats):
                logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
            # Optionally log TFLOPS or other derived metrics here if applicable
        else:
            passed = False
            logger.log(f"benchmark.{idx}.status", "fail")
            logger.log(f"benchmark.{idx}.error", str(result))
            # Optionally break here if one benchmark fails, or continue testing others
            # break

    if passed:
        logger.log("check", "pass")
        # Maybe log aggregate stats here (e.g., geometric mean of means)
        return 0
    else:
        logger.log("check", "fail")
        return 112


def load_problem_modules(problem_type):
    """
    Dynamically loads the reference and task modules for the specified problem type.
    
    @param problem_type: A string indicating the problem type ('gemm' or 'moe').
    @return: A tuple of (reference_module, task_module, problem_dir).
    """
    base_dir = Path(__file__).parent
    problem_dir = base_dir / "problems" / problem_type
    
    if not problem_dir.exists():
        print(f"Error: Problem directory for '{problem_type}' not found at {problem_dir}", file=sys.stderr)
        sys.exit(1)
    
    # First import task module since reference depends on it
    task_path = problem_dir / "task.py"
    if not task_path.exists():
        print(f"Error: Task module not found at {task_path}", file=sys.stderr)
        sys.exit(1)
    
    # Temporarily add the problem directory to sys.path so task can be imported by reference
    original_path = sys.path.copy()
    sys.path.insert(0, str(problem_dir))
    
    try:
        # Import task module first
        task_spec = importlib.util.spec_from_file_location("task", task_path)
        task_module = importlib.util.module_from_spec(task_spec)
        # Add to sys.modules so reference.py can find it
        sys.modules['task'] = task_module
        task_spec.loader.exec_module(task_module)
        
        # Now import reference module
        reference_path = problem_dir / "reference.py"
        if not reference_path.exists():
            print(f"Error: Reference module not found at {reference_path}", file=sys.stderr)
            sys.exit(1)
        
        reference_spec = importlib.util.spec_from_file_location("reference", reference_path)
        reference_module = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference_module)
        
        return reference_module, task_module, problem_dir
    finally:
        # Restore the original sys.path
        sys.path = original_path
        # Clean up sys.modules if needed
        if 'task' in sys.modules and sys.modules['task'] == task_module:
            del sys.modules['task']


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for AMD FP8 MM competition.")
    parser.add_argument("mode", choices=["test", "benchmark", "leaderboard", "gen_submission"],
                        help="The mode to run in.")
    parser.add_argument("--prob", choices=["gemm", "moe"], default="gemm",
                        help="The problem type to evaluate (default: gemm).")
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    
    args = parser.parse_args()
    mode = args.mode
    problem_type = args.prob
    
    if mode == "gen_submission":
        print("Error: 'gen_submission' mode is deprecated.", file=sys.stderr)
        print("Please use 'python src/gen_submission.py' instead.", file=sys.stderr)
        local_arg = " --local" if args.local else ""
        print(f"Example: python src/gen_submission.py {problem_type}{local_arg}", file=sys.stderr)
        return 1  # Return an error code
    
    # Get problem directory
    base_dir = Path(__file__).parent
    problem_dir = base_dir / "problems" / problem_type
    
    if not problem_dir.exists():
        print(f"Error: Problem directory for '{problem_type}' not found at {problem_dir}", file=sys.stderr)
        sys.exit(1)
    
    # For module import in the main process (needed for TestSpec)
    sys.path.insert(0, str(problem_dir))
    try:
        # Import task module for TestSpec
        import task
        # Make the TestSpec class available globally if needed
        TestSpec = getattr(task, "TestSpec", dict)
    except ImportError:
        print(f"Error: Could not import task module from {problem_dir}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore path (though we'll need to add it back in the child processes)
        if sys.path[0] == str(problem_dir):
            sys.path.pop(0)
    
    seed = os.getenv("POPCORN_SEED")
    os.unsetenv("POPCORN_SEED")
    seed = int(seed) if seed else None
    set_seed(seed or 42)
    tests = get_test_cases_from_yaml(mode, seed, problem_dir)

    with PopcornOutput() as logger:
        import multiprocessing
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests, str(problem_dir))
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests, str(problem_dir))
            
            if mode == "leaderboard":
                # warmup
                run_single_benchmark(pool, tests[0], False, 100, 1e7, str(problem_dir))
                logger.log("benchmark-count", len(tests))
                passed = True
                for i in range(len(tests)):
                    result = run_single_benchmark(pool, tests[i], True, 100, 30e9, str(problem_dir))
                    logger.log(f"benchmark.{i}.spec", tests[i].spec)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{i}.{field.name}", getattr(result, field.name))
                    else:
                        passed = False
                        logger.log(f"benchmark.{i}.status", "fail")
                        logger.log(f"benchmark.{i}.error", str(result))
                        break

                logger.log("check", "pass" if passed else "fail")
            else:
                # TODO: Implement script and profile mode
                return 2


if __name__ == "__main__":
    # Ensure CUDA is available before starting if modes require it
    if len(sys.argv) > 1 and sys.argv[1] in ["test", "benchmark", "leaderboard"]:
        try:
            import torch
            if not torch.cuda.is_available():
                print("Error: CUDA is not available. Please check your ROCm/CUDA installation.", file=sys.stderr)
                sys.exit(1)
            if torch.cuda.device_count() == 0:
                 print("Error: No CUDA devices found.", file=sys.stderr)
                 sys.exit(1)
            # print(f"Found {torch.cuda.device_count()} CUDA devices. Using device 0.")
        except ImportError:
            print("Error: PyTorch is not installed or not found.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
             print(f"Error during CUDA check: {e}", file=sys.stderr)
             sys.exit(1)

    sys.exit(main())
