import os
import sys
import shutil
import subprocess
import traceback
import jinja2
import pathlib
from pathlib import Path
import re
import argparse
import bz2
import base64

def remove_cpp_comments(content):
    """
    Remove C++ style comments from the content using regex:
    1. Remove // line comments
    2. Remove /* */ block comments
    
    Args:
        content (str): The content with C++ comments
        
    Returns:
        str: Content with comments removed
    """
    # First remove block comments (/* */)
    # Non-greedy match for multi-line comments
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    
    # Then remove line comments (//), preserving the newline
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    
    return content

def process_content(content, source_path=None):
    """
    Process the source content to prepare it for the submission:
    1. Remove C++ comments
    2. Clean up excessive blank lines
    3. Handle any other necessary content processing
    
    Args:
        content (str): The raw content to process
        source_path: Optional path to the source file for logging
        
    Returns:
        str: The processed content
    """
    # Log what we're doing
    if source_path:
        print(f"  Processing content from {source_path}")
    
    
    # Clean up excessive blank lines
    while "\n\n\n" in content:
        content = content.replace("\n\n\n", "\n\n")
    
    return content

def clean_cuda_source(content):
    """
    Clean CUDA source content by:
    1. Removing all empty lines
    2. Removing leading whitespace from each line
    
    Args:
        content (str): The CUDA source content to clean
        
    Returns:
        str: The cleaned CUDA source content
    """
    # Split content into lines
    lines = content.split('\n')
    
    # Process each line to remove leading whitespace and filter out empty lines
    cleaned_lines = [line.lstrip() for line in lines if line.strip()]
    
    # Join lines back into a single string
    return '\n'.join(cleaned_lines)

def remove_static_asserts(content):
    """
    Remove all lines containing static_assert from the content.
    
    Args:
        content (str): The content to process
        
    Returns:
        str: Content with static_assert lines removed
    """
    lines = content.split('\n')
    filtered_lines = [line for line in lines if "static_assert" not in line]
    return '\n'.join(filtered_lines)

def gen_submission():
    try:
        try:
            import jinja2
        except ImportError:
            print("jinja2 not installed. Attempting to install it...")
            import subprocess
            subprocess.check_call(["pip", "install", "jinja2"])
            import jinja2
            print("jinja2 successfully installed")
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate submission file for different components')
        parser.add_argument('component', choices=['gemm', 'moe'], default='gemm', nargs='?', 
                            help='Component to generate (gemm or moe)')
        parser.add_argument('--local', action='store_true', help='Generate for local testing')
        
        # Extract args from sys.argv, but keep backward compatibility
        if "--local" in sys.argv and len(sys.argv) <= 2:
            # Only --local flag is present, use default component
            args = parser.parse_args([arg for arg in sys.argv[1:] if arg == "--local"])
            args.component = "gemm"  # Default
        else:
            # Parse normally
            args = parser.parse_args([arg for arg in sys.argv[1:]])
        
        component = args.component
        local_mode = args.local
        
        print(f"Generating submission for component: {component}")
        
        # Get current file's directory
        current_dir = Path(__file__).parent.absolute()
        
        # Backup original submission.py
        submission_path = current_dir / "submission.py"
        if submission_path.exists():
            shutil.copy(submission_path, current_dir / "submission.py.bak")
            print(f"Backed up original submission.py to submission.py.bak")
        
        # Common include files
        base_dir = current_dir.parent
        common_sources = [
            base_dir / "include" / "gpu_libs.h",
            base_dir / "include" / "gpu_types.h",
            base_dir / "tests" / "checker" / "metrics.h",
            base_dir / "src" / "utils" / "arithmetic.h",
        ]
        
        # Define component-specific source files
        component_sources = {
            "gemm": [
                base_dir / "src" / "gemm" / "gemm_kernel.cpp",
                base_dir / "src" / "gemm" / "transpose_kernel.cpp",
                base_dir / "src" / "gemm" / "gemm_kernel_legacy.cpp",
                base_dir / "src" / "gemm" / "gemm_launcher.cpp",

            ],
            "moe": [
                base_dir / "src" / "moe" / "moe.h",
                base_dir / "src" / "moe" / "moe_kernels.h",
                base_dir / "src" / "moe" / "gemm_thirdparty.cpp",
                base_dir / "src" / "moe" / "transpose.cpp",
                base_dir / "src" / "moe" / "moe_gemm_pipeline_kernel.cpp",
                base_dir / "src" / "moe" / "moe_topk_kernel.cpp",
                base_dir / "src" / "moe" / "moe.cpp",
            ]
        }
        
        # Combine common and component-specific sources
        sources = common_sources + component_sources[component]
        
        print(f"Reading files for {component} component")
        
        all_content = ""
        
        for source_path in sources:
            print(f"Processing file: {source_path}")
            with open(source_path, "r") as f:
                content = f.read()
                
                # Remove header includes, as we'll manually include them in the merged code
                content = "\n".join([line for line in content.split("\n") 
                                     if not line.strip().startswith("#include") and not line.strip().startswith("HOST_CODE_BELOW") and not line.strip().startswith("DEVICE_CODE_BELOW")])
                
                # Process the content (remove comments, clean up, etc.)
                # content = process_content(content, source_path)
                
                # # Escape \n strings in the original code
                # content = content.replace("\\n", "\\\\n")
                
                # # Escape double quotes in printf statements
                # if 'printf(' in content and '"' in content:
                #     content = content.replace('"', '\\"')
                # content = content.replace("#pragma once", "")
                all_content += f"\n\n// From {source_path.name}\n{content}"
        
        # Remove extern "C" wrapper block
        # all_content = all_content.replace("extern \"C\" {", "") 
        # all_content = all_content.replace("} // extern \"C\"", "")
        # Remove commits
        
        # Clean up excessive blank lines
        while "\n\n\n" in all_content:
            all_content = all_content.replace("\n\n\n", "\n\n")
        
        # Create jinja2 environment
        environment = jinja2.Environment()
        
        # Define common CUDA source template
        cuda_src_template = """#pragma once
{% if local_mode %}#define TEST_ON_RDNA4
{% endif %}
#include <stdio.h>         // For printf
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <hip/amd_detail/amd_warp_sync_functions.h>
#include <ck/utility/amd_buffer_addressing.hpp>
#include <ck/utility/data_type.hpp>
#include <hipcub/hipcub.hpp>
#include <hipblas-common/hipblas-common.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <rocwmma/rocwmma.hpp>
#include <ATen/hip/HIPContext.h>
#define PARAMETERIZE_LIBRARY

namespace wmma = rocwmma;
struct KernelTimer {
  KernelTimer(size_t ops, float *, float *) {
  }
  void start_timer(hipStream_t stream = 0) {
  }
  void stop_timer(hipStream_t stream = 0) {
  }
  
  void synchronize() {}
};

class KernelTimerScoped {

  public:
    KernelTimerScoped(std::vector<std::shared_ptr<KernelTimer>> &timers, size_t calc_ops, float *time, float *gflops,
                      hipStream_t stream = 0) {}

    ~KernelTimerScoped() {}
};

// Combined source code
{{ all_content }}

{% if component == "gemm" %}
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    at::hip::HIPStream stream = at::hip::getCurrentHIPStream();
    hipStream_t raw_stream = stream.stream();
    run(
        (__FP8_TYPE*)a.data_ptr(), 
        (__FP8_TYPE*)b.data_ptr(), 
        as.data_ptr<float>(), 
        bs.data_ptr<float>(), 
        (__BF16_TYPE*)c.data_ptr(), 
        m, n, k, nullptr, raw_stream
    );
}

{% endif %}"""
        
        # Render the CUDA source content
        cuda_src = environment.from_string(cuda_src_template).render(
            all_content=all_content,
            local_mode=local_mode,
            component=component
        )
        
        # Process the CUDA source content
        processed_cuda_src = cuda_src
        # processed_cuda_src = process_backslashes(cuda_src)
        # processed_cuda_src = remove_cpp_comments(processed_cuda_src)
        # processed_cuda_src = remove_static_asserts(processed_cuda_src)
        # processed_cuda_src = clean_cuda_source(processed_cuda_src)
        
        # Compress with bzip2 and encode with base64
        compressed_src = bz2.compress(processed_cuda_src.encode('utf-8'))
        compressed_src_b64 = base64.b64encode(compressed_src).decode('ascii')
        
        # Write to submission.hip
        hip_file_path = current_dir / "submission.hip"
        with open(hip_file_path, "w") as f:
            f.write(processed_cuda_src)
        print(f"Successfully created submission.hip at {hip_file_path} for {component}")
        
        # Create template for submission.py based on component
        if component == "gemm":
            template_str = """#!POPCORN leaderboard  amd-fp8-mm
# This script provides a template for using load_inline to run a HIP kernel for
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
import base64
import bz2
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

CPP_WRAPPER = \"\"\"
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
\"\"\"

# Compressed CUDA source code
CUDA_SRC_COMPRESSED = \"\"\"
{{ compressed_src_b64 }}
\"\"\"

# Decompress the CUDA source
CUDA_SRC = bz2.decompress(base64.b64decode(CUDA_SRC_COMPRESSED)).decode('utf-8')

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch={% if local_mode %}gfx1201{% else %}gfx942{% endif %}", "-std=c++20", "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__"],
    extra_cflags=["-Ofast", "-ffast-math", "-march=native", "-funroll-loops", "-fomit-frame-pointer"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c
"""
        elif component == "moe":
            template_str = """#!POPCORN leaderboard amd-mixture-of-experts
# This script provides a template for using load_inline to run a HIP kernel for MOE
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
from torch.utils.cpp_extension import load_inline
from torch import empty_like
from task import input_t, output_t
import base64
import bz2

CPP_WRAPPER = \"\"\"
void run_from_python(int seq_len, int batch_size, int d_hidden, int d_expert, int n_routed_experts,
                     int n_experts_per_token, int n_shared_experts, unsigned long long input_seq,
                     unsigned long long expert_scores, std::vector<unsigned long long> expert_weight_gate_p,
                     std::vector<unsigned long long> expert_weight_up_p,
                     std::vector<unsigned long long> expert_weight_down_p, unsigned long long shared_expert_weight_gate,
                     unsigned long long shared_expert_weight_up, unsigned long long shared_expert_weight_down,
                     unsigned long long router_weight, unsigned long long final_output);
\"\"\"

# Compressed CUDA source code
CUDA_SRC_COMPRESSED = \"\"\"
{{ compressed_src_b64 }}
\"\"\"

# Decompress the CUDA source
CUDA_SRC = bz2.decompress(base64.b64decode(CUDA_SRC_COMPRESSED)).decode('utf-8')

import os
import torch
os.environ["CXX"] = "clang++"

module = load_inline(
    name='run_from_python',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['run_from_python'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch={% if local_mode %}gfx1201{% else %}gfx942{% endif %}", "-std=c++20", "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__", "-DHIP_ENABLE_WARP_SYNC_BUILTINS"],
    extra_cflags=["-Ofast", "-ffast-math", "-march=native", "-funroll-loops", "-fomit-frame-pointer"],
)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, weights, config = data
    output_tensor = empty_like(input_tensor)
    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    n_routed_experts = config["n_routed_experts"]
    n_shared_experts = config["n_shared_experts"]
    n_experts_per_token = config["n_experts_per_token"]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    expert_weight_gate_p = []
    expert_weight_up_p = []
    expert_weight_down_p = []
    for i in range(n_routed_experts):
        expert_weight_gate_p.append(int(weights[f'experts.{i}.0.weight'].data_ptr()))
        expert_weight_up_p.append(int(weights[f'experts.{i}.1.weight'].data_ptr()))
        expert_weight_down_p.append(int(weights[f'experts.{i}.2.weight'].data_ptr()))
    
    shared_expert_weight_gate = int(weights['shared_experts.0.weight'].data_ptr())
    shared_expert_weight_up = int(weights['shared_experts.1.weight'].data_ptr())
    shared_expert_weight_down = int(weights['shared_experts.2.weight'].data_ptr())
    
    router_weight = int(weights['router.weight'].data_ptr())
    
    expert_scores = torch.matmul(input_tensor.view(-1, d_hidden), weights['router.weight'].transpose(0, 1)).contiguous()
    
    module.run_from_python(seq_len, batch_size, d_hidden, d_expert, n_routed_experts, n_experts_per_token, n_shared_experts, int(input_tensor.data_ptr()), int(expert_scores.data_ptr()), expert_weight_gate_p, expert_weight_up_p, expert_weight_down_p, shared_expert_weight_gate, shared_expert_weight_up, shared_expert_weight_down, router_weight, int(output_tensor.data_ptr()))
    
    return output_tensor
"""
        
        # Render the full submission template
        submission_template = environment.from_string(template_str)
        new_submission = submission_template.render(
            compressed_src_b64=compressed_src_b64,
            local_mode=local_mode,
        )
        
        # Process the rendered submission
        processed_submission = process_backslashes(new_submission)
        
        # Write new submission.py
        with open(submission_path, "w") as f:
            f.write(processed_submission)
            
        print(f"Successfully created new submission.py at {submission_path} for {component}")
        
        return 0
    except Exception as e:
        print(f"Error during submission generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

def process_backslashes(content):
    """
    Process the content to:
    1. Remove C++ comments from the content
    2. Join lines ending with a backslash with the following line (remove the backslash)
    3. Remove any lines containing backslash characters and log which lines were removed
    4. Remove only completely empty lines (no whitespace)
    
    Args:
        content (str): The content to process
        
    Returns:
        str: The processed content
    """
    
    lines = content.split('\n')
    processed_lines = []
    i = 0
    removed_lines = []
    
    while i < len(lines):
        line = lines[i]
        
        # Skip completely empty lines (no whitespace)
        if not line:
            i += 1
            continue
            
        # Normal line - keep it
        processed_lines.append(line)
        i += 1
    
    # Log removed lines
    if removed_lines:
        print("\nWarning: The following lines were removed:")
        for line_num, line_content in removed_lines:
            print(f"  Line {line_num}: {line_content}")
        print()
    
    return '\n'.join(processed_lines)

if __name__ == "__main__":
    print("Running submission generation script...")
    exit_code = gen_submission()

    if exit_code == 0:
        print("Submission generation completed successfully.")
    else:
        print("Submission generation failed.")

    sys.exit(exit_code)