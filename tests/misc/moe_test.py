#!POPCORN leaderboard amd-mixture-of-experts
# This script provides a template for using load_inline to run a HIP kernel for MOE
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
from torch.utils.cpp_extension import load_inline
from torch import empty_like
from typing import TypedDict, TypeVar, Tuple, Dict
import torch

input_t = TypeVar("input_t", bound=Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict])
output_t = TypeVar("output_t", bound=Tuple[torch.Tensor, Dict])


class TestSpec(TypedDict):
    d_hidden: int
    d_expert: int
    n_routed_experts: int
    n_shared_experts: int
    n_experts_per_token: int
    batch_size: int
    seq_len: int
    seed: int
import base64
import bz2
CPP_WRAPPER = """
void run_from_python(int seq_len, int batch_size, int d_hidden, int d_expert, int n_routed_experts,
                     int n_experts_per_token, int n_shared_experts, unsigned long long input_seq, unsigned long long expert_scores,
                     std::vector<unsigned long long> expert_weight_gate_p, std::vector<unsigned long long> expert_weight_up_p,
                     std::vector<unsigned long long> expert_weight_down_p, unsigned long long shared_expert_weight_gate,
                     unsigned long long shared_expert_weight_up, unsigned long long shared_expert_weight_down, unsigned long long router_weight,
                     unsigned long long final_output);
"""
# Compressed CUDA source code
CUDA_SRC_COMPRESSED = """
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
"""
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
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__"],
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



from utils import make_match_reference
from task import input_t, output_t
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math

# Reference code in PyTorch
class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        out = self.W_down(gate * self.W_up(x))
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(config["n_routed_experts"])
        ])
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(x_flat,
                                            flat_expert_indices,
                                            flat_expert_weights)

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self,
                  x: torch.Tensor,
                  flat_expert_indices: torch.Tensor,
                  flat_expert_weights: torch.Tensor
                 ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out    = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce='sum'
            )

        return expert_cache


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_dim]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    # Fill in the given weights of the model
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

    for i in range(num_experts):
        gate_proj_weight = weights[f'experts.{i}.0.weight']
        up_proj_weight = weights[f'experts.{i}.1.weight']
        down_proj_weight = weights[f'experts.{i}.2.weight']

        # Transpose weights to match expected shape for nn.Linear
        moe.experts[i].W_gate.weight = nn.Parameter(gate_proj_weight.t())
        moe.experts[i].W_up.weight = nn.Parameter(up_proj_weight.t())
        moe.experts[i].W_down.weight = nn.Parameter(down_proj_weight.t())

    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    output = moe(input_tensor)

    return output


# Input generation for the reference code

def generate_input(
    dhidden: int,
    dexpert: int,
    nroutedexperts: int,
    nsharedexperts: int,
    nexpertspertoken: int,
    bs: int,
    seqlen: int,
    seed: int
) -> input_t:

    # Really dumb but for now _ isn't parsing correctly.
    d_hidden = dhidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    n_shared_experts = nsharedexperts
    n_experts_per_token = nexpertspertoken
    batch_size = bs
    seq_len = seqlen

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "n_experts_per_token": n_experts_per_token,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    num_experts = n_routed_experts
    expert_dim = d_expert
    weights = {}

    input_tensor = torch.randn(
        (batch_size, seq_len, d_hidden),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ).contiguous()

    # Initialize router weights
    weights['router.weight'] = torch.randn(
        (num_experts, d_hidden),
        device="cuda",
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(d_hidden)

    for i in range(num_experts):
        weights[f'experts.{i}.0.weight'] = torch.randn(
            (d_hidden, expert_dim),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(expert_dim)

        weights[f'experts.{i}.1.weight'] = torch.randn(
            (d_hidden, expert_dim),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(expert_dim)

        weights[f'experts.{i}.2.weight'] = torch.randn(
            (expert_dim, d_hidden),
            device='cuda',
            dtype=torch.float16,
            generator=gen
        ) / math.sqrt(d_hidden)
    
    weights['shared_experts.0.weight'] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights['shared_experts.1.weight'] = torch.randn(
        (d_hidden, expert_dim * n_shared_experts),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(expert_dim * n_shared_experts)
    weights['shared_experts.2.weight'] = torch.randn(
        (expert_dim * n_shared_experts, d_hidden),
        device='cuda',
        dtype=torch.float16,
        generator=gen
    ) / math.sqrt(d_hidden)

    return (input_tensor, weights, config)


check_implementation = make_match_reference(ref_kernel, rtol=1e-2, atol=1e-2)

input = generate_input(dhidden=7168, dexpert=2048, nroutedexperts=32, nsharedexperts=1, nexpertspertoken=4, bs=1, seqlen=8192, seed=42)

output = ref_kernel(input)

print(output)