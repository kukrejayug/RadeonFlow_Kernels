# MLA Intermediate Reference Values

This document describes all the intermediate reference values stored in the `MLAInput` struct after running `ref_kernel()`. These values can be used to debug and verify your MLA implementation step by step.

## Step 0: KV Cache Operations
- `kv_cache_step0_ref`: Updated KV cache after slicing and copying new values
  - Shape: `[batch_size, prefill+seq_len, kv_lora_dim+rope_dim]`
- `kv_lora_ref`: KV cache split for lora dimension  
  - Shape: `[batch_size, prefill+seq_len, kv_lora_dim]`
- `k_rope_ref`: KV cache split for rope dimension
  - Shape: `[batch_size, prefill+seq_len, rope_dim]`

## Step 1: Q Projection
- `q_lora_ref`: Query after down-projection
  - Shape: `[batch_size, seq_len, q_lora_dim]`

## Step 2: Weight Splits and Tensor Computations  
- `qup_nope_ref`: Query up-projection weights for nope dimension
  - Shape: `[n_heads, nope_dim, q_lora_dim]`
- `qup_rope_ref`: Query up-projection weights for rope dimension
  - Shape: `[n_heads, rope_dim, q_lora_dim]`
- `kup_nope_ref`: Key up-projection weights for nope dimension
  - Shape: `[n_heads, nope_dim, kv_lora_dim]`
- `vup_ref`: Value up-projection weights
  - Shape: `[n_heads, v_dim, kv_lora_dim]`
- `q_rope_step2_ref`: Query rope before permutation
  - Shape: `[batch_size, seq_len, n_heads, rope_dim]`
- `q_nope_ref`: Query nope before permutation
  - Shape: `[batch_size, seq_len, n_heads, nope_dim]`
- `q_absorb_ref`: Query absorbed with key nope weights
  - Shape: `[batch_size, seq_len, n_heads, kv_lora_dim]`

## Step 2.5: Permutations and RoPE
- `q_rope_permuted_ref`: Query rope after permutation
  - Shape: `[batch_size, n_heads, seq_len, rope_dim]`
- `q_absorb_permuted_ref`: Query absorb after permutation  
  - Shape: `[batch_size, n_heads, seq_len, kv_lora_dim]`
- `theta_ref`: Theta values for RoPE
  - Shape: `[rope_dim/2]`
- `q_rope_final_ref`: Query rope after RoPE application
  - Shape: `[batch_size, n_heads, seq_len, rope_dim]`
- `k_rope_final_ref`: Key rope after RoPE application
  - Shape: `[batch_size, prefill+seq_len, rope_dim]`

## Step 3: Attention Computation
- `attn_nope_ref`: Attention scores from nope part
  - Shape: `[batch_size, n_heads, seq_len, prefill+seq_len]`
- `attn_rope_ref`: Attention scores from rope part
  - Shape: `[batch_size, n_heads, seq_len, prefill+seq_len]`
- `scores_ref`: Combined attention scores before softmax
  - Shape: `[batch_size, n_heads, seq_len, prefill+seq_len]`
- `attention_ref`: Final attention weights after softmax
  - Shape: `[batch_size, n_heads, seq_len, prefill+seq_len]`

## Step 4: Output Computation
- `o_step4_0_ref`: Output after first einsum operation
  - Shape: `[batch_size, n_heads, seq_len, kv_lora_dim]`
- `o_step4_1_ref`: Output after second einsum operation  
  - Shape: `[batch_size, n_heads, seq_len, v_dim]`
- `output_ref`: Final output after wo projection
  - Shape: `[batch_size, seq_len, hidden_dim]`

## Usage Example

```cpp
// In your MLA kernel implementation
MLAInput* data = (MLAInput*)input;

// Run reference kernel to populate all intermediate values
ref_kernel(*data);

// Now you can compare your intermediate results with reference values
torch::Tensor my_q_lora = my_compute_q_lora(data->x, data->Q_proj_down_weight);
bool q_lora_correct = torch::allclose(my_q_lora, data->q_lora_ref, 1e-3, 1e-3);

torch::Tensor my_attention = my_compute_attention(...);
bool attention_correct = torch::allclose(my_attention, data->attention_ref, 1e-3, 1e-3);

// Print debugging info
if (!q_lora_correct) {
    std::cout << "Q lora mismatch!" << std::endl;
    std::cout << "Max diff: " << torch::max(torch::abs(my_q_lora - data->q_lora_ref)) << std::endl;
}
```

## Constants Reference
From `src/mla/mla.h`:
- `batch_size = 128`
- `hidden_dim = 7168` 
- `dq = 1536` (q_lora_dim)
- `seq_len = 1`
- `n_heads = 128`
- `kv_lora_dim = 512`
- `nope_dim = 128`
- `rope_dim = 64`
- `v_dim = 128`
- `max_seq_len = 8192` 