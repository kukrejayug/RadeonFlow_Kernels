#pragma once
constexpr int MAX_SPLITK_FACTOR = 1;
struct InputParams {
  int batch_size;
  int seq_len;
  int d_hidden;
  int d_expert;
  int n_routed_experts;
  int n_experts_per_token;
  int n_shared_experts;
  int seed;
};

struct MoEInputRaw {
  InputParams params;
  void **expert_weight_gate;
  void **expert_weight_up;
  void **expert_weight_down;

  void *expert_weight_gate_transposed;
  void *expert_weight_up_transposed;
  void *expert_weight_down_transposed;

  void *input_seq;

  void *shared_expert_weight_gate;
  void *shared_expert_weight_gate_transposed;
  void *shared_expert_weight_up;
  void *shared_expert_weight_up_transposed;
  void *shared_expert_weight_down;
  void *shared_expert_weight_down_transposed;

  void *router_weight;

  void *expert_scores;

  void *seq_experts_index;
  void *seq_experts_softmax;

  void *source_rows;

  void *permuted_experts;
  void *permuted_rows;

  void *first_token_offset;

  void *expanded_seq_input;
  void *expanded_seq_softmax;
  void *rev_permuted_rows;

  void *expanded_fc1_output;
  void *expanded_fc1_gate_splitk;
  void *expanded_fc1_up_splitk;
  void *expanded_fc2_output;
  void *expanded_fc2_splitk;

  void *shared_fc1_output;
  void *shared_fc1_gate_splitk;
  void *shared_fc1_up_splitk;
  void *shared_fc2_output;
  void *shared_fc2_splitk;

  void *final_output;
  void *final_output_splitk;
};