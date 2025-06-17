#include <hip/hip_runtime.h>

// bs
constexpr int BatchSize = 128;
// sq
constexpr int Decode = 1;
// head
constexpr int NumHeads = 128;
// dim
// q:   hidden -> qlora -> nhead * (qnope + qrope)
// kv:  hidden -> kvlora + krope -> nhead * (knope + v) + krope -> nhead * (knope + krope), nhead * v
// all qrope heads share one krope head
constexpr int HiddenDim = 7168;
constexpr int KvLoraDim = 512;
constexpr int QLoraDim = 1536;
constexpr int NopeHeadDim = 128;
constexpr int RopeHeadDim = 64;
constexpr int VHeadDim = 128;

struct mla_arg_t {
    void *stream;
    // input
    int prefill;
    void *hidden;   // [bs, 1, hidden_dim]
    void *kvcache;  // [bs, prefill, kv_lora_dim]
    // weight
    void *qdown;    // [hidden_dim, q_lora_dim]
    void *qup;      // [q_lora_dim, n_head, nope_dim + rope_dim]
    void *kvdown;   // [hidden_dim, kv_lora_dim + rope_dim]
    void *kvup;     // [kv_lora_dim, n_head, nope_dim + v_dim]
    void *wo;       // [n_head * v_dim, hidden_dim]
    // output
    void *out;      // [bs, hidden_dim]
};

__global__ void mla_kernel() {

}

void launch_mla(mla_arg_t &args) {
    dim3 grid{BatchSize, Decode, NumHeads};
}
