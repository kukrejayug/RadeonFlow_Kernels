#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

struct mla_arg_t {
    void *stream;
    // input
    int prefill;
    void *hidden;   // [bs, 1, hidden_dim]
    void *kvcache;  // [bs, prefill, kv_lora_dim]
    // weight
    void *qdown;    // [hidden_dim, q_lora_dim]
    void *qup;      // [hidden_dim, kv_lora_dim + rope_dim]
    void *kvdown;   // [q_lora_dim, n_head, nope_dim + rope_dim]
    void *kvup;     // [kv_lora_dim, n_head, nope_dim + v_dim]
    void *wo;       // [bs, n_head * v_dim, hidden_dim]
    // output
    void *out;      // [bs, hidden_dim]
};

void launch_mla(mla_arg_t &args);


PYBIND11_MODULE(mla, m) {
    m.def(
        "mla_decode",
        [](int prefill, torch::Tensor &hidden, torch::Tensor &kvcache, torch::Tensor &qdown, torch::Tensor &qup, torch::Tensor &kvdown, torch::Tensor &kvup, torch::Tensor &wo, torch::Tensor &out) {
            auto stream = at::cuda::getCurrentHIPStream().stream();
            mla_arg_t args = {
                stream,
                prefill,
                hidden.data_ptr(),
                kvcache.data_ptr(),
                qdown.data_ptr(),
                qup.data_ptr(),
                kvdown.data_ptr(),
                kvup.data_ptr(),
                wo.data_ptr(),
                out.data_ptr()
            };
            launch_mla(args);
        },
        "Launch mla decode kernel"
    );
}
