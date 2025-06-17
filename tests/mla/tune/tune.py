import os

# workaround a rocblas memory leak that causes OOM
# see: https://github.com/pytorch/pytorch/issues/138532
os.putenv('ROCBLAS_DEVICE_MEMORY_SIZE', '32000000')
os.putenv('HIPBLASLT_WORKSPACE_SIZE', '262144')

os.putenv('PYTORCH_TUNABLEOP_ENABLED', '1')
os.putenv('PYTORCH_TUNABLEOP_TUNING', '1')
os.putenv('PYTORCH_TUNABLEOP_RECORD_UNTUNED', '0')
os.putenv('PYTORCH_TUNABLEOP_VERBOSE', '3')
os.putenv('PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE', '0')
os.putenv('PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED', '1')
os.putenv('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '100')

import torch.cuda.tunable as tunable

tunable.tune_gemm_in_file("tunableop_untuned0.csv")
