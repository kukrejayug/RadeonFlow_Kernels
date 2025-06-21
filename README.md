# RadeonFlow Kernels

![RadeonFlow Kernels](https://img.shields.io/badge/Project%20Status-Active-brightgreen) ![GitHub Releases](https://img.shields.io/badge/Releases-v1.0.0-blue)

🏆 **Grand Prize Winner Project for AMD Developer Challenge 2025**

### News 🔥

- **[2025/06]** [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) now provides free AMD Instinct Accelerators. You can try out our project using their MI300X.
  
- **[2025/06]** FP8-GEMM, MoE, and MLA kernels are open-sourced now!

## Introduction

This project contains implementations for three GPU kernels specifically tuned for the AMD Instinct MI300X Accelerator. The kernels include:

- **FP8 Blockwise-scaled GEMM**
- **Mixture of Experts (MoE)**
- **Multi-Layer Attention (MLA)**

These GPU kernels achieve high performance, providing at least an 8x speedup for FP8 GEMM and the other kernels compared to the reference implementation written in PyTorch from AMD. This performance led us to win the grand prize for the AMD Developer Challenge 2025: Inference Sprint.

For detailed implementation insights, refer to our [Technical Report](./TechnicalReport.md).

## Project Structure

We implemented the FP8-GEMM and MoE kernels in HIP, while the MLA kernel is in PyTorch. You can find the source code for the first two kernels in the `src` directory.

### Directory Layout

```
RadeonFlow_Kernels/
├── src/
│   ├── fp8_gemm/
│   │   ├── fp8_gemm.h
│   │   ├── fp8_gemm.cpp
│   ├── moe/
│   │   ├── moe.h
│   │   ├── moe.cpp
│   ├── mla/
│   │   ├── mla.py
├── README.md
├── TechnicalReport.md
```

## Installation

To get started with RadeonFlow Kernels, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kukrejayug/RadeonFlow_Kernels.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd RadeonFlow_Kernels
   ```

3. **Build the project:**

   For HIP kernels, ensure you have the ROCm environment set up. You can build the HIP kernels with:

   ```bash
   make
   ```

   For the MLA kernel, ensure you have PyTorch installed. You can install it via pip:

   ```bash
   pip install torch
   ```

## Usage

To run the kernels, you will need to have access to an AMD Instinct MI300X Accelerator. After building the project, you can execute the kernels as follows:

### FP8-GEMM

```bash
./bin/fp8_gemm
```

### MoE

```bash
./bin/moe
```

### MLA

```bash
python mla.py
```

## Performance Benchmarking

We conducted extensive benchmarking to evaluate the performance of our kernels. The results demonstrate significant improvements over the reference implementation.

### Benchmark Results

| Kernel      | Speedup (vs. PyTorch) |
|-------------|-----------------------|
| FP8-GEMM   | 8x                    |
| MoE        | 8x                    |
| MLA        | 8x                    |

For detailed benchmarking methodology and results, refer to our [Technical Report](./TechnicalReport.md).

## Contributions

We welcome contributions to enhance the RadeonFlow Kernels project. If you would like to contribute, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch:**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Make your changes and commit them:**

   ```bash
   git commit -m "Add YourFeature"
   ```

4. **Push to the branch:**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a pull request.**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank AMD for their support and for hosting the Developer Challenge. Their resources and feedback greatly contributed to the success of this project.

## Links

For the latest releases, visit our [Releases](https://github.com/kukrejayug/RadeonFlow_Kernels/releases) section. You can download the latest version and execute the files to explore the kernels.

## Conclusion

RadeonFlow Kernels aims to push the boundaries of GPU performance on AMD hardware. With our open-source approach, we hope to foster collaboration and innovation in the field of high-performance computing.

For further updates, please check the [Releases](https://github.com/kukrejayug/RadeonFlow_Kernels/releases) section regularly.