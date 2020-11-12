# CudaVector

### Requirements

1. CUDA Toolkit

In linux execute the following to install:

```bash
sudo apt install nvidia-cuda-toolkit
```

### Compiling

If you are importing this library in your code, compile your code using nvcc.

Sample:

```bash
nvcc myprogram.cpp -o output.exe
```