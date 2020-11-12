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

### Supported operations
1. Vector Addition
2. Vector Substraction
3. Vector dot product
4. Multiplication with constantt
5. Division with constant

## Example usage
```cpp

// Two vectors of size 10
CudaVector<int> a(10);
CudaVector<int> b(10);

CudaVector<int> c = a + b;

for(int i=0; i<c.size(); i++)
  std::cout<<c[i]<<", ";
cout<<std::endl;
```
