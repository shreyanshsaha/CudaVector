#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "CudaVector.h"


// Kernel Function
template<class T>
__global__ void add(T* a, T*b, T*c, unsigned long n){
  unsigned long index = threadIdx.x + blockIdx.x*blockDim.x;
  if(index < n)
    c[index] = a[index] + b[index];
}





template <class T>
CudaVector<T>::CudaVector(unsigned long size) {
  this->_array = new T[size];
  this->_size = size;
}

template <class T>
CudaVector<T>::~CudaVector(){
  delete[] _array;
}

template <class T>
CudaVector<T> CudaVector<T>::operator+(const CudaVector<T>& a){
  if (this->_size != a._size) throw CudaVectorException("Vector size not same!");

  CudaVector<T> result(this->_size);

  T *d_a, *d_b, *d_c;
  const size_t size = this->_size * sizeof(T);
  
  // allocate memory
  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);
  cudaMalloc((void**) &d_c, size);

  // copy memory to device
  cudaMemcpy(d_a, this->_array, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, a._array, size, cudaMemcpyHostToDevice);

  add<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, d_b, d_c, this->_size);

  // copy to host
  cudaMemcpy(result._array, d_c, size, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < this->_size; i++) 
  //   result._array[i] = this->_array[i] + a._array[i];

  return result;
}

template <class T>
T& CudaVector<T>::operator[](unsigned long i) {
  if (i > this->_size) throw CudaVectorException("Index out of bounds!");

  return this->_array[i];
}

template<class T>
unsigned long CudaVector<T>::size(){
  return this->_size;
}



/**
 * Exception
 */

template<class T>
CudaVector<T>::CudaVectorException::CudaVectorException(const std::string& message)
    : message_(message) {}
