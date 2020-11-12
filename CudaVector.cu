#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "CudaVector.h"

/**
 * Kernel Functions
 */

template <class T>
__global__ void add(T* a, T* b, T* c, unsigned long n) {
  unsigned long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) c[index] = a[index] + b[index];
}

template <class T>
__global__ void subtract(T* a, T* b, T* c, unsigned long n) {
  unsigned long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) c[index] = a[index] - b[index];
}

template <class T>
__global__ void dotProduct(T* a, T* b, T* c, unsigned long n) {
  unsigned long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) c[index] = a[index] * b[index];
}

template <class T>
__global__ void constantMultiply(T* a, double b, T* c, unsigned long n) {
  unsigned long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) c[index] = a[index] * b;
}

template <class T>
__global__ void contantDivide(T* a, double b, T* c, unsigned long n) {
  unsigned long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) c[index] = a[index] / b;
}

/** =================
 *  Class definitions
 *  =================
 */

/**
 * Creates CudaVector object and sets the size
 */
template <class T>
CudaVector<T>::CudaVector(unsigned long size) {
  this->_array = new T[size];
  this->_size = size;
}

/**
 * Free up memory
 */
template <class T>
CudaVector<T>::~CudaVector() {
  delete[] _array;
}

/**
 * Adds vector in GPU
 *
 * Returns CudaVector object
 */
template <class T>
CudaVector<T> CudaVector<T>::operator+(const CudaVector<T>& a) {
  if (this->_size != a._size)
    throw CudaVectorException("Vector size not same!");

  // create result array
  CudaVector<T> result(this->_size);

  T *d_a, *d_b, *d_c;
  const size_t size = this->_size * sizeof(T);

  // allocate memory
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  // copy memory to device
  cudaMemcpy(d_a, this->_array, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, a._array, size, cudaMemcpyHostToDevice);

  add<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, d_b, d_c, this->_size);

  // copy to host
  cudaMemcpy(result._array, d_c, size, cudaMemcpyDeviceToHost);

  return result;
}

/**
 * Subtract vector in GPU
 *
 * Returns CudaVector object
 */
template <class T>
CudaVector<T> CudaVector<T>::operator-(const CudaVector<T>& a) {
  if (this->_size != a._size)
    throw CudaVectorException("Vector size not same!");

  // create result array
  CudaVector<T> result(this->_size);

  T *d_a, *d_b, *d_c;
  const size_t size = this->_size * sizeof(T);

  // allocate memory
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  // copy memory to device
  cudaMemcpy(d_a, this->_array, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, a._array, size, cudaMemcpyHostToDevice);

  subtract<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, d_b, d_c, this->_size);

  // copy to host
  cudaMemcpy(result._array, d_c, size, cudaMemcpyDeviceToHost);

  return result;
}

/**
 * Dot Product vector in GPU
 *
 * Returns CudaVector object
 */
template <class T>
CudaVector<T> CudaVector<T>::operator*(const CudaVector<T>& a) {
  if (this->_size != a._size)
    throw CudaVectorException("Vector size not same!");

  // create result array
  CudaVector<T> result(this->_size);

  T *d_a, *d_b, *d_c;
  const size_t size = this->_size * sizeof(T);

  // allocate memory
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_b, size);
  cudaMalloc((void**)&d_c, size);

  // copy memory to device
  cudaMemcpy(d_a, this->_array, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, a._array, size, cudaMemcpyHostToDevice);

  dotProduct<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, d_b, d_c, this->_size);

  // copy to host
  cudaMemcpy(result._array, d_c, size, cudaMemcpyDeviceToHost);

  return result;
}

/**
 * Constant Multiplication of vector in GPU
 *
 * Returns CudaVector object
 */
template <class T>
CudaVector<T> CudaVector<T>::operator*(const double& a) {
  // create result array
  CudaVector<T> result(this->_size);

  T *d_a, *d_c;
  const size_t size = this->_size * sizeof(T);

  // allocate memory
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_c, size);

  // copy memory to device
  cudaMemcpy(d_a, this->_array, size, cudaMemcpyHostToDevice);

  constantMultiply<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, a, d_c,
                                                          this->_size);

  // copy to host
  cudaMemcpy(result._array, d_c, size, cudaMemcpyDeviceToHost);

  return result;
}

/**
 * Constant Division of vector in GPU
 *
 * Returns CudaVector object
 */
template <class T>
CudaVector<T> CudaVector<T>::operator/(const double& a) {
  // create result array
  CudaVector<T> result(this->_size);

  T *d_a, *d_c;
  const size_t size = this->_size * sizeof(T);

  // allocate memory
  cudaMalloc((void**)&d_a, size);
  cudaMalloc((void**)&d_c, size);

  // copy memory to device
  cudaMemcpy(d_a, this->_array, size, cudaMemcpyHostToDevice);

  contantDivide<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_a, a, d_c,
                                                       this->_size);

  // copy to host
  cudaMemcpy(result._array, d_c, size, cudaMemcpyDeviceToHost);

  return result;
}

/**
 * Subscript operator overloading
 *
 * Returns CudaVector object
 */
template <class T>
T& CudaVector<T>::operator[](unsigned long i) {
  if (i > this->_size || i < 0)
    throw CudaVectorException("Index out of bounds!");

  return this->_array[i];
}

template <class T>
unsigned long CudaVector<T>::size() {
  return this->_size;
}

/**
 * Exception Class definition
 */

template <class T>
CudaVector<T>::CudaVectorException::CudaVectorException(
    const std::string& message)
    : message_(message) {}
