#pragma once
#include <string>

#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 1024

template <class T>
class CudaVector {
 private:
  T* _array;
  unsigned long _size;

  // kernel functions
  // __global__ void add(T*, T*, T*, unsigned long);

 public:
  CudaVector(unsigned long size);
  ~CudaVector();

  // Member functions
  unsigned long size();

  // Operator overloading
  T& operator[](unsigned long);
  CudaVector<T> operator+(const CudaVector<T>&);

  /**
   * Exception Class
   */
  class CudaVectorException : public std::exception {
   private:
    std::string message_;

   public:
    explicit CudaVectorException(const std::string& message);
    const char* what() const noexcept override { return message_.c_str(); }
  };
};

#include "CudaVector.cu"
