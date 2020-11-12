#pragma once
#include <string>

#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 1024

/**
 * Class CudaVector
 * ----------------
 * 
 * Handles vector operations such as:
 * 1. Addition
 * 2. Division
 * 3. Subtraction
 * 4. Multiplication
 * 
 * All operations are done on GPU without the programmer
 * needing to specify how.
 * 
 */ 

template <class T>
class CudaVector {
 private:
  T* _array;                      // vector array
  unsigned long _size;            // size of vector array

 public:
  CudaVector(unsigned long size); // constructor
  ~CudaVector();                  // destructor

  /**
   * Member Functions
   */ 

  unsigned long size();           // returns size of array

  // Operator overloading
  T& operator[](unsigned long);
  CudaVector<T> operator+(const CudaVector<T>&);
  CudaVector<T> operator-(const CudaVector<T>&);
  CudaVector<T> operator*(const CudaVector<T>&);
  CudaVector<T> operator*(const double&);
  CudaVector<T> operator/(const double&);

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


// Including here to avoid import errors
#include "CudaVector.cu"
