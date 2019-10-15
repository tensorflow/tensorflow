//===- mlir_runner_utils.cpp - Utils for MLIR CPU execution ---------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Utilities for interfacing MLIR types with C code as well as printing,
// debugging etc.
//
//===----------------------------------------------------------------------===//

#include "include/mlir_runner_utils.h"
#include <assert.h>
#include <iostream>

template <int... Dims> struct StaticSizeMult {
  static constexpr int value = 1;
};

template <int N, int... Dims> struct StaticSizeMult<N, Dims...> {
  static constexpr int value = N * StaticSizeMult<Dims...>::value;
};

static void printSpace(std::ostream &os, int count) {
  for (int i = 0; i < count; ++i) {
    os << ' ';
  }
}

template <typename T, int M, int... Dims> struct VectorDataPrinter {
  static void print(std::ostream &os, const Vector<T, M, Dims...> &val);
};

template <typename T, int M, int... Dims>
void VectorDataPrinter<T, M, Dims...>::print(std::ostream &os,
                                             const Vector<T, M, Dims...> &val) {
  static_assert(M > 0, "0 dimensioned tensor");
  static_assert(sizeof(val) == M * StaticSizeMult<Dims...>::value * sizeof(T),
                "Incorrect vector size!");
  // First
  os << "(" << val.vector[0];
  if (M > 1)
    os << ", ";
  if (sizeof...(Dims) > 1)
    os << "\n";
  // Kernel
  for (unsigned i = 1; i + 1 < M; ++i) {
    printSpace(os, 2 * sizeof...(Dims));
    os << val.vector[i] << ", ";
    if (sizeof...(Dims) > 1)
      os << "\n";
  }
  // Last
  printSpace(os, sizeof...(Dims));
  os << val.vector[M - 1] << ")";
}

template <typename T, int M, int... Dims>
std::ostream &operator<<(std::ostream &os, const Vector<T, M, Dims...> &v) {
  VectorDataPrinter<T, M, Dims...>::print(os, v);
  return os;
}

template <typename T, int N> struct MemRefDataPrinter {
  static void print(std::ostream &os, T *base, int64_t rank, int64_t offset,
                    int64_t *sizes, int64_t *strides);
  static void printFirst(std::ostream &os, T *base, int64_t rank,
                         int64_t offset, int64_t *sizes, int64_t *strides);
  static void printLast(std::ostream &os, T *base, int64_t rank, int64_t offset,
                        int64_t *sizes, int64_t *strides);
};

template <typename T> struct MemRefDataPrinter<T, 0> {
  static void print(std::ostream &os, T *base, int64_t rank, int64_t offset,
                    int64_t *sizes = nullptr, int64_t *strides = nullptr);
};

template <typename T>
static void printNewLineIfVector(std::ostream &os, T &t) {}

template <typename T, int Dim, int... Dims>
static void printNewLineIfVector(std::ostream &os, Vector<T, Dim, Dims...> &t) {
  os << "\n";
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::printFirst(std::ostream &os, T *base,
                                         int64_t rank, int64_t offset,
                                         int64_t *sizes, int64_t *strides) {
  os << "[";
  MemRefDataPrinter<T, N - 1>::print(os, base, rank, offset, sizes + 1,
                                     strides + 1);
  if (sizes[0] > 0) {
    os << ", ";
    printNewLineIfVector(os, *base);
  }
  if (N > 1)
    os << "\n";
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::print(std::ostream &os, T *base, int64_t rank,
                                    int64_t offset, int64_t *sizes,
                                    int64_t *strides) {
  printFirst(os, base, rank, offset, sizes, strides);
  for (unsigned i = 1; i + 1 < sizes[0]; ++i) {
    printSpace(os, rank - N + 1);
    MemRefDataPrinter<T, N - 1>::print(os, base, rank, offset + i * (*strides),
                                       sizes + 1, strides + 1);
    os << ", ";
    printNewLineIfVector(os, *base);
    if (N > 1)
      os << "\n";
  }
  printLast(os, base, rank, offset, sizes, strides);
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::printLast(std::ostream &os, T *base, int64_t rank,
                                        int64_t offset, int64_t *sizes,
                                        int64_t *strides) {
  printSpace(os, rank - N + 1);
  MemRefDataPrinter<T, N - 1>::print(os, base, rank,
                                     offset + (sizes[0] - 1) * (*strides),
                                     sizes + 1, strides + 1);
  os << "]";
}

template <typename T>
void MemRefDataPrinter<T, 0>::print(std::ostream &os, T *base, int64_t rank,
                                    int64_t offset, int64_t *sizes,
                                    int64_t *strides) {
  os << base[offset];
}

template <typename T, int N> void printMemRef(StridedMemRefType<T, N> &M) {
  static_assert(N > 0, "Expected N > 0");
  printMemRefMetaData(std::cout, M);
  std::cout << " data = " << std::endl;
  MemRefDataPrinter<T, N>::print(std::cout, M.data, N, M.offset, M.sizes,
                                 M.strides);
  std::cout << std::endl;
}

template <typename T> void printZeroDMemRef(StridedMemRefType<T, 0> &M) {
  std::cout << "\nMemref base@ = " << M.data << " rank = " << 0
            << " offset = " << M.offset << " data = [";
  MemRefDataPrinter<T, 0>::print(std::cout, M.data, 0, M.offset);
  std::cout << "]" << std::endl;
}

extern "C" void print_memref_0d_f32(StridedMemRefType<float, 0> *M) {
  printZeroDMemRef(*M);
}
extern "C" void print_memref_1d_f32(StridedMemRefType<float, 1> *M) {
  printMemRef(*M);
}
extern "C" void print_memref_2d_f32(StridedMemRefType<float, 2> *M) {
  printMemRef(*M);
}
extern "C" void print_memref_3d_f32(StridedMemRefType<float, 3> *M) {
  printMemRef(*M);
}
extern "C" void print_memref_4d_f32(StridedMemRefType<float, 4> *M) {
  printMemRef(*M);
}

extern "C" void
print_memref_vector_4x4xf32(StridedMemRefType<Vector2D<4, 4, float>, 2> *M) {
  printMemRef(*M);
}
