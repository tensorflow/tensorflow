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

template <typename T, int N> struct MemRefDataPrinter {
  static void print(T *base, int64_t rank, int64_t offset, int64_t *sizes,
                    int64_t *strides);
  static void printFirst(T *base, int64_t rank, int64_t offset, int64_t *sizes,
                         int64_t *strides);
  static void printLast(T *base, int64_t rank, int64_t offset, int64_t *sizes,
                        int64_t *strides);
};

template <typename T> struct MemRefDataPrinter<T, 0> {
  static void print(T *base, int64_t rank, int64_t offset,
                    int64_t *sizes = nullptr, int64_t *strides = nullptr);
};

static void printSpace(int count) {
  for (int i = 0; i < count; ++i) {
    std::cout << " ";
  }
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::printFirst(T *base, int64_t rank, int64_t offset,
                                         int64_t *sizes, int64_t *strides) {
  std::cout << "[";
  MemRefDataPrinter<T, N - 1>::print(base, rank, offset, sizes + 1,
                                     strides + 1);
  if (sizes[0] > 0)
    std::cout << ", ";
  if (N > 1)
    std::cout << "\n";
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::printLast(T *base, int64_t rank, int64_t offset,
                                        int64_t *sizes, int64_t *strides) {
  printSpace(rank - N + 1);
  MemRefDataPrinter<T, N - 1>::print(
      base, rank, offset + (sizes[0] - 1) * (*strides), sizes + 1, strides + 1);
  std::cout << "]";
}

template <typename T, int N>
void MemRefDataPrinter<T, N>::print(T *base, int64_t rank, int64_t offset,
                                    int64_t *sizes, int64_t *strides) {
  printFirst(base, rank, offset, sizes, strides);
  for (unsigned i = 1; i + 1 < sizes[0]; ++i) {
    printSpace(rank - N + 1);
    MemRefDataPrinter<T, N - 1>::print(base, rank, offset + i * (*strides),
                                       sizes + 1, strides + 1);
    std::cout << ", ";
    if (N > 1)
      std::cout << "\n";
  }
  printLast(base, rank, offset, sizes, strides);
}

template <typename T>
void MemRefDataPrinter<T, 0>::print(T *base, int64_t rank, int64_t offset,
                                    int64_t *sizes, int64_t *strides) {
  std::cout << base[offset];
}

template <typename T, int N> void printMemRef(StridedMemRefType<T, N> &M) {
  static_assert(N > 0, "Expected N > 0");
  printMemRefMetaData(std::cout, M);
  std::cout << " data = " << std::endl;
  MemRefDataPrinter<T, N>::print(M.data, N, M.offset, M.sizes, M.strides);
  std::cout << std::endl;
}

template <typename T> void printZeroDMemRef(StridedMemRefType<T, 0> &M) {
  std::cout << "\nMemref base@ = " << M.data << " rank = " << 0
            << " offset = " << M.offset << " data = [";
  MemRefDataPrinter<T, 0>::print(M.data, 0, M.offset);
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
