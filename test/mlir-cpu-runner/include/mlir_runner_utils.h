//===- mlir_runner_utils.h - Utils for debugging MLIR CPU execution -------===//
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
#ifndef MLIR_CPU_RUNNER_MLIRUTILS_H_
#define MLIR_CPU_RUNNER_MLIRUTILS_H_

#include <cstdint>

#ifdef _WIN32
#ifndef MLIR_RUNNER_UTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
/* We are building this library */
#define MLIR_RUNNER_UTILS_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define MLIR_RUNNER_UTILS_EXPORT __declspec(dllimport)
#endif
#endif
#else
#define MLIR_RUNNER_UTILS_EXPORT
#endif

/// StridedMemRef descriptor type with static rank.
template <typename T, int N> struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

/// StridedMemRef descriptor type specialized for rank 0.
template <typename T> struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;
};

template <typename StreamType, typename T, int N>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, N> &V) {
  static_assert(N > 0, "Expected N > 0");
  os << "Memref base@ = " << V.data << " rank = " << N
     << " offset = " << V.offset << " sizes = [" << V.sizes[0];
  for (unsigned i = 1; i < N; ++i)
    os << ", " << V.sizes[i];
  os << "] strides = [" << V.strides[0];
  for (unsigned i = 1; i < N; ++i)
    os << ", " << V.strides[i];
  os << "]";
}

template <typename StreamType, typename T>
void printMemRefMetaData(StreamType &os, StridedMemRefType<T, 0> &V) {
  os << "Memref base@ = " << V.data << " rank = 0"
     << " offset = " << V.offset;
}

extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_0d_f32(StridedMemRefType<float, 0> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_1d_f32(StridedMemRefType<float, 1> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_2d_f32(StridedMemRefType<float, 2> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_3d_f32(StridedMemRefType<float, 3> *M);
extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_4d_f32(StridedMemRefType<float, 4> *M);

template <typename T, int Dim, int... Dims> struct Vector {
  Vector<T, Dims...> vector[Dim];
};
template <typename T, int Dim> struct Vector<T, Dim> { T vector[Dim]; };

template <int D1, typename T> using Vector1D = Vector<T, D1>;
template <int D1, int D2, typename T> using Vector2D = Vector<T, D1, D2>;
template <int D1, int D2, int D3, typename T>
using Vector3D = Vector<T, D1, D2, D3>;
template <int D1, int D2, int D3, int D4, typename T>
using Vector4D = Vector<T, D1, D2, D3, D4>;

extern "C" MLIR_RUNNER_UTILS_EXPORT void
print_memref_vector_4x4xf32(StridedMemRefType<Vector2D<4, 4, float>, 2> *M);

#endif // MLIR_CPU_RUNNER_MLIRUTILS_H_
