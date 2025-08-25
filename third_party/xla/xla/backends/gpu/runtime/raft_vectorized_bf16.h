/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_RAFT_VECTORIZED_BF16_H_
#define XLA_BACKENDS_GPU_RUNTIME_RAFT_VECTORIZED_BF16_H_

#pragma once
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "raft/util/vectorized.cuh"

namespace raft {

template <>
struct IOType<__nv_bfloat16, 1> {
  typedef __nv_bfloat16 Type;
};
template <>
struct IOType<__nv_bfloat16, 2> {
  typedef __nv_bfloat162 Type;
};
template <>
struct IOType<__nv_bfloat16, 4> {
  typedef uint2 Type;
};
template <>
struct IOType<__nv_bfloat16, 8> {
  typedef uint4 Type;
};
template <>
struct IOType<__nv_bfloat162, 1> {
  typedef __nv_bfloat162 Type;
};
template <>
struct IOType<__nv_bfloat162, 2> {
  typedef uint2 Type;
};
template <>
struct IOType<__nv_bfloat162, 4> {
  typedef uint4 Type;
};

}  // namespace raft

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAFT_VECTORIZED_BF16_H_
