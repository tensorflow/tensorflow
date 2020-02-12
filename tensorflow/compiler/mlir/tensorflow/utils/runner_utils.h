/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_RUNNER_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_RUNNER_UTILS_H_

#include <assert.h>
#include <cstdint>
#include <string>
#include <iostream>

#include "mlir/ExecutionEngine/CRunnerUtils.h"  // TF:llvm-project


extern "C" 
void _global_unique_ids(
    StridedMemRefType<int64_t, 1> *intput_ids,
    StridedMemRefType<int64_t, 0> *id_count,
    StridedMemRefType<int64_t, 1> *output_ids); 

extern "C" 
void _global_unique_index32(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *ids_index); 

extern "C" 
void _global_unique_index64(
    StridedMemRefType<int64_t, 1> *ids,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *ids_index); 

extern "C" 
void _global_unique_i64_i64(
    StridedMemRefType<int64_t, 1> *input,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *idx);

extern "C" 
void _global_unique_i64_i32(
    StridedMemRefType<int64_t, 1> *input,
    StridedMemRefType<int64_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *idx);

extern "C" 
void _global_unique_i32_i64(
    StridedMemRefType<int32_t, 1> *input,
    StridedMemRefType<int32_t, 1> *unique_ids,
    StridedMemRefType<int64_t, 1> *idx);

extern "C" 
void _global_unique_i32_i32(
    StridedMemRefType<int32_t, 1> *input,
    StridedMemRefType<int32_t, 1> *unique_ids,
    StridedMemRefType<int32_t, 1> *idx);

#endif TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_RUNNER_UTILS_H_
