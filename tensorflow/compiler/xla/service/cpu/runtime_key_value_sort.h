/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"

extern "C" {

// Each entry in 'values' represents a 3-dimensional shape with dimensions
// [a, b, c]. The 'b' dimension of the first shape is sorted into ascending
// order according to the results of comparisons using the provided 'less_than'
// function. 'values_count' must be > 0 and specifies the number of entries in
// 'values' and 'values_primitive_type_size_in_bytes'. The size of the primitive
// type of the i-th shape has exactly 'values_primitive_type_size_in_bytes[i]'
// bytes. The elements in each 'values' shape are reordered in the same way
// according to the comparisons using the first shape.
extern void __xla_cpu_runtime_KeyValueSort(
    tensorflow::int64 a, tensorflow::int64 b, tensorflow::int64 c,
    char** values, tensorflow::int32 values_count,
    tensorflow::int32* values_primitive_type_size_in_bytes,
    bool (*less_than)(char*, char*));
}

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_KEY_VALUE_SORT_H_
