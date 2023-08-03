/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_LIB_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_LIB_H_

#include <cstdint>
#include <vector>

namespace ml_adj {

/// Standard Types ///

// Length of axis.
typedef uint32_t dim_t;

// Dimensions of data.
typedef std::vector<dim_t> dims_t;

// 1d index into data.
typedef uint64_t ind_t;

// Integral type of data element.
enum etype_t : uint8_t {
  i32 = 0,
  f32 = 1,
  f64 = 2,
};

// Size in bytes of data element.
typedef uint8_t width_t;

}  // namespace ml_adj

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ML_ADJACENT_LIB_H_
