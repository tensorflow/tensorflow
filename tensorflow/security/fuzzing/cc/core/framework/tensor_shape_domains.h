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

#ifndef TENSORFLOW_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_
#define TENSORFLOW_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_

#include <limits>
#include <tuple>
#include <utility>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow::fuzzing {

/// Returns a fuzztest domain with valid TensorShapes.
/// The domain can be customized by setting the maximum rank,
/// and the minimum and maximum size of all dimensions.
fuzztest::Domain<TensorShape> AnyValidTensorShape(
    size_t max_rank = std::numeric_limits<int>::max(),
    int64_t dim_lower_bound = std::numeric_limits<int64_t>::min(),
    int64_t dim_upper_bound = std::numeric_limits<int64_t>::max());

}  // namespace tensorflow::fuzzing

#endif  // TENSORFLOW_SECURITY_FUZZING_CC_CORE_FRAMEWORK_TENSOR_SHAPE_DOMAINS_H_
