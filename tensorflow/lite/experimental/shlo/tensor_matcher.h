/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_MATCHER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_MATCHER_H_

#include <gmock/gmock.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {
namespace testing {

MATCHER_P(TensorEq, tensor, "") {
  if (!::testing::ExplainMatchResult(::testing::Eq(tensor.type), arg.type,
                                     result_listener)) {
    return false;
  }
  switch (tensor.StorageType()) {
    case DataType::kI1:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kI1>()),
          arg.template Flat<DataType::kI1>(), result_listener);
    case DataType::kSI4:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kSI4>()),
          arg.template Flat<DataType::kSI4>(), result_listener);
    case DataType::kSI8:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kSI8>()),
          arg.template Flat<DataType::kSI8>(), result_listener);
    case DataType::kSI16:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kSI16>()),
          arg.template Flat<DataType::kSI16>(), result_listener);
    case DataType::kSI32:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kSI32>()),
          arg.template Flat<DataType::kSI32>(), result_listener);
    case DataType::kBF16:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kBF16>()),
          arg.template Flat<DataType::kBF16>(), result_listener);
    case DataType::kF16:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kF16>()),
          arg.template Flat<DataType::kF16>(), result_listener);
    case DataType::kF32:
      return ::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::Eq(),
                               tensor.template Flat<DataType::kF32>()),
          arg.template Flat<DataType::kF32>(), result_listener);
  }
}

}  // namespace testing
}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_TENSOR_MATCHER_H_
