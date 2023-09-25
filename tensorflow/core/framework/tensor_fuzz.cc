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
#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/security/fuzzing/cc/core/framework/datatype_domains.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_shape_domains.h"
#include "tsl/lib/core/status_test_util.h"

namespace tensorflow::fuzzing {
namespace {

void BuildTensorAlwaysSucceedsWithValidTensorShape(DataType type,
                                                   const TensorShape& shape) {
  Tensor out;
  Status status = Tensor::BuildTensor(type, shape, &out);
  TF_EXPECT_OK(status);
}
FUZZ_TEST(TensorFuzz, BuildTensorAlwaysSucceedsWithValidTensorShape)
    .WithDomains(AnyValidDataType(),
                 AnyValidTensorShape(/*max_rank=*/3, /*dim_lower_bound=*/0,
                                     /*dim_upper_bound=*/10));

void DebugStringCheck(const Tensor& tensor) {
  string out = tensor.DeviceSafeDebugString();
}
FUZZ_TEST(TensorFuzz, DebugStringCheck)
    .WithDomains(
        AnyValidNumericTensor(AnyValidTensorShape(/*max_rank=*/3,
                                                  /*dim_lower_bound=*/0,
                                                  /*dim_upper_bound=*/10),
                              AnyValidDataType()));

}  // namespace
}  // namespace tensorflow::fuzzing
