// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_shape_domains.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Creates FuzzConcat class that wraps a single operation node session.
class FuzzConcat : public FuzzSession<Tensor, Tensor, int32> {
  void BuildGraph(const Scope& scope) override {
    auto value1 =
        tensorflow::ops::Placeholder(scope.WithOpName("value1"), DT_INT32);
    Input value1_input(value1);
    auto value2 =
        tensorflow::ops::Placeholder(scope.WithOpName("value2"), DT_INT32);
    Input value2_input(value2);
    InputList values_input_list({value1_input, value2_input});
    auto axis =
        tensorflow::ops::Placeholder(scope.WithOpName("axis"), DT_INT32);
    tensorflow::ops::Concat(scope.WithOpName("output"), values_input_list,
                            axis);
  }
  void FuzzImpl(const Tensor& value1, const Tensor& value2,
                const int32& axis) final {
    Tensor axis_tensor(DT_INT32, {});
    axis_tensor.scalar<int32_t>()() = axis;
    Status s = RunInputsWithStatus(
        {{"value1", value1}, {"value2", value2}, {"axis", axis_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.error_message();
    }
  }
};

// Setup up fuzzing test.
FUZZ_TEST_F(FuzzConcat, Fuzz)
    .WithDomains(fuzzing::AnyValidTensor(fuzzing::AnyValidTensorShape(
                                             /*max_rank=*/5,
                                             /*dim_lower_bound=*/0,
                                             /*dim_upper_bound=*/10),
                                         fuzztest::Just(DT_INT32)),
                 fuzzing::AnyValidTensor(fuzzing::AnyValidTensorShape(
                                             /*max_rank=*/5,
                                             /*dim_lower_bound=*/0,
                                             /*dim_upper_bound=*/10),
                                         fuzztest::Just(DT_INT32)),
                 fuzztest::InRange<int32>(0, 6));

}  // end namespace fuzzing
}  // end namespace tensorflow
