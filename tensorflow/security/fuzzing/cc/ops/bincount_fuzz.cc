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

// Creates FuzzBincount class that wraps a single operation node session.
class FuzzBincount : public FuzzSession<Tensor, int32, Tensor> {
  void BuildGraph(const Scope& scope) override {
    auto arr = tensorflow::ops::Placeholder(scope.WithOpName("arr"), DT_INT32);
    auto size =
        tensorflow::ops::Placeholder(scope.WithOpName("size"), DT_INT32);
    auto weights =
        tensorflow::ops::Placeholder(scope.WithOpName("weights"), DT_INT32);
    tensorflow::ops::Bincount(scope.WithOpName("output"), arr, size, weights);
  }
  void FuzzImpl(const Tensor& arr, const int32& nbins,
                const Tensor& weights) final {
    Tensor size(DT_INT32, {});
    size.flat<int32>()(0) = nbins;

    Status s = RunInputsWithStatus(
        {{"arr", arr}, {"size", size}, {"weights", weights}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};

// Setup up fuzzing test.
// TODO(unda, b/275737422): Make the values in arr be within [0, size) with high
// chance
FUZZ_TEST_F(FuzzBincount, Fuzz)
    .WithDomains(fuzzing::AnyValidNumericTensor(fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/5,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/10),
                                                fuzztest::Just(DT_INT32)),
                 fuzztest::InRange<int32>(0, 10),
                 fuzzing::AnyValidNumericTensor(fuzzing::AnyValidTensorShape(
                                                    /*max_rank=*/5,
                                                    /*dim_lower_bound=*/0,
                                                    /*dim_upper_bound=*/10),
                                                fuzztest::Just(DT_INT32)));

}  // end namespace fuzzing
}  // end namespace tensorflow
