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
#include <string>

#include "fuzztest/fuzztest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Creates FuzzStringToNumber class that wraps a single operation node session.
class FuzzStringToNumber : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    tensorflow::ops::StringToNumber(scope.WithOpName("output"), op_node);
  }
  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tensorflow::tstring>()() = input_string;
    absl::Status s = RunInputsWithStatus({{"input", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};

// Setup up fuzzing test.
FUZZ_TEST_F(FuzzStringToNumber, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

}  // end namespace fuzzing
}  // end namespace tensorflow
