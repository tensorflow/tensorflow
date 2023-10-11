// Copyright 2023 Google LLC
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

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzStringOpsStringSplit : public FuzzSession<std::string, std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto op_node2 =
        tensorflow::ops::Placeholder(scope.WithOpName("delimiter"), DT_STRING);

    tensorflow::ops::StringSplit(scope.WithOpName("output"), op_node, op_node2);
  }

  void FuzzImpl(const std::string& input_string,
                const std::string& separator_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, {2});

    auto svec = input_tensor.flat<tstring>();
    svec(0) = input_string.c_str();
    svec(1) = input_string.c_str();

    Tensor separator_tensor(tensorflow::DT_STRING, TensorShape({}));
    separator_tensor.scalar<tensorflow::tstring>()() = separator_string;

    Status s = RunInputsWithStatus(
        {{"input", input_tensor}, {"delimiter", separator_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzStringOpsStringSplit, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()),
                 fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

class FuzzStringOpsStringSplitV2
    : public FuzzSession<std::string, std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto op_node2 =
        tensorflow::ops::Placeholder(scope.WithOpName("separator"), DT_STRING);

    tensorflow::ops::StringSplitV2(scope.WithOpName("output"), op_node,
                                   op_node2);
  }

  void FuzzImpl(const std::string& input_string,
                const std::string& separator_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, {2});

    auto svec = input_tensor.flat<tstring>();
    svec(0) = input_string.c_str();
    svec(1) = input_string.c_str();

    Tensor separator_tensor(tensorflow::DT_STRING, TensorShape({}));
    separator_tensor.scalar<tensorflow::tstring>()() = separator_string;

    Status s = RunInputsWithStatus(
        {{"input", input_tensor}, {"separator", separator_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzStringOpsStringSplitV2, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()),
                 fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

class FuzzStringOpsStringUpper : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);

    tensorflow::ops::StringUpper(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tensorflow::tstring>()() = input_string;

    Status s = RunInputsWithStatus({{"input", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzStringOpsStringUpper, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

}  // end namespace fuzzing
}  // end namespace tensorflow
