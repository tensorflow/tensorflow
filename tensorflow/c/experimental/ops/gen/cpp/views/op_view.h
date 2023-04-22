/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_OP_VIEW_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_OP_VIEW_H_

#include <vector>

#include "tensorflow/c/experimental/ops/gen/cpp/views/arg_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/attr_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_argument_view.h"
#include "tensorflow/c/experimental/ops/gen/model/op_spec.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

class OpView {
 public:
  explicit OpView(OpSpec op);

  const std::vector<ArgView> &Inputs() const;
  const std::vector<ArgView> &Outputs() const;
  const std::vector<AttrView> &Attributes() const;
  const std::vector<OpArgumentView> &AllArguments() const;

  int NumInputs() const;
  int NumOutputs() const;
  ArgView OnlyInput() const;
  ArgView OnlyOutput() const;

  string FunctionName() const;
  string VariableName() const;
  string OpNameString() const;
  string Summary() const;
  std::vector<string> Description() const;
  bool IsListOp() const;

 private:
  OpSpec op_;
  std::vector<ArgView> input_args_;
  std::vector<ArgView> output_args_;
  std::vector<AttrView> argument_attrs_;
  std::vector<OpArgumentView> all_arguments_;
};

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_VIEWS_OP_VIEW_H_
