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
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"

#include "tensorflow/c/experimental/ops/gen/common/view_util.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/arg_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/attr_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_argument_view.h"
#include "tensorflow/c/experimental/ops/gen/model/op_spec.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace generator {
namespace cpp {

OpView::OpView(OpSpec op)
    : op_(op),
      input_args_(op_.Inputs().begin(), op_.Inputs().end()),
      output_args_(op_.Outputs().begin(), op_.Outputs().end()),
      argument_attrs_(op_.Attributes().begin(), op_.Attributes().end()) {
  // Initialize function arguments
  all_arguments_.push_back(OpArgumentView("AbstractContext*", "ctx"));
  for (const auto& arg : op_.Inputs()) {
    all_arguments_.push_back(OpArgumentView(arg));
  }
  for (const auto& arg : op_.Outputs()) {
    all_arguments_.push_back(OpArgumentView(arg));
  }
  for (const auto& attr : op.Attributes()) {
    all_arguments_.push_back(OpArgumentView(attr));
  }
  all_arguments_.push_back(OpArgumentView("const char*", "name", "nullptr"));
  all_arguments_.push_back(
      OpArgumentView("const char*", "raw_device_name", "nullptr"));
}

const std::vector<ArgView>& OpView::Inputs() const { return input_args_; }

const std::vector<ArgView>& OpView::Outputs() const { return output_args_; }

const std::vector<AttrView>& OpView::Attributes() const {
  return argument_attrs_;
}

const std::vector<OpArgumentView>& OpView::AllArguments() const {
  return all_arguments_;
}

int OpView::NumInputs() const { return input_args_.size(); }

int OpView::NumOutputs() const { return output_args_.size(); }

ArgView OpView::OnlyInput() const {
  CHECK_EQ(input_args_.size(), 1);  // Crash OK
  return input_args_.front();
}

ArgView OpView::OnlyOutput() const {
  CHECK_EQ(output_args_.size(), 1);  // Crash OK
  return output_args_.front();
}

string OpView::FunctionName() const { return op_.name(); }

string OpView::OpNameString() const { return Quoted(op_.name()); }

string OpView::VariableName() const { return "op_ptr"; }

std::vector<string> OpView::Description() const {
  return str_util::Split(op_.description(), "\n");
}

string OpView::Summary() const { return op_.summary(); }

// Context
bool OpView::IsListOp() const {
  return NumOutputs() == 1 && OnlyOutput().IsList();
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
