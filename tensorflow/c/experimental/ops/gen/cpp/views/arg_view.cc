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
#include "tensorflow/c/experimental/ops/gen/cpp/views/arg_view.h"

namespace tensorflow {
namespace generator {
namespace cpp {

ArgView::ArgView(ArgSpec arg) : arg_(arg) {}

string ArgView::VariableName() const { return arg_.name(); }

string ArgView::SetterMethod() const {
  if (IsList()) {
    return "AddInputList";
  } else {
    return "AddInput";
  }
}

std::vector<string> ArgView::SetterArgs() const { return {VariableName()}; }

bool ArgView::IsList() const { return arg_.arg_type().is_list(); }

int ArgView::Position() const { return arg_.position(); }

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
