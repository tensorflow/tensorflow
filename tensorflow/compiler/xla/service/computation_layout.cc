/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/computation_layout.h"

#include <algorithm>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {

ComputationLayout::ComputationLayout(const ProgramShape& program_shape,
                                     bool ignore_layouts)
    : result_layout_(program_shape.result()) {
  for (auto& shape : program_shape.parameters()) {
    parameter_layouts_.emplace_back(shape);
  }
  if (ignore_layouts) {
    SetToDefaultLayout();
  }
}

void ComputationLayout::SetToDefaultLayout() {
  for (auto& parameter_layout : parameter_layouts_) {
    parameter_layout.SetToDefaultLayout();
  }
  result_layout_.SetToDefaultLayout();
}

bool ComputationLayout::LayoutIsSet() const {
  return std::all_of(parameter_layouts_.begin(), parameter_layouts_.end(),
                     [](const ShapeLayout& s) { return s.LayoutIsSet(); }) &&
         result_layout_.LayoutIsSet();
}

string ComputationLayout::ToString() const {
  std::vector<string> params;
  for (auto& param_layout : parameter_layouts_) {
    params.push_back(param_layout.ToString());
  }
  return tensorflow::strings::StrCat("(",
                                     tensorflow::str_util::Join(params, ", "),
                                     ") => ", result_layout_.ToString());
}

}  // namespace xla
