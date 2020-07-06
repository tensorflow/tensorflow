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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {

ComputationLayout::ComputationLayout(const ProgramShape& program_shape,
                                     bool ignore_layouts)
    : result_layout_(program_shape.result()) {
  for (auto& shape : program_shape.parameters()) {
    parameter_layouts_.emplace_back(shape);
  }
  if (ignore_layouts) {
    SetToDefaultLayout();
  } else {
    SetToDefaultLayoutIfEmpty();
  }
}

void ComputationLayout::SetToDefaultLayout() {
  for (auto& parameter_layout : parameter_layouts_) {
    parameter_layout.SetToDefaultLayout();
  }
  result_layout_.SetToDefaultLayout();
}

void ComputationLayout::SetToDefaultLayoutIfEmpty() {
  for (auto& parameter_layout : parameter_layouts_) {
    if (!parameter_layout.LayoutIsSet()) {
      parameter_layout.SetToDefaultLayout();
    }
  }
  if (!result_layout_.LayoutIsSet()) {
    result_layout_.SetToDefaultLayout();
  }
}

bool ComputationLayout::LayoutIsSet() const {
  return absl::c_all_of(parameter_layouts_,
                        [](const ShapeLayout& s) { return s.LayoutIsSet(); }) &&
         result_layout_.LayoutIsSet();
}

string ComputationLayout::ToString() const {
  std::vector<string> params;
  for (auto& param_layout : parameter_layouts_) {
    params.push_back(param_layout.ToString());
  }
  return absl::StrCat("(", absl::StrJoin(params, ", "), ") => ",
                      result_layout_.ToString());
}

ProgramShape ComputationLayout::ComputeProgramShape() const {
  ProgramShape program_shape;
  for (int64 i = 0; i < parameter_layouts_.size(); ++i) {
    *program_shape.add_parameters() = parameter_layouts_[i].shape();
    *program_shape.add_parameter_names() = absl::StrCat("p", i);
  }
  *program_shape.mutable_result() = result_layout_.shape();
  return program_shape;
}

bool ComputationLayout::operator==(const ComputationLayout& other) const {
  return result_layout() == other.result_layout() &&
         parameter_layouts() == other.parameter_layouts();
}

bool ComputationLayout::operator!=(const ComputationLayout& other) const {
  return result_layout() != other.result_layout() ||
         parameter_layouts() != other.parameter_layouts();
}

uint64 ComputationLayout::Hash() const {
  uint64 hash_value = ShapeUtil::Hash(result_layout_.shape());
  for (const auto& parameter_layout : parameter_layouts_) {
    hash_value = tensorflow::Hash64Combine(
        hash_value, ShapeUtil::Hash(parameter_layout.shape()));
  }
  return hash_value;
}

}  // namespace xla
