/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/computation_layout.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "xla/layout.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"

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
  return absl::c_all_of(parameter_layouts_,
                        [](const ShapeLayout& s) { return s.LayoutIsSet(); }) &&
         result_layout_.LayoutIsSet();
}

bool ComputationLayout::AnyLayoutSet() const {
  return absl::c_any_of(
             parameter_layouts_,
             [](const ShapeLayout& s) { return s.AnyLayoutIsSet(); }) ||
         result_layout_.AnyLayoutIsSet();
}

absl::StatusOr<std::vector<Layout>>
ComputationLayout::FlattenedParameterLayouts() const {
  std::vector<Layout> result;
  for (int i = 0; i < parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        parameter_shape(i),
        [this, &result](const Shape& subshape,
                        const ShapeIndex& index) -> absl::Status {
          if (subshape.IsTuple()) {
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return Unimplemented(
                "ComputationLayout::FlattenedParameterLayouts doesn't support "
                "token or opaque parameters (got: %s)",
                ToString());
          }
          if (!subshape.has_layout()) {
            return InvalidArgument(
                "ComputationLayout::FlattenedParameterLayouts can only be "
                "called after all parameters have layouts assigned (got: %s)",
                ToString());
          }
          result.push_back(subshape.layout());
          return absl::OkStatus();
        }));
  }
  return result;
}

absl::StatusOr<std::vector<Layout>> ComputationLayout::FlattenedResultLayouts()
    const {
  std::vector<Layout> result;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      result_shape(),
      [this, &result](const Shape& subshape,
                      const ShapeIndex& index) -> absl::Status {
        if (subshape.IsTuple()) {
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return Unimplemented(
              "ComputationLayout::FlattenedResultLayouts doesn't support "
              "token or opaque outputs (got: %s)",
              ToString());
        }
        if (!subshape.has_layout()) {
          return InvalidArgument(
              "ComputationLayout::FlattenedResultLayouts can only be called "
              "after all outputs have layouts assigned (got: %s)",
              ToString());
        }
        result.push_back(subshape.layout());
        return absl::OkStatus();
      }));
  return result;
}

void ComputationLayout::Print(Printer* printer) const {
  printer->Append("(");
  if (!parameter_layouts_.empty()) {
    parameter_layouts_[0].Print(printer);
    for (int i = 1; i < parameter_layouts_.size(); ++i) {
      if (i % 5 == 0) {
        printer->Append(absl::StrFormat(", /*index=%lld*/", i));
      } else {
        printer->Append(", ");
      }
      parameter_layouts_[i].Print(printer);
    }
  }
  printer->Append(")->");
  result_layout_.Print(printer);
}

std::string ComputationLayout::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

ProgramShape ComputationLayout::ComputeProgramShape() const {
  ProgramShape program_shape;
  for (int64_t i = 0; i < parameter_layouts_.size(); ++i) {
    program_shape.AddParameter(parameter_layouts_[i].shape(),
                               absl::StrCat("p", i));
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

}  // namespace xla
