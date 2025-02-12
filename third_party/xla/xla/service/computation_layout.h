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

#ifndef XLA_SERVICE_COMPUTATION_LAYOUT_H_
#define XLA_SERVICE_COMPUTATION_LAYOUT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/printer.h"
#include "xla/shape_layout.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Class which contains the layouts of the parameters and results of a
// computation. The layouts are stored as ShapeLayouts with immutable shapes and
// mutable layouts.
class ComputationLayout {
 public:
  // Creates a new ComputationLayout with the given result layout.
  explicit ComputationLayout(ShapeLayout result_layout)
      : result_layout_(std::move(result_layout)) {}

  // Constructs a ComputationLayout from a ProgramShape. The layouts of the
  // parameters and results are set to the default layout. Layouts in the
  // ProgramShape are ignored if ignore_layouts is true.
  explicit ComputationLayout(const ProgramShape& program_shape,
                             bool ignore_layouts = true);

  // Adds a new parameter layout to the computation layout.
  void add_parameter_layout(ShapeLayout shape_layout) {
    parameter_layouts_.push_back(std::move(shape_layout));
  }

  // Returns the layout of a particular parameter.
  const ShapeLayout& parameter_layout(int64_t param_no) const {
    return parameter_layouts_[param_no];
  }
  ShapeLayout* mutable_parameter_layout(int64_t param_no) {
    return &parameter_layouts_[param_no];
  }

  // Returns the number of parameters in the computation.
  int parameter_count() const { return parameter_layouts_.size(); }

  // Returns the ShapeLayouts of the parameters of the computation.
  const std::vector<ShapeLayout>& parameter_layouts() const {
    return parameter_layouts_;
  }

  // Returns the ShapeLayout of a result of the computation.
  const ShapeLayout& result_layout() const { return result_layout_; }
  ShapeLayout* mutable_result_layout() { return &result_layout_; }

  // Returns the shape of the particular parameter or result of the computation
  // with layout.
  const Shape& parameter_shape(int64_t param_no) const {
    return parameter_layouts_[param_no].shape();
  }
  const Shape& result_shape() const { return result_layout_.shape(); }

  // Sets layouts of all parameters and the result to the default layout.
  void SetToDefaultLayout();

  // Returns true if all layouts (parameters and result) have been set.
  bool LayoutIsSet() const;
  // Returns true if any layouts (parameters and result) have been set.
  bool AnyLayoutSet() const;

  // Returns a list of each parameter's layout. If the parameters are tupled,
  // returns an untupled list. Must only be called if all parameters have
  // layouts set (check with LayoutIsSet()).
  absl::StatusOr<std::vector<Layout>> FlattenedParameterLayouts() const;

  // Returns a list of each output's layout. If the result shape is a tuple,
  // returns an untupled list. Must only be called if all outputs have layouts
  // set (check with LayoutIsSet()).
  absl::StatusOr<std::vector<Layout>> FlattenedResultLayouts() const;

  // Prints a string representation of this object.
  void Print(Printer* printer) const;

  // Returns a string representation of this object.
  std::string ToString() const;

  // Create a ProgramShape proto based on the parameter and result shapes held
  // within this object.
  ProgramShape ComputeProgramShape() const;

  bool operator==(const ComputationLayout& other) const;
  bool operator!=(const ComputationLayout& other) const;

  template <typename H>
  friend H AbslHashValue(H h, const ComputationLayout& computation_layout) {
    h = H::combine(std::move(h), computation_layout.result_layout_.shape());
    for (const auto& parameter_layout : computation_layout.parameter_layouts_) {
      h = H::combine(std::move(h), parameter_layout.shape());
    }
    h = H::combine(std::move(h), computation_layout.parameter_layouts_.size());
    return h;
  }

 private:
  std::vector<ShapeLayout> parameter_layouts_;
  ShapeLayout result_layout_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPUTATION_LAYOUT_H_
