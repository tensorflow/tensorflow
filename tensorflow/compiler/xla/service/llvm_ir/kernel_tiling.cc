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

#include "tensorflow/compiler/xla/service/llvm_ir/kernel_tiling.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

namespace {
// Returns the indices of the first elements of all consecutive subarrays of the
// given array. For example:
// ConsecutiveSegments({m, m+1, m+2, n, k, k+1}) = {0, 3, 4}
std::vector<size_t> ConsecutiveSegments(tensorflow::gtl::ArraySlice<int64> xs) {
  std::vector<size_t> is = {0};
  for (size_t i = 1; i < xs.size(); ++i) {
    if (1 != xs[i] - xs[i - 1]) {
      is.push_back(i);
    }
  }
  return is;
}

// Merges the sequences of dimensions of the given shape which start at the
// given indices `segs`.
Shape MergeDimensions(tensorflow::gtl::ArraySlice<size_t> segs,
                      const Shape& shape) {
  std::vector<int64> dimensions;
  for (size_t i = 1; i <= segs.size(); ++i) {
    dimensions.push_back(std::accumulate(
        shape.dimensions().begin() + segs[i - 1],
        shape.dimensions().begin() +
            (segs.size() == i ? shape.dimensions().size() : segs[i]),
        1, std::multiplies<int64>()));
  }
  return ShapeUtil::MakeShapeWithDescendingLayout(shape.element_type(),
                                                  dimensions);
}
}  // namespace

tensorflow::gtl::optional<std::vector<int64> > FindTranspose021(
    const Shape& a, const Shape& b) {
  if (!ShapeUtil::CompatibleIgnoringElementType(a, b)) {
    return tensorflow::gtl::nullopt;
  }

  std::vector<int64> perm(a.dimensions().size());
  {
    auto layout_a_orig = LayoutUtil::MinorToMajor(a);
    std::vector<int64> layout_a(layout_a_orig.rbegin(), layout_a_orig.rend());
    auto layout_b_orig = LayoutUtil::MinorToMajor(b);
    std::vector<int64> layout_b(layout_b_orig.rbegin(), layout_b_orig.rend());
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = PositionInContainer(layout_b, layout_a[i]);
    }
  }
  auto segs = ConsecutiveSegments(perm);
  if ((3 == segs.size() && 0 == perm[0]) || 2 == segs.size()) {
    Shape norm_a =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(a);
    Shape reduced_a = MergeDimensions(segs, norm_a);
    auto reduced_a_dims = reduced_a.dimensions();
    std::vector<int64> dims_021;
    if (2 == segs.size()) {
      // The logical component-0 is of size one.
      dims_021 = {1, reduced_a_dims[1], reduced_a_dims[0]};
    } else {
      dims_021 = {reduced_a_dims[0], reduced_a_dims[2], reduced_a_dims[1]};
    }

    return dims_021;
  }

  return tensorflow::gtl::nullopt;
}

IrArray::Index GetUnreducedOutputIndex(
    const IrArray::Index& reduced_output_index,
    const Shape& reduced_output_shape, const Shape& unreduced_output_shape,
    llvm::IRBuilder<>* ir_builder) {
  auto bounds = reduced_output_shape.dimensions();
  auto minor_to_major = reduced_output_shape.layout().minor_to_major();
  llvm::Value* linear_index = reduced_output_index.GetConstantWithIndexType(0);
  int64 multiplier = 1;
  for (int i = 0; i < reduced_output_index.size(); ++i) {
    int64 dim = minor_to_major[i];
    llvm::Value* addend = ir_builder->CreateMul(
        reduced_output_index[dim],
        reduced_output_index.GetConstantWithIndexType(multiplier),
        "linearizing",
        /*HasNUW=*/true, /*HasNSW=*/true);
    linear_index = ir_builder->CreateAdd(linear_index, addend, "",
                                         /*HasNUW=*/true, /*HasNSW=*/true);
    multiplier *= bounds[dim];
  }

  return IrArray::Index(linear_index, unreduced_output_shape, ir_builder);
}

}  // namespace llvm_ir
}  // namespace xla
