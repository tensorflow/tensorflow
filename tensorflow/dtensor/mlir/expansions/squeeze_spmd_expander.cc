/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/squeeze_spmd_expander.h"

#include <utility>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

std::set<int64_t> GetSqueezeDims(mlir::Operation* op, int64_t rank) {
  auto array_attribute = op->getAttrOfType<mlir::ArrayAttr>("squeeze_dims");
  std::set<int64_t> squeeze_dims;
  if (array_attribute) {
    auto attr_list = array_attribute.getValue().vec();
    for (const auto& attr : attr_list) {
      int64_t dim = attr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
      // Offset the negative indices to positive range.
      squeeze_dims.insert((dim + rank) % rank);
    }
  }
  return squeeze_dims;
}

}  // namespace

StatusOr<llvm::DenseMap<int, Layout>> SqueezeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If there is no tensor layout then do not infer any output layouts.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto shape, ExtractGlobalInputShape(op->getOpOperand(0)));
  std::set<int64_t> squeeze_dims = GetSqueezeDims(op, /*rank=*/shape.size());

  std::vector<ShardingSpec> layout_specs;
  layout_specs.reserve(input_layout.rank());
  for (int64 i = 0; i < input_layout.rank(); ++i) {
    if (squeeze_dims.empty()) {
      if (shape[i] > 1) {
        layout_specs.push_back(input_layout.dim(i));
      }
    } else {
      if (squeeze_dims.find(i) == squeeze_dims.end()) {
        layout_specs.push_back(input_layout.dim(i));
      }
    }
  }

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      Layout::GetLayout(layout_specs, input_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
SqueezeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If there is no output layout present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto shape, ExtractGlobalInputShape(op->getOpOperand(0)));
  std::set<int64_t> squeeze_dims = GetSqueezeDims(op, /*rank=*/shape.size());

  ShardingSpec unsharded_spec;
  unsharded_spec.set_sharding_spec(Layout::kUnshardedDim);

  std::vector<ShardingSpec> layout_specs;
  layout_specs.reserve(output_layout.rank());
  size_t j = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (squeeze_dims.empty()) {
      if (shape[i] > 1) {
        layout_specs.push_back(output_layout.dim(j++));
      } else {
        layout_specs.push_back(unsharded_spec);
      }
    } else {
      if (squeeze_dims.find(i) == squeeze_dims.end()) {
        layout_specs.push_back(output_layout.dim(j++));
      } else {
        layout_specs.push_back(unsharded_spec);
      }
    }
  }

  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      Layout::GetLayout(layout_specs, output_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, input_layout}});
}

StatusOr<mlir::Operation*> SqueezeSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto squeeze_op = mlir::cast<mlir::TF::SqueezeOp>(op);
  TF_ASSIGN_OR_RETURN(auto layout, ExtractSingleLayoutFromOp(op));

  if (!layout) {
    return errors::InvalidArgument(
        "layout of SqueezeOp must be known before SPMD expansion.");
  }

  TF_ASSIGN_OR_RETURN(auto input_shape,
                      ExtractGlobalInputShape(op->getOpOperand(0)));
  std::set<int64_t> squeeze_dims =
      GetSqueezeDims(op, /*rank=*/input_shape.size());

  if (squeeze_dims.empty()) {
    // If the squeeze dim is empty, make sure the squeeze only happens on the
    // dims that is not sharded and has global_shape is 1. Otherwise if the
    // local shape happens to have size 1 on the dim, it got squeezed
    // unexpected.
    for (int i = 0; i < input_shape.size(); ++i) {
      // Global shape 1 implies the dim cannot be sharded -- worst case it can
      // be sharded over a mesh with dim size 1, and we would squeeze it as is.
      if (input_shape[i] == 1) {
        squeeze_dims.insert(i);
      }
    }
    mlir::OpBuilder builder(squeeze_op);
    squeeze_op->setAttr("squeeze_dims",
                        builder.getI64ArrayAttr(llvm::SmallVector<int64_t>(
                            squeeze_dims.begin(), squeeze_dims.end())));
  }

  return InferSPMDExpandedLocalShape(squeeze_op);
}

}  // namespace dtensor
}  // namespace tensorflow
