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

#include "tensorflow/dtensor/mlir/expansions/reduce_spmd_expander.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

absl::string_view StringRefToView(llvm::StringRef ref) {
  return absl::string_view(ref.data(), ref.size());
}

absl::string_view DefiningOpName(mlir::Value operand) {
  return StringRefToView(operand.getDefiningOp()->getName().getStringRef());
}

Status AssertReplicated(mlir::Value operand) {
  TF_ASSIGN_OR_RETURN(auto layout, ExtractLayoutFromOperand(operand));
  if (!layout) return absl::OkStatus();

  if (!layout->IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Expected layout for ", DefiningOpName(operand),
        " to be fully replicated, but found ", layout->ToString());
  }
  return absl::OkStatus();
}

absl::flat_hash_set<std::string> ReducedMeshDimensions(
    const dtensor::Layout& input, const dtensor::Layout& output) {
  absl::flat_hash_set<std::string> mesh_dims;
  for (const auto& dim : input.sharding_spec_strs()) {
    mesh_dims.insert(dim);
  }
  for (const auto& dim : output.sharding_spec_strs()) {
    mesh_dims.erase(dim);
  }
  return mesh_dims;
}

template <typename OpType>
Status ExtractDims(mlir::Operation* op,
                   llvm::SmallVector<int64_t, 4>* reduced_dims, bool* keep_dims,
                   bool* matched) {
  if (!llvm::isa<OpType>(op)) return absl::OkStatus();
  auto reduce_op = llvm::cast<OpType>(op);
  *keep_dims = reduce_op.getKeepDims();
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(
      reduce_op.getReductionIndices(), reduced_dims));
  TF_RETURN_IF_ERROR(AssertReplicated(reduce_op.getReductionIndices()));
  *matched = true;

  return absl::OkStatus();
}

template <>
Status ExtractDims<mlir::TF::L2LossOp>(
    mlir::Operation* op, llvm::SmallVector<int64_t, 4>* reduced_dims,
    bool* keep_dims, bool* matched) {
  if (!llvm::isa<mlir::TF::L2LossOp>(op)) return absl::OkStatus();
  auto loss_op = llvm::cast<mlir::TF::L2LossOp>(op);
  *reduced_dims = llvm::SmallVector<int64_t, 4>{};
  reduced_dims->resize(ValueRank(loss_op->getOperand(0)));
  for (int i = 0; i < reduced_dims->size(); ++i) {
    (*reduced_dims)[i] = i;
  }
  *keep_dims = false;
  *matched = true;
  return absl::OkStatus();
}

template <>
Status ExtractDims<mlir::TF::BiasAddGradOp>(
    mlir::Operation* op, llvm::SmallVector<int64_t, 4>* reduced_dims,
    bool* keep_dims, bool* matched) {
  if (!llvm::isa<mlir::TF::BiasAddGradOp>(op)) return absl::OkStatus();
  auto bias_add_grad_op = llvm::cast<mlir::TF::BiasAddGradOp>(op);
  auto data_format = bias_add_grad_op.getDataFormat();
  // rank is at least 2 (required by BiasAddGrad).
  int rank = ValueRank(bias_add_grad_op->getOperand(0));
  if (data_format == "NHWC") {
    for (int dim = 0; dim < rank - 1; ++dim) {
      reduced_dims->push_back(dim);
    }
  } else if (data_format == "NCHW") {
    for (int dim = 0; dim < rank; ++dim) {
      if (dim == 1) continue;
      reduced_dims->push_back(dim);
    }
  } else {
    return errors::InvalidArgument("Unsupported data_format for BiasAddGrad: ",
                                   StringRefToView(data_format));
  }
  *keep_dims = false;
  *matched = true;
  return absl::OkStatus();
}

template <>
Status ExtractDims<mlir::TF::EncodePngOp>(
    mlir::Operation* op, llvm::SmallVector<int64_t, 4>* reduced_dims,
    bool* keep_dims, bool* matched) {
  if (!llvm::isa<mlir::TF::EncodePngOp>(op)) return absl::OkStatus();
  *reduced_dims = {-3, -2, -1};
  *keep_dims = false;
  *matched = true;
  return absl::OkStatus();
}

Status ExtractReductionParameters(mlir::Operation* op,
                                  absl::flat_hash_set<int>& reduced_dims_set,
                                  bool& keep_dims) {
  llvm::SmallVector<int64_t, 4> reduced_dims;
  bool matched = false;
  TF_RETURN_IF_ERROR(ExtractDims<mlir::TF::EncodePngOp>(op, &reduced_dims,
                                                        &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::SumOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::AllOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::AnyOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::MaxOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::MinOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::MeanOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::ProdOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(
      ExtractDims<mlir::TF::L2LossOp>(op, &reduced_dims, &keep_dims, &matched));
  TF_RETURN_IF_ERROR(ExtractDims<mlir::TF::BiasAddGradOp>(
      op, &reduced_dims, &keep_dims, &matched));

  if (!matched)
    return errors::Unimplemented("Op type: ", OpName(op),
                                 " not yet implemented.");

  reduced_dims_set.insert(reduced_dims.begin(), reduced_dims.end());
  return absl::OkStatus();
}

StatusOr<Layout> ComputeResultLayout(mlir::Operation* op,
                                     const Layout& input_layout) {
  absl::flat_hash_set<int> reduced_dims_set;
  bool keep_dims;
  TF_RETURN_IF_ERROR(
      ExtractReductionParameters(op, reduced_dims_set, keep_dims));

  return input_layout.GetLayoutWithReducedDims(reduced_dims_set, keep_dims);
}

}  // namespace

StatusOr<mlir::Operation*> ReduceSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto requested_output_layout,
                      ExtractSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));

  if (!input_layout || !requested_output_layout)
    return errors::InvalidArgument("is missing input or output layouts.");

  // Generate an error message for TPU int64.
  if (input_layout->mesh().is_tpu_mesh()) {
    if (auto tensor_type =
            mlir::dyn_cast<mlir::TensorType>(op->getOperand(0).getType())) {
      if (tensor_type.getElementType().isInteger(64)) {
        return errors::InvalidArgument(
            "ReduceOp on TPU does not support int64 as dtype.");
      }
    }
  }

  mlir::OpBuilder builder(op->getBlock(), ++mlir::Block::iterator(op));

  TF_ASSIGN_OR_RETURN(auto output_layout,
                      ComputeResultLayout(op, input_layout.value()));

  absl::flat_hash_set<std::string> reduced_dims =
      ReducedMeshDimensions(*input_layout, output_layout);
  InferSPMDExpandedLocalShape(op);

  mlir::Operation* reduce_op;
  if (mlir::isa<mlir::TF::SumOp, mlir::TF::L2LossOp, mlir::TF::BiasAddGradOp,
                mlir::TF::EncodePngOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpAdd));
  } else if (mlir::isa<mlir::TF::AllOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpAll));
  } else if (mlir::isa<mlir::TF::AnyOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpAny));
  } else if (mlir::isa<mlir::TF::MaxOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpMax));
  } else if (mlir::isa<mlir::TF::MinOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpMin));
  } else if (mlir::isa<mlir::TF::ProdOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpMul));
  } else if (mlir::isa<mlir::TF::MeanOp>(op)) {
    TF_ASSIGN_OR_RETURN(
        reduce_op,
        EmitAllReduce(builder, output_layout, reduced_dims, op, kReduceOpMean));
  } else {
    return DT_CTX(errors::Unimplemented(
        "Failed to create AllReduce op during SPMD expansion. Op type: ",
        OpName(op), " not yet implemented in DTensor SPMD pass."));
  }

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  TF_ASSIGN_OR_RETURN(
      auto final_output,
      EmitAllScatter(builder, reduce_op->getOpResult(0), output_layout,
                     requested_output_layout.value(), &newly_created_ops));
  reduce_op->getOpResult(0).replaceAllUsesExcept(final_output,
                                                 newly_created_ops);
  return final_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> ReduceSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // Do not infer any output layout if no input layout exists.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(auto result_layout,
                      ComputeResultLayout(op, input_layouts.lookup(0)));
  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

// For Reduction op, we do not propagate consumer preferred layouts to operand
// layouts as reduction operation explicitly converts tensor dimension of
// reduced dimensions to replicated.
StatusOr<llvm::DenseMap<int, Layout>> ReduceSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // Do not infer any operand layouts if no output layouts exists.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  Layout output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto input_shape, GetShapeOfValue(op->getOperand(0)));

  std::vector<std::string> inferred_operand_layout_str;

  absl::flat_hash_set<int> reduced_dims_set;
  bool keep_dims;
  TF_RETURN_IF_ERROR(
      ExtractReductionParameters(op, reduced_dims_set, keep_dims));

  // For each dimension, if dimension is not reduced dimension, then propagate
  // the sharding of output value to operand. Else, set input dimension as
  // replicated.
  int output_dim = 0;
  for (int i = 0; i < input_shape.size(); ++i) {
    // reduced_dims may contain negative values.
    if (reduced_dims_set.contains(i) ||
        reduced_dims_set.contains(i - input_shape.size())) {
      inferred_operand_layout_str.push_back(Layout::kAny);
      if (keep_dims) output_dim += 1;
    } else {
      inferred_operand_layout_str.push_back(
          output_layout.sharding_spec(output_dim));
      output_dim += 1;
    }
  }
  TF_ASSIGN_OR_RETURN(
      auto inferred_operand_layout,
      Layout::GetLayout(inferred_operand_layout_str, output_layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, inferred_operand_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
