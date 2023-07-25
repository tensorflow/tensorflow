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

#include "tensorflow/dtensor/mlir/expansions/fft_spmd_expander.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/expansions/meta_spmd_expander.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Return the last unsharded axis. Return -1 for fully replicated dtensor
int LastUnshardedAxis(const std::vector<std::string>& sharding_specs) {
  for (int i = sharding_specs.size() - 1; i >= 0; --i)
    if (sharding_specs[i] == Layout::kUnshardedDim) return i;
  return -1;
}

// Return false for non-distributed *FFTN.
bool IsDistributedFFTN(int num_transform_axes, const Layout& layout) {
  std::vector<std::string> sharding_specs = layout.sharding_spec_strs();
  for (int i = sharding_specs.size() - num_transform_axes;
       i < sharding_specs.size(); ++i)
    if (sharding_specs[i] != Layout::kUnshardedDim) {
      return true;
    }
  return false;
}

bool IsComplexFFT(mlir::Value input) {
  auto data_type =
      mlir::dyn_cast<mlir::TensorType>(input.getType()).getElementType();
  return data_type.isa<mlir::ComplexType>();
}

Status IsProperFFTLength(mlir::Operation* op,
                         const llvm::SmallVector<int64_t, 4>& fft_length_vec) {
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractRequiredLayoutFromOperand(op->getOperand(0)));
  const Mesh& mesh = input_layout.mesh();
  int axes = fft_length_vec.size();
  // RFFT in DTensor requires axes except -1 to have the same shape as input.
  llvm::ArrayRef<int64_t> input_shape =
      mlir::dyn_cast<mlir::TensorType>(op->getOperand(0).getType()).getShape();
  std::vector<int64_t> input_shape_vec = input_shape.vec();
  for (int i = 0; i < axes - 1; ++i)
    if (fft_length_vec[i] != input_shape_vec[input_shape_vec.size() - axes + i])
      return absl::InvalidArgumentError(
          "DTensor RFFTOps are not suitable for current 'fft_length'.");

  // fft_length[-1] should be divisible by the corresponding device number.
  int num_of_devices_last_dim = mesh.dim_sizes()[input_shape_vec.size() - 1];
  if (axes > 1 && fft_length_vec[axes - 1] % num_of_devices_last_dim == 1)
    return absl::InvalidArgumentError(
        "The values with current 'fft_length' are not shardable.");
  return absl::OkStatus();
}

StatusOr<llvm::SmallVector<int64_t, 4>> ExtractFFTLengthFromOp(
    mlir::Operation* op) {
  mlir::Value fft_length = op->getOperand(1);
  llvm::SmallVector<int64_t, 4> fft_length_vec;
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(fft_length, &fft_length_vec));
  TF_RETURN_IF_ERROR(IsProperFFTLength(op, fft_length_vec));
  return fft_length_vec;
}

// Forward flow for FFT and backward flow for iFFT
// sharding_specs has at least one element
void PropagateFFTLayout(std::vector<std::string>& sharding_specs, int axes) {
  int last_unsharded_axis = LastUnshardedAxis(sharding_specs);
  if (last_unsharded_axis == -1)
    sharding_specs[sharding_specs.size() - 1] = Layout::kUnshardedDim;
  else if (last_unsharded_axis != sharding_specs.size() - 1)
    std::iter_swap(sharding_specs.end() - 1,
                   sharding_specs.begin() + last_unsharded_axis);
  std::string last_sharding_spec = sharding_specs.back();
  sharding_specs.pop_back();
  sharding_specs.insert(sharding_specs.end() - axes + 1, last_sharding_spec);
}

// Backward flow for FFT and forward flow for iFFT
// sharding_specs has at least one element
void PropagateIFFTLayout(std::vector<std::string>& sharding_specs, int axes) {
  int last_unsharded_axis = LastUnshardedAxis(sharding_specs);
  if (last_unsharded_axis == -1)
    sharding_specs[sharding_specs.size() - axes] = Layout::kUnshardedDim;
  else if (last_unsharded_axis != sharding_specs.size() - axes)
    std::iter_swap(sharding_specs.end() - axes,
                   sharding_specs.begin() + last_unsharded_axis);
  std::string unsharded_axis = sharding_specs[sharding_specs.size() - axes];
  sharding_specs.erase(sharding_specs.end() - axes);
  sharding_specs.push_back(unsharded_axis);
}

StatusOr<mlir::Value> EmitTransposeRelayout(mlir::OpBuilder& builder,
                                            mlir::Location location,
                                            mlir::Value input,
                                            const Layout& init_layout,
                                            const Mesh& mesh,
                                            std::pair<int, int>& perm_axes) {
  std::vector<int64_t> perm_for_transpose;
  int input_rank = ValueRank(input);
  perm_for_transpose.reserve(input_rank);
  for (int ax = 0; ax < input_rank; ++ax) {
    perm_for_transpose.push_back(ax);
  }
  std::iter_swap(perm_for_transpose.begin() + perm_axes.first,
                 perm_for_transpose.begin() + perm_axes.second);
  mlir::Operation* transpose_op =
      EmitTransposeOp(builder, location, input, perm_for_transpose);
  mlir::Value transposed_input = transpose_op->getResult(0);

  std::vector<std::string> sharding_specs = init_layout.sharding_spec_strs();
  std::iter_swap(sharding_specs.begin() + perm_axes.first,
                 sharding_specs.begin() + perm_axes.second);
  TF_ASSIGN_OR_RETURN(
      auto transposed_input_layout,
      Layout::GetLayout(init_layout.type(), sharding_specs, mesh));
  TF_ASSIGN_OR_RETURN(
      transposed_input,
      EmitRelayout(transposed_input, transposed_input_layout, init_layout));
  return transposed_input;
}

Status NormalizeAxes(std::vector<int>& transform_axes, int input_rank) {
  std::sort(transform_axes.begin(), transform_axes.end());
  for (int i = 0; i < transform_axes.size(); ++i) {
    if (transform_axes[i] >= input_rank) {
      return absl::InvalidArgumentError("Axes to perform FFTN on are invalid.");
    } else if (transform_axes[i] < 0) {
      transform_axes[i] += input_rank;
    }
  }
  if (transform_axes.empty()) {
    transform_axes.reserve(input_rank);
    for (int i = 0; i < input_rank; ++i) transform_axes.push_back(i);
  }
  return absl::OkStatus();
}

// Make the last axis of the layout unsharded by swapping with another
// axis or forcing the last axis to be unsharded if fully sharded.
StatusOr<Layout> UnshardLastAxis(int input_rank, const Layout& layout,
                                 const Mesh& mesh) {
  if (input_rank < 1)
    return absl::InvalidArgumentError("input_rank must be >= 1");
  std::vector<std::string> input_sharding_specs = layout.sharding_spec_strs();
  int last_unsharded_axis = LastUnshardedAxis(input_sharding_specs);
  if (last_unsharded_axis == -1)
    input_sharding_specs[input_rank - 1] = Layout::kUnshardedDim;
  else if (last_unsharded_axis != input_rank - 1)
    std::iter_swap(input_sharding_specs.end() - 1,
                   input_sharding_specs.begin() + last_unsharded_axis);
  return Layout::GetLayout(layout.type(), input_sharding_specs, mesh);
}

// Lowering FFTN/RFFTN operation for the first N-1 axes to N-1 1-d FFTOps.
StatusOr<mlir::Operation*> ExpandFFTNImpl(
    mlir::Operation* xfft_op, mlir::Operation* fft_op,
    std::vector<int>& transform_axes, const int input_rank,
    mlir::OpBuilder& builder, mlir::Location location,
    const Layout& input_layout, const Mesh& mesh) {
  SetSingleLayoutOnOp(xfft_op, input_layout);
  mlir::Value output = xfft_op->getResult(0);

  if (transform_axes.empty()) {
    fft_op->getResult(0).replaceAllUsesWith(xfft_op->getResult(0));
    fft_op->erase();
    return InferSPMDExpandedLocalShape(xfft_op);
  }

  std::vector<int64_t> perm;
  perm.reserve(input_rank);
  for (int i = 0; i < input_rank - 1; ++i) {
    perm.push_back(i);
  }
  perm.insert(perm.end() - transform_axes.size(), input_rank - 1);

  mlir::TF::FFTOp new_fft_op;
  Layout intermediate_layout = input_layout;
  int ax;
  while (!transform_axes.empty()) {
    ax = transform_axes.back();
    transform_axes.pop_back();

    std::pair<int, int> perm_axes = {ax, input_rank - 1};
    TF_ASSIGN_OR_RETURN(
        mlir::Value transposed_input,
        EmitTransposeRelayout(builder, location, output, intermediate_layout,
                              mesh, perm_axes));

    new_fft_op = builder.create<mlir::TF::FFTOp>(
        location, transposed_input.getType(), transposed_input);
    SetSingleLayoutOnOp(new_fft_op, intermediate_layout);
    output = new_fft_op.getOutput();
  }

  mlir::Operation* transpose_op =
      EmitTransposeOp(builder, location, output, perm);
  mlir::Value transposed_output = transpose_op->getResult(0);

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(new_fft_op);
  TF_ASSIGN_OR_RETURN(auto final_output,
                      EmitRelayout(transposed_output, intermediate_layout,
                                   intermediate_layout, &newly_created_ops));
  fft_op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
  fft_op->erase();
  return InferSPMDExpandedLocalShape(final_output.getDefiningOp());
}

StatusOr<mlir::Operation*> ExpandFFTN(mlir::Operation* fft_op,
                                      std::vector<int>& transform_axes) {
  mlir::OpBuilder builder(fft_op);
  mlir::Value input = fft_op->getOperand(0);
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractRequiredLayoutFromOperand(input));
  const int input_rank = ValueRank(input);
  const Mesh& mesh = input_layout.mesh();
  mlir::Location location = fft_op->getLoc();

  TF_RETURN_IF_ERROR(NormalizeAxes(transform_axes, input_rank));
  int num_transform_axes = transform_axes.size();

  if (!IsDistributedFFTN(num_transform_axes, input_layout))
    return InferSPMDExpandedLocalShape(fft_op);

  // FIXME(b/292286720): Since the last axis must be one of the transform_axes
  // in current 1/2/3d transform ops, we don't need to find the last
  // transofrm_axes and can just use -1. Need to be fixed by adding transpose op
  // prior to this or unshard the last transform_axes.
  TF_ASSIGN_OR_RETURN(Layout intermediate_layout,
                      UnshardLastAxis(input_rank, input_layout, mesh));
  TF_ASSIGN_OR_RETURN(mlir::Value intermediate,
                      EmitRelayout(input, input_layout, intermediate_layout));

  if (IsComplexFFT(input)) {
    // FFT for the last axis.
    mlir::TF::FFTOp fft_output_op = builder.create<mlir::TF::FFTOp>(
        location, intermediate.getType(), intermediate);
    transform_axes.pop_back();
    return ExpandFFTNImpl(fft_output_op, fft_op, transform_axes, input_rank,
                          builder, location, intermediate_layout, mesh);
  } else {
    TF_ASSIGN_OR_RETURN(auto fft_length_vec, ExtractFFTLengthFromOp(fft_op));
    mlir::Value fft_length = IntConst(
        builder, location, (int32)fft_length_vec[num_transform_axes - 1]);
    llvm::ArrayRef<int64_t> rfft_shape =
        mlir::dyn_cast<mlir::TensorType>(intermediate.getType()).getShape();
    std::vector<int64_t> rfft_shape_vec = rfft_shape.vec();
    int num_of_devices_last_dim = mesh.dim_sizes()[input_rank - 1];
    rfft_shape_vec[input_rank - 1] =
        fft_length_vec[num_transform_axes - 1] / 2 + 1;
    if (fft_length_vec.size() > 1 &&
        rfft_shape_vec[input_rank - 1] % num_of_devices_last_dim != 0)
      return absl::InvalidArgumentError(
          "No suitable algorithm in DTensor found for current 'fft_length'.");

    mlir::Type output_type = mlir::RankedTensorType::get(
        rfft_shape_vec,
        mlir::dyn_cast<mlir::TensorType>(fft_op->getResult(0).getType())
            .getElementType());
    // Real FFT for the last axis.
    mlir::TF::RFFTOp rfft_output_op = builder.create<mlir::TF::RFFTOp>(
        location, output_type, intermediate, fft_length);
    transform_axes.pop_back();
    return ExpandFFTNImpl(rfft_output_op, fft_op, transform_axes, input_rank,
                          builder, location, intermediate_layout, mesh);
  }
}

StatusOr<mlir::Operation*> ExpandIFFTN(mlir::Operation* ifft_op,
                                       std::vector<int>& transform_axes) {
  mlir::OpBuilder builder(ifft_op);
  mlir::Value input = ifft_op->getOperand(0);
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractRequiredLayoutFromOperand(input));
  const auto input_rank = ValueRank(input);
  const Mesh& mesh = input_layout.mesh();
  mlir::Location location = ifft_op->getLoc();
  std::vector<std::string> input_sharding_specs =
      input_layout.sharding_spec_strs();

  TF_RETURN_IF_ERROR(NormalizeAxes(transform_axes, input_rank));
  int num_transform_axes = transform_axes.size();

  if (!IsDistributedFFTN(num_transform_axes, input_layout))
    return InferSPMDExpandedLocalShape(ifft_op);

  input_sharding_specs.push_back(
      input_sharding_specs[input_rank - num_transform_axes]);
  input_sharding_specs.erase(input_sharding_specs.begin() + input_rank -
                             num_transform_axes);
  TF_ASSIGN_OR_RETURN(
      input_layout,
      Layout::GetLayout(input_layout.type(), input_sharding_specs, mesh));
  TF_ASSIGN_OR_RETURN(Layout intermediate_layout,
                      UnshardLastAxis(input_rank, input_layout, mesh));

  std::vector<int64_t> perm;
  perm.reserve(input_rank);
  for (int i = 0; i < input_rank; ++i)
    if (i != input_rank - num_transform_axes) {
      perm.push_back(i);
    }
  perm.push_back(input_rank - num_transform_axes);
  mlir::Operation* transpose_op =
      EmitTransposeOp(builder, location, input, perm);
  mlir::Value transposed_output = transpose_op->getResult(0);
  TF_ASSIGN_OR_RETURN(
      transposed_output,
      EmitRelayout(transposed_output, input_layout, intermediate_layout));

  mlir::TF::IFFTOp fft_new_op;
  int ax;  // current axis
  while (transform_axes.size() > 1) {
    ax = transform_axes[1] - 1;
    transform_axes.erase(transform_axes.begin() + 1);
    fft_new_op = builder.create<mlir::TF::IFFTOp>(
        location, transposed_output.getType(), transposed_output);
    SetSingleLayoutOnOp(fft_new_op, intermediate_layout);
    transposed_output = fft_new_op.getOutput();
    // Swap and relayout
    std::pair<int, int> perm_axes = {ax, input_rank - 1};
    TF_ASSIGN_OR_RETURN(
        transposed_output,
        EmitTransposeRelayout(builder, location, transposed_output,
                              intermediate_layout, mesh, perm_axes));
  }

  if (IsComplexFFT(ifft_op->getResult(0))) {
    // IFFT for the last axis.
    mlir::TF::IFFTOp ifft_output_op = builder.create<mlir::TF::IFFTOp>(
        location, transposed_output.getType(), transposed_output);
    SetSingleLayoutOnOp(ifft_output_op, intermediate_layout);
    builder.setInsertionPointAfter(ifft_output_op);

    ifft_op->getResult(0).replaceAllUsesWith(ifft_output_op);
    ifft_op->erase();
    return InferSPMDExpandedLocalShape(ifft_output_op);
  } else {
    TF_ASSIGN_OR_RETURN(auto complex_fft_length_vec,
                        ExtractFFTLengthFromOp(ifft_op));
    mlir::Value ifft_length =
        IntConst(builder, location,
                 (int32)complex_fft_length_vec[num_transform_axes - 1]);
    // IRFFT for the last axis.
    mlir::TF::IRFFTOp irfft_output_op = builder.create<mlir::TF::IRFFTOp>(
        location, ifft_op->getResult(0).getType(), transposed_output,
        ifft_length);
    SetSingleLayoutOnOp(irfft_output_op, intermediate_layout);
    builder.setInsertionPointAfter(irfft_output_op);
    ifft_op->getResult(0).replaceAllUsesWith(irfft_output_op.getOutput());
    ifft_op->erase();
    return InferSPMDExpandedLocalShape(irfft_output_op);
  }
}
}  // namespace

StatusOr<mlir::Operation*> FFTSPMDExpander::ExpandOp(mlir::Operation* op) {
  std::vector<int> last_axis{-1};
  std::vector<int> last_2_axes{-2, -1};
  std::vector<int> last_3_axes{-3, -2, -1};
  return llvm::TypeSwitch<mlir::Operation*, StatusOr<mlir::Operation*>>(op)
      // Forward prop ops.
      .Case<mlir::TF::FFTOp, mlir::TF::RFFTOp>(
          [&](auto op) { return ExpandFFTN(op, last_axis); })
      .Case<mlir::TF::FFT2DOp, mlir::TF::RFFT2DOp>(
          [&](auto op) { return ExpandFFTN(op, last_2_axes); })
      .Case<mlir::TF::FFT3DOp, mlir::TF::RFFT3DOp>(
          [&](auto op) { return ExpandFFTN(op, last_3_axes); })

      // Backward prop ops.
      .Case<mlir::TF::IFFTOp, mlir::TF::IRFFTOp>(
          [&](auto op) { return ExpandIFFTN(op, last_axis); })
      .Case<mlir::TF::IFFT2DOp, mlir::TF::IRFFT2DOp>(
          [&](auto op) { return ExpandIFFTN(op, last_2_axes); })
      .Case<mlir::TF::IFFT3DOp, mlir::TF::IRFFT3DOp>(
          [&](auto op) { return ExpandIFFTN(op, last_3_axes); })
      .Default([&](auto op) { return InferSPMDExpandedLocalShape(op); });
}

StatusOr<llvm::DenseMap<int, Layout>> FFTSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto iter = input_layouts.find(0);
  if (iter == input_layouts.end()) return llvm::DenseMap<int, Layout>();

  const Layout& input_layout = iter->second;
  std::vector<std::string> sharding_specs = input_layout.sharding_spec_strs();
  if (sharding_specs.empty())
    return absl::FailedPreconditionError(
        absl::StrCat(OpName(op), " has no sharding specs."));
  llvm::TypeSwitch<mlir::Operation*>(op)
      .Case<mlir::TF::FFTOp, mlir::TF::RFFTOp>(
          [&sharding_specs](auto op) { PropagateFFTLayout(sharding_specs, 1); })
      .Case<mlir::TF::FFT2DOp, mlir::TF::RFFT2DOp>(
          [&sharding_specs](auto op) { PropagateFFTLayout(sharding_specs, 2); })
      .Case<mlir::TF::FFT3DOp, mlir::TF::RFFT3DOp>(
          [&sharding_specs](auto op) { PropagateFFTLayout(sharding_specs, 3); })

      .Case<mlir::TF::IFFTOp, mlir::TF::IRFFTOp>([&sharding_specs](auto op) {
        PropagateIFFTLayout(sharding_specs, 1);
      })
      .Case<mlir::TF::IFFT2DOp, mlir::TF::IRFFT2DOp>(
          [&sharding_specs](auto op) {
            PropagateIFFTLayout(sharding_specs, 2);
          })
      .Case<mlir::TF::IFFT3DOp, mlir::TF::IRFFT3DOp>(
          [&sharding_specs](auto op) {
            PropagateIFFTLayout(sharding_specs, 3);
          });

  TF_ASSIGN_OR_RETURN(auto result_layout,
                      Layout::GetLayout(input_layout.type(), sharding_specs,
                                        input_layout.mesh()));
  if (result_layout.rank() != input_layout.rank())
    return absl::FailedPreconditionError(absl::StrCat(
        OpName(op), " derived output layout rank is ", result_layout.rank(),
        " not ", input_layout.rank(), " as expected."));

  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> FFTSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto iter = output_layouts.find(0);
  if (iter == output_layouts.end()) return llvm::DenseMap<int, Layout>();

  const Layout& output_layout = iter->second;
  std::vector<std::string> sharding_specs = output_layout.sharding_spec_strs();
  if (sharding_specs.empty())
    return absl::FailedPreconditionError(
        absl::StrCat(OpName(op), " has no sharding specs."));

  llvm::TypeSwitch<mlir::Operation*>(op)
      .Case<mlir::TF::FFTOp, mlir::TF::RFFTOp>([&sharding_specs](auto op) {
        PropagateIFFTLayout(sharding_specs, 1);
      })
      .Case<mlir::TF::FFT2DOp, mlir::TF::RFFT2DOp>([&sharding_specs](auto op) {
        PropagateIFFTLayout(sharding_specs, 2);
      })
      .Case<mlir::TF::FFT3DOp, mlir::TF::RFFT3DOp>([&sharding_specs](auto op) {
        PropagateIFFTLayout(sharding_specs, 3);
      })

      .Case<mlir::TF::IFFTOp, mlir::TF::IRFFTOp>(
          [&sharding_specs](auto op) { PropagateFFTLayout(sharding_specs, 1); })
      .Case<mlir::TF::IFFT2DOp, mlir::TF::IRFFT2DOp>(
          [&sharding_specs](auto op) { PropagateFFTLayout(sharding_specs, 2); })
      .Case<mlir::TF::IFFT3DOp, mlir::TF::IRFFT3DOp>(
          [&sharding_specs](auto op) {
            PropagateFFTLayout(sharding_specs, 3);
          });

  TF_ASSIGN_OR_RETURN(auto result_layout,
                      Layout::GetLayout(output_layout.type(), sharding_specs,
                                        output_layout.mesh()));
  if (result_layout.rank() != output_layout.rank())
    return absl::FailedPreconditionError(absl::StrCat(
        OpName(op), " derived output layout rank is ", result_layout.rank(),
        " not ", output_layout.rank(), " as expected."));

  return llvm::DenseMap<int, Layout>({{0, result_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
