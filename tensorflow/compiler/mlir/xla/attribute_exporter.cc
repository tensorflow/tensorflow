/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"

#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/stream_executor/dnn.h"

namespace xla {

ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::mhlo::ConvDimensionNumbers input) {
  ConvolutionDimensionNumbers output;

  output.set_input_batch_dimension(
      input.input_batch_dimension().getValue().getSExtValue());
  output.set_input_feature_dimension(
      input.input_feature_dimension().getValue().getSExtValue());

  for (auto v : input.input_spatial_dimensions().getValues<int64>()) {
    output.add_input_spatial_dimensions(v);
  }

  output.set_kernel_input_feature_dimension(
      input.kernel_input_feature_dimension().getValue().getSExtValue());
  output.set_kernel_output_feature_dimension(
      input.kernel_output_feature_dimension().getValue().getSExtValue());

  for (auto v : input.kernel_spatial_dimensions().getValues<int64>()) {
    output.add_kernel_spatial_dimensions(v);
  }

  output.set_output_batch_dimension(
      input.output_batch_dimension().getValue().getSExtValue());
  output.set_output_feature_dimension(
      input.output_feature_dimension().getValue().getSExtValue());

  for (auto v : input.output_spatial_dimensions().getValues<int64>()) {
    output.add_output_spatial_dimensions(v);
  }

  return output;
}

StatusOr<stream_executor::dnn::ActivationMode> ConvertConvActivationMode(
    llvm::StringRef input) {
  llvm::Optional<mlir::lmhlo_gpu::Activation> activation =
      mlir::lmhlo_gpu::symbolizeActivation(input);
  if (!activation) {
    return InternalError("Unexpected activation");
  }

  switch (activation.getValue()) {
    case mlir::lmhlo_gpu::Activation::None:
      return stream_executor::dnn::kNone;
    case mlir::lmhlo_gpu::Activation::Sigmoid:
      return stream_executor::dnn::kSigmoid;
    case mlir::lmhlo_gpu::Activation::Tanh:
      return stream_executor::dnn::kTanh;
    case mlir::lmhlo_gpu::Activation::Relu:
      return stream_executor::dnn::kRelu;
    case mlir::lmhlo_gpu::Activation::Relu6:
      return stream_executor::dnn::kRelu6;
    case mlir::lmhlo_gpu::Activation::ReluX:
      return stream_executor::dnn::kReluX;
    case mlir::lmhlo_gpu::Activation::BandPass:
      return stream_executor::dnn::kBandPass;
    default:
      return InternalError("Unexpected activation");
  }
}

// Convert replica group from MLIR encoding to HLO.
// See HloFunctionImporter::ConvertReplicaGroups for the MLIR encoding.
StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input) {
  mlir::RankedTensorType type =
      input.getType().dyn_cast<mlir::RankedTensorType>();
  if (!type || type.getRank() != 2 ||
      !type.getElementType().isInteger(/*width=*/64)) {
    return InternalError("Execpted replica group to be a rank 2 tensor of i64");
  }
  // rank 0 is num_groups, rank 1 is group size.
  auto replica_group_values_it = input.getValues<uint64_t>().begin();
  std::vector<ReplicaGroup> replica_groups(type.getDimSize(0));
  for (ReplicaGroup &group : replica_groups) {
    for (int64 element_idx = 0; element_idx < type.getDimSize(1);
         ++element_idx, ++replica_group_values_it) {
      // For replica group attribute, -1 indicates padding added by
      // ConvertReplicaGroups. This show always be at the end and can be dropped
      // when converting back to XLA HLO ReplicaGroups.
      if (*replica_group_values_it != -1) {
        group.add_replica_ids(*replica_group_values_it);
      }
    }
  }
  return replica_groups;
}
}  // namespace xla
