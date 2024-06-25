/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"

#include <utility>

#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/dnn.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::mhlo::ConvDimensionNumbersAttr input) {
  ConvolutionDimensionNumbers output;

  output.set_input_batch_dimension(input.getInputBatchDimension());
  output.set_input_feature_dimension(input.getInputFeatureDimension());
  for (auto v : input.getInputSpatialDimensions()) {
    output.add_input_spatial_dimensions(v);
  }

  output.set_kernel_input_feature_dimension(
      input.getKernelInputFeatureDimension());
  output.set_kernel_output_feature_dimension(
      input.getKernelOutputFeatureDimension());

  for (auto v : input.getKernelSpatialDimensions()) {
    output.add_kernel_spatial_dimensions(v);
  }

  output.set_output_batch_dimension(input.getOutputBatchDimension());
  output.set_output_feature_dimension(input.getOutputFeatureDimension());

  for (auto v : input.getOutputSpatialDimensions()) {
    output.add_output_spatial_dimensions(v);
  }

  return output;
}

// Convert replica group from MLIR encoding to HLO.
// See HloFunctionImporter::ConvertReplicaGroups for the MLIR encoding.
absl::StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input) {
  mlir::RankedTensorType type =
      mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!type || type.getRank() != 2 ||
      !type.getElementType().isInteger(/*width=*/64)) {
    return Internal("Execpted replica group to be a rank 2 tensor of i64");
  }
  // rank 0 is num_groups, rank 1 is group size.
  auto replica_group_values_it = input.getValues<uint64_t>().begin();
  std::vector<ReplicaGroup> replica_groups(type.getDimSize(0));
  for (ReplicaGroup& group : replica_groups) {
    for (int64_t element_idx = 0; element_idx < type.getDimSize(1);
         ++element_idx, ++replica_group_values_it) {
      // For replica group attribute, -1 indicates padding added by
      // HloFunctionImporter::ConvertReplicaGroups. This should always be at the
      // end and can be dropped when converting back to XLA HLO ReplicaGroups.
      if (*replica_group_values_it != -1) {
        group.add_replica_ids(*replica_group_values_it);
      }
    }
  }
  return replica_groups;
}

// Convert a (N, 2) dense attribute to a list of tuples. This is the way padding
// and source-target pairs are defined in HLO.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> ConvertNx2Attribute(
    std::optional<mlir::DenseIntElementsAttr> optional_attr) {
  if (!optional_attr.has_value())
    return std::vector<std::pair<int64_t, int64_t>>{};
  mlir::DenseIntElementsAttr attr = *optional_attr;
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
  if (!type || type.getRank() != 2 || type.getShape()[1] != 2)
    return Internal("expected Nx2 attribute to be a tensor of shape Nx2");
  auto it = attr.getValues<int64_t>().begin();
  std::vector<std::pair<int64_t, int64_t>> out(attr.getNumElements() / 2);
  for (auto& item : out) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  return out;
}

absl::StatusOr<FftType> ConvertFftType(llvm::StringRef type_string) {
  std::optional<mlir::mhlo::FftType> type =
      mlir::mhlo::symbolizeEnum<mlir::mhlo::FftType>(type_string);
  if (!type) return InvalidArgument("Unknown FFT type %s", type_string.str());

  switch (*type) {
    case mlir::mhlo::FftType::FFT:
      return xla::FftType::FFT;
    case mlir::mhlo::FftType::IFFT:
      return xla::FftType::IFFT;
    case mlir::mhlo::FftType::RFFT:
      return xla::FftType::RFFT;
    case mlir::mhlo::FftType::IRFFT:
      return xla::FftType::IRFFT;
    default:
      return InvalidArgument("Unknown FFT type enum #%d", *type);
  }
}

absl::StatusOr<TriangularSolveOptions::Transpose> ConvertTranspose(
    llvm::StringRef transpose_string) {
  std::optional<mlir::mhlo::Transpose> transpose =
      mlir::mhlo::symbolizeTranspose(transpose_string);
  if (!transpose)
    return InvalidArgument("Unknown transpose type %s", transpose_string.str());

  switch (*transpose) {
    case mlir::mhlo::Transpose::NO_TRANSPOSE:
      return TriangularSolveOptions::NO_TRANSPOSE;
    case mlir::mhlo::Transpose::TRANSPOSE:
      return TriangularSolveOptions::TRANSPOSE;
    case mlir::mhlo::Transpose::ADJOINT:
      return TriangularSolveOptions::ADJOINT;
    case mlir::mhlo::Transpose::TRANSPOSE_INVALID:
      return TriangularSolveOptions::TRANSPOSE_INVALID;
    default:
      return InvalidArgument("Unknown transpose enum value #%d", *transpose);
  }
}

absl::StatusOr<xla::CustomCallSchedule> ConvertCustomCallSchedule(
    mlir::mhlo::CustomCallSchedule schedule) {
  switch (schedule) {
    case mlir::mhlo::CustomCallSchedule::NONE:
      return xla::CustomCallSchedule::SCHEDULE_NONE;
    case mlir::mhlo::CustomCallSchedule::LATEST:
      return xla::CustomCallSchedule::SCHEDULE_LATEST;
    case mlir::mhlo::CustomCallSchedule::EARLIEST:
      return xla::CustomCallSchedule::SCHEDULE_EARLIEST;
    default:
      return InvalidArgument("Unknown CustomCallSchedule enum value #%d",
                             schedule);
  }
}

absl::StatusOr<xla::CustomCallApiVersion> ConvertCustomCallApiVersion(
    mlir::mhlo::CustomCallApiVersion api_version) {
  switch (api_version) {
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_UNSPECIFIED:
      return xla::CustomCallApiVersion::API_VERSION_UNSPECIFIED;
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL:
      return xla::CustomCallApiVersion::API_VERSION_ORIGINAL;
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      return xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING;
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      return xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED;
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI:
      return xla::CustomCallApiVersion::API_VERSION_TYPED_FFI;
    default:
      return InvalidArgument("Unknown CustomCallApiVersion enum value #%d",
                             api_version);
  }
}

absl::StatusOr<
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>
ConvertOutputOperandAliasing(mlir::ArrayAttr aliasArrayAttr) {
  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>> aliasInfo;
  for (auto attr : aliasArrayAttr.getValue()) {
    auto alias = mlir::cast<mlir::mhlo::OutputOperandAliasAttr>(attr);
    ShapeIndex outputShapeIndex(alias.getOutputTupleIndices());
    ShapeIndex operandShapeIndex(alias.getOperandTupleIndices());
    aliasInfo.push_back(std::make_pair(
        outputShapeIndex,
        std::make_pair(alias.getOperandIndex(), operandShapeIndex)));
  }
  return aliasInfo;
}

std::optional<xla::OpSharding> ConvertSharding(llvm::StringRef sharding) {
  xla::OpSharding sharding_proto;
  if (sharding_proto.ParseFromString(sharding.str())) return sharding_proto;
  absl::StatusOr<xla::HloSharding> sharding_cpp =
      xla::ParseSharding(sharding.str());
  if (sharding_cpp.ok()) return sharding_cpp->ToProto();
  return std::nullopt;
}

DotDimensionNumbers ConvertDotDimensionNumbers(
    mlir::mhlo::DotDimensionNumbersAttr input) {
  DotDimensionNumbers output;

  for (auto v : input.getLhsBatchingDimensions()) {
    output.add_lhs_batch_dimensions(v);
  }

  for (auto v : input.getRhsBatchingDimensions()) {
    output.add_rhs_batch_dimensions(v);
  }

  for (auto v : input.getLhsContractingDimensions()) {
    output.add_lhs_contracting_dimensions(v);
  }

  for (auto v : input.getRhsContractingDimensions()) {
    output.add_rhs_contracting_dimensions(v);
  }

  return output;
}

DotDimensionNumbers ConvertDotDimensionNumbers(
    absl::Span<const int64_t> lhs_batch, absl::Span<const int64_t> lhs_contract,
    absl::Span<const int64_t> rhs_batch,
    absl::Span<const int64_t> rhs_contract) {
  DotDimensionNumbers output;
  for (auto v : lhs_batch) {
    output.add_lhs_batch_dimensions(v);
  }

  for (auto v : rhs_batch) {
    output.add_rhs_batch_dimensions(v);
  }

  for (auto v : lhs_contract) {
    output.add_lhs_contracting_dimensions(v);
  }

  for (auto v : rhs_contract) {
    output.add_rhs_contracting_dimensions(v);
  }

  return output;
}

absl::StatusOr<std::vector<int64_t>> ConvertMlirArrayAttrToInt64Array(
    const mlir::ArrayAttr& array) {
  int rank = array.size();
  std::vector<int64_t> converted_array(rank);
  for (int i = 0; i < rank; i++) {
    mlir::IntegerAttr attr = mlir::dyn_cast<mlir::IntegerAttr>(array[i]);
    if (!attr) {
      return Internal("Type Error: Expected layout integer attribute");
    }
    converted_array[i] = attr.getInt();
  }
  return converted_array;
}
}  // namespace xla
