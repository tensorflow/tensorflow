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

#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/utils/unregistered_attributes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::mlir::sdy::MeshAttr;

MeshAttr getSdyMeshAttr(mlir::sdy::TensorShardingAttr sharding,
                        std::optional<mlir::DictionaryAttr> sdy_meshes) {
  mlir::Attribute mesh_or_ref = sharding.getMeshOrRef();
  if (auto mesh = mlir::dyn_cast<MeshAttr>(mesh_or_ref); mesh != nullptr) {
    return mesh;
  }
  if (sdy_meshes.has_value()) {
    auto mesh_ref = mlir::cast<mlir::FlatSymbolRefAttr>(mesh_or_ref);
    return mlir::cast<MeshAttr>(sdy_meshes->get(mesh_ref.getValue()));
  }

  CHECK(false) << "Mesh not found with name: "
               << mlir::sdy::attributeToString(mesh_or_ref);
}

OpSharding CreateOpShardingFromSdySharding(
    mlir::sdy::TensorShardingAttr sdy_sharding,
    std::optional<mlir::DictionaryAttr> sdy_meshes,
    mlir::DictionaryAttr frontend_attrs) {
  std::function<MeshAttr(mlir::sdy::TensorShardingAttr)> get_mesh_attr =
      [&](mlir::sdy::TensorShardingAttr sharding) {
        return getSdyMeshAttr(sharding, sdy_meshes);
      };

  return xla::sdy::convertToHloSharding(sdy_sharding, get_mesh_attr).ToProto();
}
}  // namespace

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

ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::stablehlo::ConvDimensionNumbersAttr input) {
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

absl::StatusOr<xla::PrecisionConfig::Algorithm> ConvertDotAlgorithm(
    mlir::mhlo::DotAlgorithmAttr attr) {
  auto algorithm = mlir::hlo::detail::getKnownDotAlgorithm(
      attr.getLhsPrecisionType(), attr.getRhsPrecisionType(),
      attr.getAccumulationType(), attr.getLhsComponentCount(),
      attr.getRhsComponentCount(), attr.getNumPrimitiveOperations(),
      attr.getAllowImpreciseAccumulation());
  if (failed(algorithm)) return Internal("Unknown dot algorithm");

  switch (algorithm.value()) {
    case mlir::hlo::detail::KnownDotAlgorithm::ANY_F8_ANY_F8_F32:
      return xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::ANY_F8_ANY_F8_F32_FAST_ACCUM:
      return xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM;
    case mlir::hlo::detail::KnownDotAlgorithm::F16_F16_F16:
      return xla::PrecisionConfig::ALG_DOT_F16_F16_F16;
    case mlir::hlo::detail::KnownDotAlgorithm::F16_F16_F32:
      return xla::PrecisionConfig::ALG_DOT_F16_F16_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_BF16:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_BF16;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32_X3:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32_X6:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32_X9:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9;
    case mlir::hlo::detail::KnownDotAlgorithm::TF32_TF32_F32:
      return xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::TF32_TF32_F32_X3:
      return xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3;
    case mlir::hlo::detail::KnownDotAlgorithm::F32_F32_F32:
      return xla::PrecisionConfig::ALG_DOT_F32_F32_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::F64_F64_F64:
      return xla::PrecisionConfig::ALG_DOT_F64_F64_F64;
  }
  return Internal("Unknown dot algorithm");
}

absl::StatusOr<xla::PrecisionConfig::Algorithm> ConvertDotAlgorithm(
    mlir::stablehlo::DotAlgorithmAttr attr) {
  auto algorithm = mlir::hlo::detail::getKnownDotAlgorithm(
      attr.getLhsPrecisionType(), attr.getRhsPrecisionType(),
      attr.getAccumulationType(), attr.getLhsComponentCount(),
      attr.getRhsComponentCount(), attr.getNumPrimitiveOperations(),
      attr.getAllowImpreciseAccumulation());
  if (failed(algorithm)) return Internal("Unknown dot algorithm");

  switch (algorithm.value()) {
    case mlir::hlo::detail::KnownDotAlgorithm::ANY_F8_ANY_F8_F32:
      return xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::ANY_F8_ANY_F8_F32_FAST_ACCUM:
      return xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM;
    case mlir::hlo::detail::KnownDotAlgorithm::F16_F16_F16:
      return xla::PrecisionConfig::ALG_DOT_F16_F16_F16;
    case mlir::hlo::detail::KnownDotAlgorithm::F16_F16_F32:
      return xla::PrecisionConfig::ALG_DOT_F16_F16_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_BF16:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_BF16;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32_X3:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32_X6:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6;
    case mlir::hlo::detail::KnownDotAlgorithm::BF16_BF16_F32_X9:
      return xla::PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9;
    case mlir::hlo::detail::KnownDotAlgorithm::TF32_TF32_F32:
      return xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::TF32_TF32_F32_X3:
      return xla::PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3;
    case mlir::hlo::detail::KnownDotAlgorithm::F32_F32_F32:
      return xla::PrecisionConfig::ALG_DOT_F32_F32_F32;
    case mlir::hlo::detail::KnownDotAlgorithm::F64_F64_F64:
      return xla::PrecisionConfig::ALG_DOT_F64_F64_F64;
  }
  return Internal("Unknown dot algorithm");
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
    mlir::stablehlo::CustomCallApiVersion api_version) {
  switch (api_version) {
    case mlir::stablehlo::CustomCallApiVersion::API_VERSION_UNSPECIFIED:
      return xla::CustomCallApiVersion::API_VERSION_UNSPECIFIED;
    case mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL:
      return xla::CustomCallApiVersion::API_VERSION_ORIGINAL;
    case mlir::stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      return xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING;
    case mlir::stablehlo::CustomCallApiVersion::
        API_VERSION_STATUS_RETURNING_UNIFIED:
      return xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED;
    case mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI:
      return xla::CustomCallApiVersion::API_VERSION_TYPED_FFI;
    default:
      return InvalidArgument("Unknown CustomCallApiVersion enum value #%d",
                             api_version);
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

std::optional<xla::OpSharding> ConvertSharding(llvm::StringRef sharding) {
  xla::OpSharding sharding_proto;
  if (sharding_proto.ParseFromString(sharding.str())) return sharding_proto;
  absl::StatusOr<xla::HloSharding> sharding_cpp =
      xla::ParseSharding(sharding.str());
  if (sharding_cpp.ok()) return sharding_cpp->ToProto();
  return std::nullopt;
}

std::optional<xla::OriginalValueProto> ConvertOriginalValue(
    llvm::StringRef original_value) {
  absl::StatusOr<std::shared_ptr<xla::OriginalValue>> hlo_original_value =
      xla::ParseOriginalValue(
          absl::string_view(original_value.data(), original_value.size()));
  if (!hlo_original_value.ok()) {
    return std::nullopt;
  }
  return hlo_original_value.value()->ToProto();
}

std::optional<xla::HloInputOutputAliasProto> ConvertInputOutputAlias(
    llvm::ArrayRef<mlir::Attribute> aliasing) {
  if (aliasing.empty()) return std::nullopt;

  xla::HloInputOutputAliasProto input_output_alias_proto;
  for (auto attr : aliasing) {
    auto entry_attr = mlir::cast<mlir::DictionaryAttr>(attr);
    auto alias_attr = mlir::cast<mlir::DictionaryAttr>(entry_attr.get("alias"));
    mlir::ArrayRef<int64_t> output_index =
        mlir::cast<mlir::DenseI64ArrayAttr>(entry_attr.get("output_index"))
            .asArrayRef();
    mlir::ArrayRef<int64_t> parameter_index =
        mlir::cast<mlir::DenseI64ArrayAttr>(alias_attr.get("parameter_index"))
            .asArrayRef();
    HloInputOutputAliasProto::AliasEntryProto entry;
    entry.mutable_output_shape_index()->Add(output_index.begin(),
                                            output_index.end());
    entry.set_parameter_number(
        mlir::cast<mlir::IntegerAttr>(alias_attr.get("parameter_number"))
            .getInt());
    entry.mutable_parameter_shape_index()->Add(parameter_index.begin(),
                                               parameter_index.end());
    mlir::StringRef kind =
        mlir::cast<mlir::StringAttr>(alias_attr.get("kind")).getValue();
    if (kind == "may_alias")
      entry.set_kind(xla::Kind::MAY_ALIAS);
    else if (kind == "must_alias")
      entry.set_kind(xla::Kind::MUST_ALIAS);
    else
      entry.set_kind(xla::Kind::UNDEFINED_ALIAS);
    input_output_alias_proto.add_entries()->Swap(&entry);
  }
  return input_output_alias_proto;
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

std::optional<xla::OpSharding> ExtractShardyArgShardingFromFrontendAttrs(
    mlir::func::FuncOp function, int64_t arg_num,
    std::optional<mlir::DictionaryAttr> sdy_meshes) {
  if (mlir::DictionaryAttr arg_frontend_attrs =
          xla::sdy::getFuncArgFrontendAttrs(function, arg_num);
      arg_frontend_attrs != nullptr) {
    auto sdy_sharding =
        xla::sdy::parseStringAttr<mlir::sdy::TensorShardingAttr>(
            arg_frontend_attrs,
            xla::ToStringRef(HloSharding::kShardingFrontendAttrName));
    if (sdy_sharding != nullptr) {
      return CreateOpShardingFromSdySharding(sdy_sharding, sdy_meshes,
                                             arg_frontend_attrs);
    }
  }

  return std::nullopt;
}

std::optional<xla::OpSharding> ExtractShardyResultShardingFromFrontendAttrs(
    mlir::func::FuncOp function, int64_t res_num,
    std::optional<mlir::DictionaryAttr> sdy_meshes) {
  // If the result has a sharding, then the result will come from a custom call
  // that has the sharding attached.
  mlir::Operation* defining_op =
      mlir::sdy::getBodyTerminatorOperand(function, res_num).getDefiningOp();
  auto custom_call_op =
      mlir::dyn_cast_or_null<mlir::stablehlo::CustomCallOp>(defining_op);

  if (custom_call_op == nullptr ||
      custom_call_op.getCallTargetName() !=
          xla::sdy::kFuncResultShardingTargetName) {
    return std::nullopt;
  }

  mlir::DictionaryAttr op_frontend_attrs =
      xla::sdy::getFrontendAttrs(custom_call_op);
  CHECK(op_frontend_attrs != nullptr)
      << "xla.sdy.FuncResultSharding custom call should have frontend attrs";
  auto sharding_per_value_attr =
      xla::sdy::parseStringAttr<mlir::sdy::TensorShardingPerValueAttr>(
          op_frontend_attrs,
          xla::ToStringRef(HloSharding::kShardingFrontendAttrName));
  CHECK(sharding_per_value_attr != nullptr)
      << "Failed to parse sharding from frontend attrs";
  CHECK_EQ(sharding_per_value_attr.size(), 1)
      << "Expected exactly one sharding per FuncResultSharding";
  return CreateOpShardingFromSdySharding(sharding_per_value_attr.getSharding(0),
                                         sdy_meshes, op_frontend_attrs);
}

mlir::FailureOr<xla::Shape> ExtractXlaShape(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(xla::kXlaShape)) {
    return *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  }
  std::vector<xla::Shape> subshapes;
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    subshapes.push_back(xla::TypeToShape(result.getType()));
    if (subshapes.back().element_type() == xla::PRIMITIVE_TYPE_INVALID) {
      return op->emitError() << "result #" << index << " type is not supported";
    }
  }
  if (subshapes.size() > 1) {
    return xla::ShapeUtil::MakeTupleShape(subshapes);
  }
  return subshapes[0];
}

}  // namespace xla
