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
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/replica_group.h"
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
#include "xla/tsl/platform/statusor.h"
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

// Looks up a mesh in the module's frontend attributes.
absl::StatusOr<mlir::sdy::MeshAttr> FindMeshInFrontendAttributes(
    mlir::ModuleOp module_op, absl::string_view failed_lookup_name) {
  auto frontend_attrs = module_op->getAttrOfType<mlir::DictionaryAttr>(
      "mhlo.frontend_attributes");
  if (!frontend_attrs) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }
  auto meshes_str_attr = mlir::dyn_cast_or_null<mlir::StringAttr>(
      frontend_attrs.get("xla.sdy.meshes"));
  if (!meshes_str_attr) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }

  std::string unescaped_value;
  std::string error;
  if (!absl::CUnescape(meshes_str_attr.getValue().str(), &unescaped_value,
                       &error)) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }

  auto parsed_attr =
      mlir::parseAttribute(unescaped_value, module_op->getContext());
  if (!parsed_attr) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }

  auto parsed_dict = mlir::dyn_cast<mlir::DictionaryAttr>(parsed_attr);
  if (!parsed_dict) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }

  mlir::sdy::MeshAttr mesh_attr;
  if (auto mesh_val = parsed_dict.get(llvm::StringRef(
          failed_lookup_name.data(), failed_lookup_name.size()))) {
    mesh_attr = mlir::cast<mlir::sdy::MeshAttr>(mesh_val);
  } else if (parsed_dict.size() == 1) {
    mesh_attr =
        mlir::cast<mlir::sdy::MeshAttr>(parsed_dict.begin()->getValue());
  }

  if (!mesh_attr) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }

  return mesh_attr;
}

// Finds the SDY mesh attribute referenced by the MHLO attribute.
absl::StatusOr<mlir::sdy::MeshAttr> FindSdyMeshAttribute(
    mlir::mhlo::ReplicaGroupMeshAxesAttr attr, mlir::Operation* op) {
  mlir::Operation* mesh_op = nullptr;
  std::string failed_lookup_name;

  std::string mesh_name;
  if (auto flat_ref = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr.getMesh())) {
    mesh_name = flat_ref.getValue().str();
  } else if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(attr.getMesh())) {
    mesh_name = str_attr.getValue().str();
  }

  if (!mesh_name.empty()) {
    mesh_op = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::sdy::MeshOp>(
        op, mlir::StringAttr::get(op->getContext(), mesh_name));
    if (!mesh_op) {
      failed_lookup_name = mesh_name;
    }
  } else if (auto inline_mesh =
                 mlir::dyn_cast<mlir::sdy::MeshAttr>(attr.getMesh())) {
    return inline_mesh;
  } else {
    return InvalidArgument("Expected mesh symbol or inlined mesh attribute");
  }

  if (mesh_op) {
    return mlir::cast<mlir::sdy::MeshOp>(mesh_op).getMesh();
  }

  // Look in module frontend attributes for meshes.
  auto module_op = op->getParentOfType<mlir::ModuleOp>();
  if (!module_op) {
    return InvalidArgument("Could not find mesh symbol: %s",
                           failed_lookup_name);
  }

  return FindMeshInFrontendAttributes(module_op, failed_lookup_name);
}

mlir::stablehlo::MeshAttr ConvertSdyMeshToStablehloMesh(
    mlir::sdy::MeshAttr sdy_mesh, mlir::MLIRContext* context) {
  llvm::SmallVector<mlir::stablehlo::MeshAxisAttr> stablehloAxes;
  for (auto axis : sdy_mesh.getAxes()) {
    stablehloAxes.push_back(mlir::stablehlo::MeshAxisAttr::get(
        context, axis.getName(), axis.getSize()));
  }
  mlir::DenseIntElementsAttr device_ids;
  if (!sdy_mesh.getDeviceIds().empty()) {
    auto i64_type = mlir::IntegerType::get(context, 64);
    auto tensor_type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(sdy_mesh.getDeviceIds().size())}, i64_type);
    device_ids = mlir::DenseIntElementsAttr::get(
        tensor_type, llvm::ArrayRef<int64_t>(sdy_mesh.getDeviceIds().begin(),
                                             sdy_mesh.getDeviceIds().end()));
  }
  return mlir::stablehlo::MeshAttr::get(context, stablehloAxes, device_ids);
}

// Finds the StableHLO mesh attribute referenced by the StableHLO attribute.
absl::StatusOr<mlir::stablehlo::MeshAttr> FindStablehloMeshAttribute(
    mlir::stablehlo::ReplicaGroupMeshAxesAttr attr, mlir::Operation* op) {
  auto mesh_ref = attr.getMesh();
  if (auto inline_mesh = mlir::dyn_cast<mlir::stablehlo::MeshAttr>(mesh_ref)) {
    return inline_mesh;
  }

  std::string mesh_name;
  if (auto flat_ref = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(mesh_ref)) {
    mesh_name = flat_ref.getValue().str();
  } else if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(mesh_ref)) {
    mesh_name = str_attr.getValue().str();
  }

  if (!mesh_name.empty()) {
    auto mesh_op =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::sdy::MeshOp>(
            op, mlir::StringAttr::get(op->getContext(), mesh_name));
    if (mesh_op) {
      return ConvertSdyMeshToStablehloMesh(mesh_op.getMesh(), op->getContext());
    }
  }

  // Look in module frontend attributes for meshes.
  auto module_op = op->getParentOfType<mlir::ModuleOp>();
  if (module_op) {
    auto sdy_mesh_or = FindMeshInFrontendAttributes(module_op, mesh_name);
    if (sdy_mesh_or.ok()) {
      return ConvertSdyMeshToStablehloMesh(sdy_mesh_or.value(),
                                           op->getContext());
    }
  }

  return InvalidArgument("Expected mesh symbol or inlined mesh attribute");
}

struct MeshInfo {
  std::vector<std::string> axes_names;
  std::vector<int64_t> axes_sizes;
  std::vector<int64_t> device_ids;
};

// Extracts mesh info from SDY mesh attribute.
MeshInfo ExtractSdyMeshInfo(mlir::sdy::MeshAttr mesh_attr) {
  MeshInfo info;
  for (auto axis : mesh_attr.getAxes()) {
    info.axes_names.push_back(axis.getName().str());
    info.axes_sizes.push_back(axis.getSize());
  }
  if (!mesh_attr.getDeviceIds().empty()) {
    info.device_ids = std::vector<int64_t>(mesh_attr.getDeviceIds().begin(),
                                           mesh_attr.getDeviceIds().end());
  }
  if (info.device_ids.empty()) {
    int64_t total_size = 1;
    for (const auto& size : info.axes_sizes) {
      total_size *= size;
    }
    info.device_ids.resize(total_size);
    std::iota(info.device_ids.begin(), info.device_ids.end(), 0);
  }
  return info;
}

// Extracts mesh info from StableHLO mesh attribute.
MeshInfo ExtractStablehloMeshInfo(
    mlir::stablehlo::MeshAttr stablehlo_mesh_attr) {
  MeshInfo info;
  for (auto axis_attr : stablehlo_mesh_attr.getAxes()) {
    auto axis = mlir::cast<mlir::stablehlo::MeshAxisAttr>(axis_attr);
    info.axes_names.push_back(axis.getName().str());
    info.axes_sizes.push_back(axis.getSize());
  }
  if (stablehlo_mesh_attr.getDeviceIds()) {
    info.device_ids = std::vector<int64_t>(
        stablehlo_mesh_attr.getDeviceIds().getValues<int64_t>().begin(),
        stablehlo_mesh_attr.getDeviceIds().getValues<int64_t>().end());
  }
  if (info.device_ids.empty()) {
    int64_t total_size = 1;
    for (const auto& size : info.axes_sizes) {
      total_size *= size;
    }
    info.device_ids.resize(total_size);
    std::iota(info.device_ids.begin(), info.device_ids.end(), 0);
  }
  return info;
}

// Builds an XLA Mesh from extracted mesh info.
xla::Mesh BuildXlaMesh(const MeshInfo& info) {
  std::vector<absl::string_view> axes_names_sv;
  axes_names_sv.reserve(info.axes_names.size());
  for (const auto& name : info.axes_names) {
    axes_names_sv.push_back(name);
  }

  std::vector<int64_t> iota_ids(info.device_ids.size());
  std::iota(iota_ids.begin(), iota_ids.end(), 0);
  if (info.device_ids == iota_ids) {
    return xla::Mesh(info.axes_sizes, axes_names_sv);
  }
  xla::Array<int64_t> device_assignment(info.axes_sizes);
  device_assignment.SetValues(info.device_ids);
  return xla::Mesh(device_assignment, axes_names_sv);
}

// Builds a list of AxisRefs from an MHLO ReplicaGroupMeshAxesAttr.
// Builds a list of AxisRefs from a ReplicaGroupMeshAxesAttr.
template <typename AttrTy, typename AxisRefAttrTy>
absl::StatusOr<std::vector<xla::AxisRef>> BuildAxisRefs(
    AttrTy attr, const std::vector<std::string>& axes_names) {
  std::vector<xla::AxisRef> group_axes;
  for (auto axis_ref_attr : attr.getAxes()) {
    auto axis_ref = mlir::cast<AxisRefAttrTy>(axis_ref_attr);
    std::string name = axis_ref.getName().str();

    auto it = absl::c_find(axes_names, name);
    if (it == axes_names.end()) {
      return InvalidArgument("Unknown axis %s", name.c_str());
    }
    int64_t index = std::distance(axes_names.begin(), it);

    xla::AxisRefProto proto;
    proto.set_mesh_axis_index(index);
    if (auto sub = axis_ref.getSubAxisInfo()) {
      if (sub.getPreSize() <= 0) {
        return InvalidArgument("sub-axis pre-size must be >= 1, got %d",
                               sub.getPreSize());
      }
      if (sub.getSize() <= 1) {
        return InvalidArgument("sub-axis size must be > 1, got %d",
                               sub.getSize());
      }
      auto* sub_proto = proto.mutable_sub_axis_info();
      sub_proto->set_pre_size(sub.getPreSize());
      sub_proto->set_size(sub.getSize());
    }
    group_axes.push_back(xla::AxisRef::FromProto(proto));
  }
  return group_axes;
}

// Converts an MHLO mesh axes replica group attribute to XLA.
absl::StatusOr<std::unique_ptr<xla::CollectiveDeviceListBase>>
ConvertMhloMeshAxes(mlir::mhlo::ReplicaGroupMeshAxesAttr attr,
                    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto mesh_attr, FindSdyMeshAttribute(attr, op));
  auto info = ExtractSdyMeshInfo(mesh_attr);
  auto xla_mesh = BuildXlaMesh(info);
  TF_ASSIGN_OR_RETURN(
      auto group_axes,
      (BuildAxisRefs<mlir::mhlo::ReplicaGroupMeshAxesAttr,
                     mlir::mhlo::AxisRefAttr>(attr, info.axes_names)));
  return std::make_unique<xla::MeshAxesReplicaGroupList>(xla_mesh, group_axes);
}

// Converts a StableHLO mesh axes replica group attribute to XLA.
absl::StatusOr<std::unique_ptr<xla::CollectiveDeviceListBase>>
ConvertStablehloMeshAxes(mlir::stablehlo::ReplicaGroupMeshAxesAttr attr,
                         mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto mesh_attr, FindStablehloMeshAttribute(attr, op));
  auto info = ExtractStablehloMeshInfo(mesh_attr);
  auto xla_mesh = BuildXlaMesh(info);
  TF_ASSIGN_OR_RETURN(
      auto group_axes,
      (BuildAxisRefs<mlir::stablehlo::ReplicaGroupMeshAxesAttr,
                     mlir::stablehlo::AxisRefAttr>(attr, info.axes_names)));
  return std::make_unique<xla::MeshAxesReplicaGroupList>(xla_mesh, group_axes);
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

// Converts a replica groups attribute from MLIR to HLO.
// The attribute can be either a dense elements attribute (V1) or a
// ReplicaGroupMeshAxesAttr (V3).
// For V3, it looks up the corresponding mesh attribute, extracts mesh info,
// builds the XLA mesh, and builds the axis references.
absl::StatusOr<std::unique_ptr<xla::CollectiveDeviceListBase>>
ConvertReplicaGroups(mlir::Attribute replica_groups, mlir::Operation* op) {
  if (!replica_groups) {
    return std::make_unique<xla::CollectiveDeviceList>(
        std::vector<ReplicaGroup>());
  }

  if (auto dense_attr =
          mlir::dyn_cast<mlir::DenseIntElementsAttr>(replica_groups)) {
    TF_ASSIGN_OR_RETURN(std::vector<ReplicaGroup> groups,
                        ConvertReplicaGroups(dense_attr));
    return std::make_unique<xla::CollectiveDeviceList>(std::move(groups));
  }

  if (auto attr = mlir::dyn_cast<mlir::mhlo::ReplicaGroupMeshAxesAttr>(
          replica_groups)) {
    return ConvertMhloMeshAxes(attr, op);
  }
  if (auto attr = mlir::dyn_cast<mlir::stablehlo::ReplicaGroupMeshAxesAttr>(
          replica_groups)) {
    return ConvertStablehloMeshAxes(attr, op);
  }

  return InvalidArgument("Unknown replica groups attribute type");
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

absl::StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroupsToV1(
    mlir::Attribute replica_groups, mlir::Operation* op) {
  if (auto dense_attr =
          mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(replica_groups)) {
    return ConvertReplicaGroups(dense_attr);
  }
  TF_ASSIGN_OR_RETURN(auto device_list,
                      ConvertReplicaGroups(replica_groups, op));
  return device_list->replica_groups();
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
