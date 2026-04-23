/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/infer_dispatch_info.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

static std::vector<Shape> GetParameterShapes(const ComputationLayout& layout) {
  // For now, XLA programs compiled with multiple arguments for PJRT cannot use
  // tuples for any of their arguments, so we can assume that a tuple can only
  // arise when there is a single argument.
  std::vector<Shape> shapes;
  if (layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple()) {
    shapes.reserve(layout.parameter_shape(0).tuple_shapes().size());
    absl::c_copy(layout.parameter_shape(0).tuple_shapes(),
                 std::back_inserter(shapes));
  } else {
    shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& sl : layout.parameter_layouts()) {
      shapes.push_back(sl.shape());
    }
  }
  return shapes;
}

static absl::StatusOr<CommonPjRtLoadedExecutable::DispatchInfo>
InferDispatchInfo(
    CommonPjRtClient* client, std::vector<Shape> parameter_device_shapes,
    Shape output_device_shape, const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<CommonPjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    std::unique_ptr<CommonPjRtLoadedExecutable::DispatchInfo::Extras> extras,
    bool tuple_inputs) {
  CommonPjRtLoadedExecutable::DispatchInfo result{
      .parameter_device_shapes = std::move(parameter_device_shapes),
      .output_device_shape =
          std::make_shared<const Shape>(std::move(output_device_shape)),
      .addressable_devices = std::move(addressable_devices),
      .addressable_device_logical_ids =
          std::move(addressable_device_logical_ids),
      .device_assignment = std::move(device_assignment),
      .extras = std::move(extras),
  };
  for (const auto& shape : result.parameter_device_shapes) {
    TF_ASSIGN_OR_RETURN(int kind, client->GetMemorySpaceKindForShape(shape));
    result.parameter_memory_space_kind_ids.push_back(kind);
  }
  {
    absl::Span<const Shape> shapes =
        result.output_device_shape->IsTuple()
            ? absl::MakeSpan(result.output_device_shape->tuple_shapes())
            : absl::MakeSpan(&*result.output_device_shape, 1);
    result.output_memory_space_kind_ids.reserve(shapes.size());
    for (const auto& shape : shapes) {
      TF_ASSIGN_OR_RETURN(int kind, client->GetMemorySpaceKindForShape(shape));
      result.output_memory_space_kind_ids.push_back(kind);
    }
  }
  // Initializes information about which arguments to which executables must
  // be donated due to aliases that were specified by the computation.
  TF_ASSIGN_OR_RETURN(
      result.parameters_that_must_be_donated,
      ComputeParametersThatMustBeDonated(
          alias_config, result.parameter_device_shapes.size(), tuple_inputs));
  result.input_buffer_sizes_in_bytes.reserve(
      result.parameter_device_shapes.size());
  for (const Shape& shape : result.parameter_device_shapes) {
    DCHECK(!shape.IsTuple());
    TF_ASSIGN_OR_RETURN(int kind, client->GetMemorySpaceKindForShape(shape));
    TF_ASSIGN_OR_RETURN(int64_t size_in_bytes,
                        client->GetOnDeviceBytesCount(kind, shape));
    result.input_buffer_sizes_in_bytes.push_back(size_in_bytes);
  }
  return result;
}

absl::StatusOr<CommonPjRtLoadedExecutable::DispatchInfo> InferDispatchInfo(
    CommonPjRtClient* client, const ComputationLayout& layout,
    const HloInputOutputAliasConfig& alias_config,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<CommonPjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs) {
  return InferDispatchInfo(
      client, GetParameterShapes(layout), layout.result_shape(), alias_config,
      std::move(device_assignment), std::move(addressable_device_logical_ids),
      std::move(addressable_devices), nullptr, tuple_inputs);
}

absl::StatusOr<std::vector<int64_t>> GetShardShape(
    const xla::HloSharding& sharding, llvm::ArrayRef<int64_t> shape,
    size_t num_devices) {
  if (sharding.IsReplicatedOrSingleDevice() || sharding.IsManual() ||
      sharding.IsUnreduced() || sharding.IsUnknown()) {
    return std::vector<int64_t>{shape.begin(), shape.end()};
  }
  if (sharding.TotalNumTiles() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile count and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        sharding.TotalNumTiles(), num_devices, "??", sharding.ToString()));
  }
  if (shape.size() != sharding.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.size(), sharding.TiledDataRank()));
  }
  const absl::Span<const int64_t> tile_assignment_dims =
      sharding.tile_assignment().dimensions();
  std::vector<int64_t> tile_shape;
  tile_shape.reserve(shape.size());
  for (int64_t i = 0; i < shape.size(); ++i) {
    tile_shape.push_back(xla::CeilOfRatio(shape[i], tile_assignment_dims[i]));
  }
  return tile_shape;
}

absl::StatusOr<CommonPjRtLoadedExecutable::DispatchInfo> InferDispatchInfo(
    CommonPjRtClient* client, mlir::ModuleOp mlir_module,
    const CompileOptions& options,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<CommonPjRtLoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, bool tuple_inputs) {
  if (!device_assignment) {
    return absl::UnimplementedError(
        "Async compilation requires a device_assignment");
  }
  auto extras =
      std::make_unique<CommonPjRtLoadedExecutable::DispatchInfo::Extras>();
  extras->name =
      std::string(mlir_module.getSymName().value_or("?unknown program name?"));
  extras->num_partitions = device_assignment->replica_count();
  extras->num_replicas = device_assignment->replica_count();
  auto mesh = mlir_module.getOps<mlir::sdy::MeshOp>();
  if (mesh.empty() || std::next(mesh.begin()) != mesh.end()) {
    return absl::UnimplementedError("Exactly one shardy mesh is required.");
  }
  auto main = mlir_module.lookupSymbol<mlir::func::FuncOp>("main");
  if (main == nullptr) {
    return absl::UnimplementedError(
        "InferAvalFromModule called on module without main function.");
  }
  auto sharding_fn = [&](mlir::sdy::TensorShardingAttr sharding) {
    return sharding.getMesh(main);
  };
  auto get_xla_shape = [&](xla::HloSharding sharding, mlir::Type type)
      -> absl::StatusOr<
          std::tuple<xla::Shape, std::shared_ptr<const xla::PjRtLayout>>> {
    if (mlir::isa<mlir::UnrankedTensorType>(type)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type ", mlir::debugString(type)));
    }
    std::vector<int64_t> shard_shape;
    xla::PrimitiveType primitive_type;
    if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      llvm::ArrayRef<int64_t> dims = tensor_type.getShape();
      TF_ASSIGN_OR_RETURN(
          shard_shape,
          GetShardShape(sharding, dims,
                        extras->num_replicas * extras->num_partitions));
      primitive_type =
          xla::ConvertMlirTypeToPrimitiveType(tensor_type.getElementType());
    } else {
      primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
    }
    TF_ASSIGN_OR_RETURN(auto* memory_space,
                        addressable_devices[0]->default_memory_space());
    auto xla_shard_shape =
        xla::ShapeUtil::MakeShape(primitive_type, shard_shape);
    // TODO(parkers): Fix the nullptr layout.
    TF_ASSIGN_OR_RETURN(auto xla_shape,
                        client->MakeDefaultShapeForMemorySpace(
                            memory_space, xla_shard_shape, nullptr));
    auto layout = std::make_shared<PjRtLayout>(xla_shape.layout());
    return std::make_tuple(xla_shape, layout);
  };
  std::vector<xla::Shape> parameter_device_shapes;
  extras->parameter_shardings.emplace();
  extras->parameter_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumArguments(); ++i) {
    auto sharding_attr = mlir::sdy::getSharding(main.getArgument(i));
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Arg[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});
    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    TF_ASSIGN_OR_RETURN(
        (std::tie(shape, layout)),
        get_xla_shape(hlo_sharding, main.getArgumentTypes()[i]));

    parameter_device_shapes.push_back(shape);
    extras->parameter_shardings->push_back(hlo_sharding.ToProto());
    extras->parameter_layouts->push_back(std::move(layout));
  }
  std::vector<xla::Shape> result_shapes;
  extras->output_shardings.emplace();
  extras->output_layouts = std::vector<std::shared_ptr<const PjRtLayout>>();
  for (size_t i = 0; i < main.getNumResults(); ++i) {
    auto sharding_attr = mlir::sdy::getFuncResultSharding(main, i);
    // sharding_attr.dump();
    if (sharding_attr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Result[%d] is missing a sharding", i));
    }
    xla::HloSharding hlo_sharding =
        xla::sdy::convertToHloSharding(sharding_attr, sharding_fn,
                                       /*manualAxes=*/{});

    xla::Shape shape;
    std::shared_ptr<const xla::PjRtLayout> layout;
    TF_ASSIGN_OR_RETURN((std::tie(shape, layout)),
                        get_xla_shape(hlo_sharding, main.getResultTypes()[i]));

    result_shapes.push_back(std::move(shape));
    extras->output_shardings->push_back(hlo_sharding.ToProto());
    extras->output_layouts->push_back(std::move(layout));
  }
  Shape output_device_shape = (result_shapes.size() == 1)
                                  ? std::move(result_shapes[0])
                                  : xla::Shape(std::move(result_shapes));

  // TODO(parkers): implement aliasing.
  extras->input_output_alias_config =
      HloInputOutputAliasConfig(output_device_shape);

  const auto& input_output_alias_config = extras->input_output_alias_config;
  TF_ASSIGN_OR_RETURN(
      auto result,
      InferDispatchInfo(client, std::move(parameter_device_shapes),
                        std::move(output_device_shape),
                        input_output_alias_config, std::move(device_assignment),
                        std::move(addressable_device_logical_ids),
                        std::move(addressable_devices), std::move(extras),
                        tuple_inputs));

  result.extras->fingerprint = "";
  result.extras->parameter_memory_kinds.reserve(
      result.parameter_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.parameter_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    TF_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.parameter_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->parameter_memory_kinds.push_back(memory_space->kind());
  }
  result.extras->output_memory_kinds.reserve(
      result.output_memory_space_kind_ids.size());
  for (size_t i = 0; i < result.output_memory_space_kind_ids.size(); ++i) {
    auto* device = result.addressable_devices[0];
    TF_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
    for (auto* ms : device->memory_spaces()) {
      if (ms->kind_id() == result.output_memory_space_kind_ids[i]) {
        memory_space = ms;
      }
    }
    result.extras->output_memory_kinds.push_back(memory_space->kind());
  }
  return result;
}

}  // namespace xla
