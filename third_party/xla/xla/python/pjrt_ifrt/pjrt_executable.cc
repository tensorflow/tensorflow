/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/pjrt_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/type_registry.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/utils.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/serdes_week_4_old_version_accessor.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_device.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/python/pjrt_ifrt/pjrt_memory.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pjrt_ifrt/xla_executable_version.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/runtime/device_id.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {
namespace ifrt {

namespace {

static const MemoryKind kDefaultMemoryKind("device");

// Returns a pair of flat lists of IFRT dtypes and shapes from XLA shapes
// extracted from an MLIR module's signature.
absl::StatusOr<std::pair<std::vector<DType>, std::vector<Shape>>>
GetDTypesAndShapes(absl::Span<const xla::Shape> mlir_module_xla_shapes) {
  std::vector<DType> dtypes;
  dtypes.reserve(mlir_module_xla_shapes.size());
  std::vector<Shape> shapes;
  shapes.reserve(mlir_module_xla_shapes.size());
  for (const xla::Shape& xla_shape : mlir_module_xla_shapes) {
    TF_ASSIGN_OR_RETURN(DType dtype, ToDType(xla_shape.element_type()));
    dtypes.push_back(dtype);
    if (dtype.kind() == DType::kToken) {
      // Token uses a scalar shape by convention.
      shapes.push_back(Shape({}));
    } else {
      shapes.push_back(Shape(xla_shape.dimensions()));
    }
  }
  return std::make_pair(std::move(dtypes), std::move(shapes));
}

// Returns a flat list of HLO shardings from the sharding information that a
// PjRt executable returns (a flat list of `OpSharding`s, with some special
// cases). Returns `std::nullopt` if the executable does not have sharding
// information.
absl::StatusOr<std::optional<std::vector<xla::HloSharding>>> GetHloShardings(
    const std::optional<std::vector<xla::OpSharding>>&
        pjrt_executable_op_shardings,
    absl::Span<const DType> dtypes, bool is_output) {
  if (!pjrt_executable_op_shardings.has_value()) {
    return std::nullopt;
  }
  std::vector<xla::HloSharding> hlo_shardings;
  if (is_output && dtypes.empty()) {
    // If the HLO module output is an empty tuple, the output sharding will have
    // a single element for the tuple as a special case. We allow this condition
    // by checking this condition specifically.
    if (pjrt_executable_op_shardings->size() != 1) {
      return FailedPrecondition(
          "HLO module output is an empty tuple, but the output sharding has "
          "%d elements",
          pjrt_executable_op_shardings->size());
    }
    return std::vector<xla::HloSharding>();
  }
  if (pjrt_executable_op_shardings->size() != dtypes.size()) {
    return FailedPrecondition(
        "Output shardings and dtypes have different sizes: %d vs. %d",
        pjrt_executable_op_shardings->size(), dtypes.size());
  }
  hlo_shardings.reserve(pjrt_executable_op_shardings->size());
  for (int i = 0; i < pjrt_executable_op_shardings->size(); ++i) {
    if (dtypes[i].kind() == DType::kToken) {
      // Token uses a fully replicated sharding by convention.
      hlo_shardings.push_back(xla::HloSharding::Replicate());
    } else {
      TF_ASSIGN_OR_RETURN(
          auto hlo_sharding,
          xla::HloSharding::FromProto((*pjrt_executable_op_shardings)[i]));
      hlo_shardings.push_back(hlo_sharding);
    }
  }
  return hlo_shardings;
}

// Returns a flat list of IFRT memory kinds from the memory kind information
// that a PjRt executable returns (per-module lists of memory kind strings).
// PjRt-IFRT always uses the first module's information.
absl::StatusOr<std::vector<MemoryKind>> GetMemoryKinds(
    const absl::StatusOr<std::vector<std::vector<absl::string_view>>>&
        pjrt_executable_memory_kinds,
    absl::Span<const DType> dtypes) {
  std::vector<MemoryKind> memory_kinds;
  // An unimplemented error is converted into all-default memory kinds.
  if (absl::IsUnimplemented(pjrt_executable_memory_kinds.status())) {
    memory_kinds.resize(/*size=*/dtypes.size(), /*value=*/kDefaultMemoryKind);
    return memory_kinds;
  }
  TF_RETURN_IF_ERROR(pjrt_executable_memory_kinds.status());
  if (pjrt_executable_memory_kinds->empty()) {
    return FailedPrecondition("No module found");
  }
  if (pjrt_executable_memory_kinds->front().size() != dtypes.size()) {
    return FailedPrecondition(
        "Memory kinds and dtypes have different sizes: %d vs. %d",
        pjrt_executable_memory_kinds->front().size(), dtypes.size());
  }
  memory_kinds.reserve(pjrt_executable_memory_kinds->front().size());
  for (int i = 0; i < pjrt_executable_memory_kinds->front().size(); ++i) {
    if (dtypes[i].kind() == DType::kToken) {
      // Token uses a device memory kind by convention.
      memory_kinds.push_back(kDefaultMemoryKind);
    } else {
      absl::string_view memory_kind = pjrt_executable_memory_kinds->front()[i];
      if (memory_kind.empty()) {
        memory_kinds.push_back(kDefaultMemoryKind);
      } else {
        memory_kinds.push_back(MemoryKind(memory_kind));
      }
    }
  }
  return memory_kinds;
}

// Makes IFRT shardings created from HLO shardings and memory kinds.
std::vector<ShardingRef> MakeShardings(
    absl::Span<const Shape> shapes,
    const std::optional<std::vector<xla::HloSharding>>& hlo_shardings,
    absl::Span<const MemoryKind> memory_kinds,
    const DeviceListRef& executable_devices) {
  std::vector<ShardingRef> shardings;
  shardings.reserve(memory_kinds.size());
  if (hlo_shardings.has_value()) {
    for (int i = 0; i < memory_kinds.size(); ++i) {
      shardings.push_back(ifrt::HloSharding::Create(
          executable_devices, memory_kinds[i], (*hlo_shardings)[i]));
    }
  } else {
    // Assume a traditional replication computation where tile shapes are the
    // same as global shapes.
    //
    // TODO(hyeontaek): Remove this branch once every output uses `HloSharding`.
    for (int i = 0; i < memory_kinds.size(); ++i) {
      shardings.push_back(ifrt::ConcreteEvenSharding::Create(
          executable_devices, memory_kinds[i],
          /*shape=*/shapes[i],
          /*shard_shape=*/shapes[i]));
    }
  }
  return shardings;
}

// Returns a flat list of layouts by combining layout modes and PjRt executable
// layouts.
// If any error other than an unimplemented error happens, returns
// `std::nullopt`. The layout will be determined at execute time.
//
// TODO(hyeontaek): Remove the nullopt path once obtaining layout modes and
// concrete layouts avoids HLO module serialization/deserialization and always
// succeeds.
absl::StatusOr<
    std::optional<std::vector<std::shared_ptr<const xla::PjRtLayout>>>>
GetLayouts(
    const absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>&
        pjrt_executable_layouts,
    absl::Span<const xla::LayoutMode> layout_modes) {
  // An unimplemented error is converted into all-default layouts.
  if (absl::IsUnimplemented(pjrt_executable_layouts.status())) {
    return std::vector<std::shared_ptr<const xla::PjRtLayout>>(
        /*size=*/layout_modes.size(), /*value=*/nullptr);
  }
  if (!pjrt_executable_layouts.ok()) {
    return std::nullopt;
  }
  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  if (pjrt_executable_layouts->size() != layout_modes.size()) {
    return FailedPrecondition(
        "Layouts and layout modes have different sizes: %d vs. %d",
        pjrt_executable_layouts->size(), layout_modes.size());
  }
  layouts.reserve(pjrt_executable_layouts->size());
  for (int i = 0; i < pjrt_executable_layouts->size(); ++i) {
    if (layout_modes[i].mode == xla::LayoutMode::Mode::kDefault) {
      layouts.push_back(nullptr);
    } else {
      layouts.push_back(std::move((*pjrt_executable_layouts)[i]));
    }
  }
  return layouts;
}

// Returns a list of donatable input indices from the given MLIR module.
absl::StatusOr<std::vector<int>> GetDonatableInputIndicesFromMlirModule(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) {
    return absl::InvalidArgumentError("MLIR module must have a main function");
  }
  mlir::FunctionType func_type = main_func.getFunctionType();

  std::optional<mlir::TypeRange> arg_types;
  bool tupled_args = false;
  if (func_type.getNumInputs() == 1) {
    auto tuple_type = llvm::dyn_cast<mlir::TupleType>(func_type.getInput(0));
    if (tuple_type) {
      tupled_args = true;
      arg_types = tuple_type.getTypes();
    }
  }
  if (!arg_types.has_value()) {
    arg_types = func_type.getInputs();
  }

  std::vector<int> donatable_input_indices;
  for (const auto& [i, arg] : llvm::enumerate(*arg_types)) {
    const int index = tupled_args ? 0 : i;
    if (auto donor = main_func.getArgAttrOfType<mlir::BoolAttr>(
            index, "jax.buffer_donor");
        donor && donor.getValue()) {
      donatable_input_indices.push_back(index);
    } else if (main_func.getArgAttrOfType<mlir::IntegerAttr>(
                   index, "tf.aliasing_output")) {
      donatable_input_indices.push_back(index);
    }
  }
  return donatable_input_indices;
}

// Returns a list of result shapes from the given MLIR module.
absl::StatusOr<std::vector<xla::Shape>> ResultShapesOfModule(
    mlir::ModuleOp module) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return InvalidArgument("MLIR module has no main function");
  }
  mlir::FunctionType type = main.getFunctionType();
  std::vector<xla::Shape> result_shapes;
  result_shapes.reserve(type.getNumResults());
  for (unsigned i = 0; i < type.getNumResults(); ++i) {
    mlir::Type result_type = type.getResult(i);
    result_shapes.push_back(xla::TypeToShape(result_type));
  }
  return result_shapes;
}

// Returns a new `DeviceListRef` that contains the addressable devices of the
// PjRt executable if the supplied `executable_devices` has an incomplete set of
// devices.
absl::StatusOr<DeviceListRef> AdjustExecutableDevicesForPmap(
    PjRtClient* client, const xla::PjRtLoadedExecutable* pjrt_loaded_executable,
    DeviceListRef executable_devices) {
  // For jit(pmap(...)), the device assignment (passed as `executable_devices`)
  // may contain a single device while the PjRt executable has multiple
  // addressable devices. We check for this condition and replace
  // `executable_devices` with the executable's addressable devices if
  // necessary.
  if (pjrt_loaded_executable->num_replicas() > 1 &&
      executable_devices->devices().size() == 1) {
    if (pjrt_loaded_executable->addressable_devices().size() > 1) {
      BasicDeviceList::Devices ds;
      ds.reserve(pjrt_loaded_executable->addressable_devices().size());
      for (xla::PjRtDevice* device :
           pjrt_loaded_executable->addressable_devices()) {
        TF_ASSIGN_OR_RETURN(Device * ifrt_device,
                            client->LookupPjRtDevice(device));
        ds.push_back(ifrt_device);
      }
      executable_devices = BasicDeviceList::Create(std::move(ds));
    } else if (pjrt_loaded_executable->addressable_devices().size() == 1) {
      TF_ASSIGN_OR_RETURN(
          Device * ifrt_device,
          client->LookupPjRtDevice(
              pjrt_loaded_executable->addressable_devices().front()));
      if (ifrt_device != executable_devices->devices().front()) {
        return FailedPrecondition(
            "Addressable device does not match sharding device");
      }
    }
  }
  if (executable_devices->devices().size() <
      pjrt_loaded_executable->addressable_devices().size()) {
    return FailedPrecondition(
        "Sharding devices must be at least as many as addressable devices");
  }
  return executable_devices;
}

// Gathers all `PjRtHostSendAndRecvLoadedHostCallback` from the given list of
// loaded host callbacks.
std::vector<PjRtHostSendAndRecvLoadedHostCallback*>
GatherHostSendAndRecvCallbacks(
    absl::Span<const tsl::RCReference<LoadedHostCallback>>
        loaded_host_callbacks) {
  std::vector<PjRtHostSendAndRecvLoadedHostCallback*>
      host_send_and_recv_callbacks;
  host_send_and_recv_callbacks.reserve(loaded_host_callbacks.size());
  // Gather all `PjRtLoadedHostCallback` separately, as each execution will
  // register `PjRtLoadedHostCallback` for host send and recv. All host
  // callbacks will be referenced by the executable and any pending execution to
  // guarantee the liveliness of host callbacks during executions.
  for (auto& loaded_host_callback : loaded_host_callbacks) {
    auto* host_send_and_recv_callback =
        llvm::dyn_cast<PjRtHostSendAndRecvLoadedHostCallback>(
            loaded_host_callback.get());
    if (host_send_and_recv_callback != nullptr) {
      host_send_and_recv_callbacks.push_back(host_send_and_recv_callback);
    }
  }
  return host_send_and_recv_callbacks;
}

}  // namespace

char PjRtCompatibleExecutable::ID = 0;
char PjRtCompatibleLoadedExecutable::ID = 0;
char PjRtExecutable::ID = 0;
char PjRtLoadedExecutable::ID = 0;

absl::StatusOr<ExecutableRef> PjRtExecutable::Create(
    mlir::ModuleOp module, xla::CompileOptions compile_options,
    const xla::PjRtTopologyDescription& topology) {
  const bool is_portable = compile_options.compile_portable_executable;

  // We have to do process the MLIR before the compile call, since the latter
  // will use the MLIR as scratch space, or possibly even deallocate it.
  TF_ASSIGN_OR_RETURN(std::vector<int> donatable_input_indices,
                      GetDonatableInputIndicesFromMlirModule(module));
  TF_ASSIGN_OR_RETURN(
      const std::vector<xla::Shape> mlir_module_output_xla_shapes,
      ResultShapesOfModule(module));
  TF_ASSIGN_OR_RETURN(const std::vector<xla::LayoutMode> output_layout_modes,
                      GetOutputLayoutModes(module));

  TF_ASSIGN_OR_RETURN(auto pjrt_executable,
                      PjRtCompile(std::move(compile_options), std::move(module),
                                  topology, /*client=*/nullptr));

  TF_ASSIGN_OR_RETURN(auto output_dtypes_and_shapes,
                      GetDTypesAndShapes(mlir_module_output_xla_shapes));
  std::vector<DType> output_dtypes = std::move(output_dtypes_and_shapes.first);
  std::vector<Shape> output_shapes = std::move(output_dtypes_and_shapes.second);
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<xla::HloSharding>> output_hlo_shardings,
      GetHloShardings(pjrt_executable->GetOutputShardings(), output_dtypes,
                      /*is_output=*/true));

  TF_ASSIGN_OR_RETURN(
      std::vector<MemoryKind> output_memory_kinds,
      GetMemoryKinds(pjrt_executable->GetOutputMemoryKinds(), output_dtypes));

  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
          output_layouts,
      GetLayouts(pjrt_executable->GetOutputLayouts(), output_layout_modes));

  CommonMetadata common_metadata;
  common_metadata.is_portable = is_portable;
  common_metadata.donatable_input_indices = std::move(donatable_input_indices);
  common_metadata.output_dtypes = std::move(output_dtypes);
  common_metadata.output_shapes = std::move(output_shapes);
  common_metadata.output_hlo_shardings = std::move(output_hlo_shardings);
  common_metadata.output_memory_kinds = std::move(output_memory_kinds);
  common_metadata.output_layouts = std::move(output_layouts);

  return ExecutableRef(new PjRtExecutable(std::move(pjrt_executable),
                                          std::move(common_metadata)));
}

PjRtExecutable::PjRtExecutable(
    std::shared_ptr<xla::PjRtExecutable> pjrt_executable,
    CommonMetadata common_metadata)
    : pjrt_executable_(std::move(pjrt_executable)),
      common_metadata_(std::move(common_metadata)) {
  pjrt_output_memory_kinds_.emplace_back();
  for (const MemoryKind& memory_kind : common_metadata_.output_memory_kinds) {
    pjrt_output_memory_kinds_.back().push_back(
        memory_kind.memory_kind().value_or(""));
  }
}

absl::StatusOr<std::optional<std::string>> PjRtExecutable::Fingerprint() const {
  DCHECK(this);
  return pjrt_executable_->FingerprintExecutable();
}

absl::StatusOr<std::string> PjRtExecutable::CommonMetadata::Serialize(
    xla::PjRtExecutable* pjrt_executable) const {
  const SerDesVersion serdes_version = SerDesWeek4OldVersionAccessor::Get();

  SerializedXlaExecutableMetadata metadata;
  metadata.set_ifrt_version_number(serdes_version.version_number().value());
  metadata.set_runtime_name(PjRtClient::kRuntimeType);

  // PjRt-IFRT currently does not track XLA executable versions.
  auto xla_executable_version = std::make_unique<XlaExecutableVersion>();
  TF_ASSIGN_OR_RETURN(
      SerializedXlaExecutableVersion serialized_executable_version,
      xla_executable_version->ToProto(serdes_version));
  *metadata.mutable_executable_version() = serialized_executable_version;

  metadata.set_computation_name(pjrt_executable->name());

  // Encode output specs.
  for (int i = 0; i < output_dtypes.size(); ++i) {
    SerializedXlaExecutableMetadata::OutputSpec& output_spec =
        *metadata.add_output_specs();
    // Layout - only populate if it's not the default layout
    if (output_layouts.has_value() && (*output_layouts)[i] != nullptr) {
      auto pjrt_layout = PjRtLayout::Create((*output_layouts)[i]);
      TF_ASSIGN_OR_RETURN(*output_spec.mutable_layout(),
                          pjrt_layout->ToProto(serdes_version));
    }

    // Sharding
    if (output_hlo_shardings.has_value()) {
      *output_spec.mutable_op_sharding() = (*output_hlo_shardings)[i].ToProto();
    }

    // Memory kind
    output_spec.set_memory_kind(
        std::string(output_memory_kinds[i].memory_kind().value_or("")));

    // Shape
    std::optional<Shape> shard_shape;
    {
      // If the output sharding information is missing (e.g., serializing a
      // previously deserialized executable) or the output dtype is token, the
      // output shard shape is the same as the output shape. The former case
      // will be removed once serialization/deserialization roundtrip preserves
      // the output sharding information.
      if (!output_hlo_shardings.has_value() ||
          output_dtypes[i].kind() == DType::kToken) {
        shard_shape = output_shapes[i];
      } else {
        TF_ASSIGN_OR_RETURN(xla::PrimitiveType element_type,
                            ToPrimitiveType(output_dtypes[i]));
        xla::Shape xla_shape(element_type, output_shapes[i].dims());
        xla::Shape xla_shard_shape = xla::hlo_sharding_util::TileShape(
            (*output_hlo_shardings)[i], xla_shape);
        shard_shape = Shape(xla_shape.dimensions());
      }
    }
    *output_spec.mutable_shard_shape() = shard_shape->ToProto(serdes_version);
    *output_spec.mutable_shape() = output_shapes[i].ToProto(serdes_version);

    // DType
    *output_spec.mutable_dtype() = output_dtypes[i].ToProto(serdes_version);
  }

  // Get parameter specs.
  const std::optional<std::vector<xla::OpSharding>> parameter_shardings =
      pjrt_executable->GetParameterShardings();
  uint64_t num_parameters;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts;
  if (auto maybe_layouts = pjrt_executable->GetParameterLayouts();
      maybe_layouts.ok()) {
    num_parameters = maybe_layouts->size();
    parameter_layouts = *std::move(maybe_layouts);
  } else if (absl::IsUnimplemented(maybe_layouts.status())) {
    // An unimplemented error is converted into all-default layouts. This
    // happens when a parameter is a token or opaque. Since parameter layouts
    // are not load bearing in such a case (no custom layouts are expected to be
    // used), we can use a default layout. However, this should be fixed in both
    // PjRt and PjRt-IFRT so that token and opaque parameters can be used with
    // other parameters that have custom layouts.
    //
    // TODO(hyeontaek): Once parameter layouts are changed to use the IFRT
    // layout style (default layouts are not using concrete PjRt layouts),
    // extract parameters directly from StableHLO and avoid calling
    // `GetParameterLayouts()`.
    num_parameters =
        parameter_shardings.has_value() ? parameter_shardings->size() : 0;
    parameter_layouts = std::vector<std::shared_ptr<const xla::PjRtLayout>>(
        /*size=*/num_parameters,
        /*value=*/nullptr);
  } else {
    return maybe_layouts.status();
  }
  if (parameter_shardings.has_value()) {
    if (parameter_shardings->size() != num_parameters) {
      return FailedPrecondition(
          "Parameter shardings have a different size from parameter layouts: "
          "%d vs. %d",
          parameter_shardings->size(), num_parameters);
    }
  }
  absl::flat_hash_set<int> donated_inputs_set(donatable_input_indices.begin(),
                                              donatable_input_indices.end());

  // Encode parameter specs.
  for (int i = 0; i < num_parameters; ++i) {
    SerializedXlaExecutableMetadata::ParameterSpec& parameter_spec =
        *metadata.add_parameter_specs();
    // Layout
    auto pjrt_layout = PjRtLayout::Create(parameter_layouts[i]);
    TF_ASSIGN_OR_RETURN(*parameter_spec.mutable_layout(),
                        pjrt_layout->ToProto(serdes_version));

    // Sharding
    if (parameter_shardings.has_value()) {
      *parameter_spec.mutable_op_sharding() = parameter_shardings->at(i);
    }

    // Donated input
    bool is_donated = donated_inputs_set.contains(i);
    parameter_spec.set_donated_input(is_donated);
  }

  metadata.set_portable(is_portable);

  // Write the metadata to a serialized executable string.
  std::string serialized_executable;
  {
    tsl::protobuf::io::StringOutputStream output_stream(&serialized_executable);
    if (!tsl::protobuf::util::SerializeDelimitedToZeroCopyStream(
            std::move(metadata), &output_stream)) {
      return absl::InternalError("Failed to serialize executable metadata.");
    }
  }

  // Get and write the serialized PjRt executable to string.
  TF_ASSIGN_OR_RETURN(std::string serialized_pjrt_executable,
                      pjrt_executable->SerializeExecutable());
  serialized_executable.append(std::move(serialized_pjrt_executable));

  return serialized_executable;
}

absl::StatusOr<std::pair<PjRtExecutable::CommonMetadata, absl::string_view>>
PjRtExecutable::CommonMetadata::Deserialize(
    absl::string_view serialized_executable,
    absl::FunctionRef<absl::Status(const ExecutableVersion& executable_version,
                                   const DeviceListRef& devices)>
        is_executable_version_compatible,
    const XlaDeserializeExecutableOptions& xla_deserialize_executable_options) {
  SerializedXlaExecutableMetadata metadata;
  tsl::protobuf::io::ArrayInputStream input_stream(
      serialized_executable.data(), serialized_executable.size());
  if (!tsl::protobuf::util::ParseDelimitedFromZeroCopyStream(
          &metadata, &input_stream, nullptr)) {
    return absl::InvalidArgumentError(
        "Failed to parse SerializedXlaExecutableMetadata");
  }

  absl::string_view serialized_pjrt_executable =
      serialized_executable.substr(input_stream.ByteCount());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::XlaExecutableVersion> executable_version,
      xla::ifrt::XlaExecutableVersion::FromProto(
          metadata.executable_version()));
  if (xla_deserialize_executable_options.devices.has_value()) {
    auto executable_version_compatible = is_executable_version_compatible(
        *executable_version, *xla_deserialize_executable_options.devices);
    // Handle an `UnimplementedError` gracefully. PjRt-IFRT currently does not
    // track XLA executable versions.
    if (absl::IsUnimplemented(executable_version_compatible)) {
      LOG(WARNING) << "Assume version compatibility. PjRt-IFRT does not track "
                      "XLA executable versions.";
    } else {
      TF_RETURN_IF_ERROR(executable_version_compatible);
    }
  } else {
    // Accept unspecified `devices` for now.
    // TODO(hyeontaek): Require all callers to specify `devices` and remove this
    // branch.
  }

  std::vector<int> donated_input_indices;
  std::vector<DType> output_dtypes;
  std::vector<Shape> output_shapes;
  std::optional<std::vector<xla::HloSharding>> output_hlo_shardings;
  std::vector<MemoryKind> output_memory_kinds;
  std::optional<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
      output_layouts;
  output_layouts.emplace();

  output_dtypes.reserve(metadata.output_specs_size());
  output_shapes.reserve(metadata.output_specs_size());
  output_memory_kinds.reserve(metadata.output_specs_size());
  output_layouts->reserve(metadata.output_specs_size());

  for (int i = 0; i < metadata.parameter_specs_size(); ++i) {
    if (metadata.parameter_specs(i).donated_input()) {
      donated_input_indices.push_back(i);
    }
  }

  for (const auto& output_spec : metadata.output_specs()) {
    TF_ASSIGN_OR_RETURN(auto dtype, DType::FromProto(output_spec.dtype()));
    output_dtypes.push_back(dtype);
    TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(output_spec.shape()));
    output_shapes.push_back(std::move(shape));
    if (output_spec.has_op_sharding()) {
      if (!output_hlo_shardings.has_value()) {
        output_hlo_shardings.emplace();
        output_hlo_shardings->reserve(metadata.output_specs_size());
      }
      TF_ASSIGN_OR_RETURN(auto hlo_sharding, xla::HloSharding::FromProto(
                                                 output_spec.op_sharding()));
      output_hlo_shardings->push_back(std::move(hlo_sharding));
    } else {
      // TODO(hyeontaek): Remove this branch once every output uses
      // `HloSharding` as `output_spec.has_op_sharding()` would be always true.
      if (output_hlo_shardings.has_value()) {
        return absl::InvalidArgumentError(
            "All outputs must use either HloSharding or ConcreteEvenSharding, "
            "not a mix of the two.");
      }
    }
    if (output_spec.memory_kind().empty()) {
      output_memory_kinds.push_back(MemoryKind());
    } else {
      output_memory_kinds.push_back(MemoryKind(output_spec.memory_kind()));
    }
    if (output_spec.has_layout()) {
      TF_ASSIGN_OR_RETURN(auto layout, Layout::FromProto(output_spec.layout()));
      output_layouts->push_back(
          llvm::cast<PjRtLayout>(layout.get())->pjrt_layout());
    } else {
      output_layouts->push_back(nullptr);
    }
  }

  CommonMetadata common_metadata;
  common_metadata.is_portable = metadata.portable();
  common_metadata.donatable_input_indices = std::move(donated_input_indices);
  common_metadata.output_dtypes = std::move(output_dtypes);
  common_metadata.output_shapes = std::move(output_shapes);
  common_metadata.output_hlo_shardings = std::move(output_hlo_shardings);
  common_metadata.output_memory_kinds = std::move(output_memory_kinds);
  common_metadata.output_layouts = std::move(output_layouts);
  return std::make_pair(std::move(common_metadata), serialized_pjrt_executable);
}

absl::StatusOr<std::string> PjRtExecutable::Serialize() const {
  DCHECK(this);
  return common_metadata_.Serialize(pjrt_executable_.get());
}

absl::StatusOr<LoadedExecutableRef> PjRtLoadedExecutable::Create(
    PjRtClient* client,
    std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
    std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
    DeviceListRef executable_devices,
    PjRtExecutable::CommonMetadata common_metadata) {
  VLOG(3) << "PjRtLoadedExecutable::Create";

  TF_ASSIGN_OR_RETURN(
      executable_devices,
      AdjustExecutableDevicesForPmap(client, pjrt_loaded_executable.get(),
                                     std::move(executable_devices)));

  return LoadedExecutableRef(new PjRtLoadedExecutable(
      client, std::move(pjrt_loaded_executable), std::move(executable_devices),
      std::move(loaded_host_callbacks), std::move(common_metadata)));
}

absl::StatusOr<LoadedExecutableRef> PjRtLoadedExecutable::Create(
    PjRtClient* client, mlir::ModuleOp module,
    xla::CompileOptions compile_options,
    std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
    DeviceListRef executable_devices) {
  VLOG(3) << "PjRtLoadedExecutable::Create";
  if (VLOG_IS_ON(3)) {
    module.dump();
  }
  VLOG(3) << compile_options.ToProto()->DebugString();

  const bool is_portable = compile_options.compile_portable_executable;
  // We have to do process the MLIR before the compile call, since the latter
  // will use the MLIR as scratch space, or possibly even deallocate it.
  TF_ASSIGN_OR_RETURN(std::vector<int> donatable_input_indices,
                      GetDonatableInputIndicesFromMlirModule(module));
  TF_ASSIGN_OR_RETURN(
      const std::vector<xla::Shape> mlir_module_output_xla_shapes,
      ResultShapesOfModule(module));
  TF_ASSIGN_OR_RETURN(const std::vector<xla::LayoutMode> output_layout_modes,
                      GetOutputLayoutModes(module));

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
      client->pjrt_client()->CompileAndLoad(std::move(module),
                                            std::move(compile_options)));

  TF_ASSIGN_OR_RETURN(
      executable_devices,
      AdjustExecutableDevicesForPmap(client, pjrt_loaded_executable.get(),
                                     std::move(executable_devices)));

  TF_ASSIGN_OR_RETURN(auto output_dtypes_and_shapes,
                      GetDTypesAndShapes(mlir_module_output_xla_shapes));
  std::vector<DType> output_dtypes = std::move(output_dtypes_and_shapes.first);
  std::vector<Shape> output_shapes = std::move(output_dtypes_and_shapes.second);
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<xla::HloSharding>> output_hlo_shardings,
      GetHloShardings(pjrt_loaded_executable->GetOutputShardings(),
                      output_dtypes, /*is_output=*/true));
  TF_ASSIGN_OR_RETURN(
      std::vector<MemoryKind> output_memory_kinds,
      GetMemoryKinds(pjrt_loaded_executable->GetOutputMemoryKinds(),
                     output_dtypes));
  std::vector<ShardingRef> output_shardings =
      MakeShardings(output_shapes, output_hlo_shardings, output_memory_kinds,
                    executable_devices);
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
          output_layouts,
      GetLayouts(pjrt_loaded_executable->GetOutputLayouts(),
                 output_layout_modes));

  PjRtExecutable::CommonMetadata common_metadata;
  common_metadata.is_portable = is_portable;
  common_metadata.donatable_input_indices = std::move(donatable_input_indices);
  common_metadata.output_dtypes = std::move(output_dtypes);
  common_metadata.output_shapes = std::move(output_shapes);
  common_metadata.output_hlo_shardings = std::move(output_hlo_shardings);
  common_metadata.output_memory_kinds = std::move(output_memory_kinds);
  common_metadata.output_layouts = std::move(output_layouts);

  return LoadedExecutableRef(new PjRtLoadedExecutable(
      client, std::move(pjrt_loaded_executable), std::move(executable_devices),
      std::move(loaded_host_callbacks), std::move(common_metadata)));
}

PjRtLoadedExecutable::PjRtLoadedExecutable(
    PjRtClient* client,
    std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
    DeviceListRef devices,
    std::vector<tsl::RCReference<LoadedHostCallback>> all_loaded_host_callbacks,
    PjRtExecutable::CommonMetadata common_metadata)
    : client_(client),
      pjrt_loaded_executable_(std::move(pjrt_loaded_executable)),
      devices_(std::move(devices)),
      addressable_devices_(devices_->AddressableDeviceList()->devices()),
      all_loaded_host_callbacks_(
          std::make_shared<std::vector<tsl::RCReference<LoadedHostCallback>>>(
              std::move(all_loaded_host_callbacks))),
      host_send_recv_callbacks_(
          GatherHostSendAndRecvCallbacks(*all_loaded_host_callbacks_)),
      common_metadata_(std::move(common_metadata)),
      user_context_(UserContextScope::current()) {
  output_shardings_ = MakeShardings(
      common_metadata_.output_shapes, common_metadata_.output_hlo_shardings,
      common_metadata_.output_memory_kinds, devices_);
  pjrt_output_memory_kinds_.emplace_back();
  for (const MemoryKind& memory_kind : common_metadata_.output_memory_kinds) {
    pjrt_output_memory_kinds_.back().push_back(
        memory_kind.memory_kind().value_or(""));
  }
}

PjRtLoadedExecutable::~PjRtLoadedExecutable() = default;

absl::StatusOr<PjRtLoadedExecutable::ExecuteResult>
PjRtLoadedExecutable::Execute(absl::Span<ArrayRef> args,
                              const ExecuteOptions& options,
                              std::optional<DeviceListRef> devices) {
  DCHECK(this);
  // TODO(hyeontaek): Check input sharding consistency.

  // Convert an Array vector into 2-level PjRtBuffer vectors, optionally copying
  // to new devices.
  std::vector<std::vector<PjRtBuffer*>> argument_handles;
  std::vector<std::unique_ptr<PjRtBuffer>> owned_buffers;

  int num_computations;
  const bool portable_execution = devices.has_value();
  PjRtCompatibleDevice* portable_execution_device = nullptr;
  if (portable_execution) {
    if ((*devices)->size() != 1) {
      return InvalidArgument(
          "Only single-shard portable execution is supported");
    }
    num_computations = 1;
    portable_execution_device =
        static_cast<PjRtDevice*>((*devices)->devices().front());
  } else {
    if (devices_->devices().empty()) {
      return InvalidArgument("No devices provided for portable executable");
    }
    num_computations = addressable_devices_.size();
  }

  argument_handles.resize(num_computations);
  for (int i = 0; i < num_computations; ++i) {
    argument_handles[i].reserve(args.size());
  }
  for (int i = 0; i < args.size(); ++i) {
    auto* pjrt_array =
        llvm::dyn_cast_or_null<PjRtCompatibleArray>(args[i].get());
    if (!pjrt_array) {
      return InvalidArgument(
          "Only PjRtCompatibleArray is supported, but argument %d is %s", i,
          pjrt_array->DebugString());
    }
    int j = 0;
    // TODO(hyeontaek): Check pjrt_array->pjrt_buffers().size() ==
    // num_computations
    for (const auto& pjrt_buffer : pjrt_array->pjrt_buffers()) {
      argument_handles[j].push_back(pjrt_buffer.get());
      ++j;
    }
  }

  xla::ExecuteOptions opts;
  opts.launch_id = options.launch_id;
  opts.use_major_to_minor_data_layout_for_callbacks = true;
  opts.non_donatable_input_indices = options.non_donatable_input_indices;
  opts.execution_stream_id = options.execution_stream_id;
  absl::StatusOr<absl::flat_hash_map<int, IncarnationId>> incarnations =
      client()->Incarnations();
  if (incarnations.ok()) {
    opts.incarnations = *std::move(incarnations);
  } else {
    VLOG(3) << "Unable to get incarnations: " << incarnations.status();
  }

  if (options.custom_options.has_value()) {
    auto call_location = options.custom_options->Get<std::string>(
        std::string(xla::ifrt::PjRtCompatibleLoadedExecutable::kCallLocation));
    if (call_location.ok()) {
      opts.call_location = *call_location;
    }
  }

  auto context = std::make_unique<xla::ExecuteContext>();
  auto platform_id = pjrt_loaded_executable_->client()->platform_id();
  auto ffi_callbacks = std::make_unique<xla::FfiLoadedHostCallbacks>();
  auto callbacks = std::make_unique<std::vector<void*>>();
  // Forward callbacks via FFI's ExecutionContext for CPU/GPU platforms only.
  if (platform_id == CpuId() || platform_id == CudaId() ||
      platform_id == RocmId() || platform_id == SyclId()) {
    for (const auto& loaded_host_callback : *all_loaded_host_callbacks_) {
      auto* ffi_loaded_host_callback =
          llvm::dyn_cast<PjRtFfiLoadedHostCallback>(loaded_host_callback.get());
      if (ffi_loaded_host_callback != nullptr) {
        void* callback = ffi_loaded_host_callback->callable();
        callbacks->push_back(callback);
      }
    }
    // NOTE(dsuo): For now, check that either all or none of the host callbacks
    // are FFI callbacks. Otherwise, we have an error.
    // TODO(b/406585850): Improve how we determine when loaded host callbacks
    // are forwarded to ffi::ExecutionContext.
    if (!callbacks->empty() &&
        callbacks->size() != all_loaded_host_callbacks_->size()) {
      return InvalidArgument(
          "ifrt::LoadedHostCallbacks must either be all "
          "ifrt::PjRtFfiLoadedHostCallback or none.");
    }
    ffi_callbacks->callbacks = callbacks->data();
    ffi_callbacks->num_callbacks = callbacks->size();
    ffi::TypeRegistry::TypeId type_id(FfiLoadedHostCallbacks::id.type_id);
    CHECK_OK(context->ffi_context().Insert(type_id, ffi_callbacks.get()));
    opts.context = context.get();
  }

  // When using host callbacks on CPU, we need to use synchronous dispatch to
  // avoid deadlocks with reentrant callbacks. Note that this option only
  // affects the CPU runtime.
  if (!all_loaded_host_callbacks_->empty()) {
    opts.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;
  }

  std::unique_ptr<HostCallbackStates> host_callback_states;
  if (!host_send_recv_callbacks_.empty()) {
    host_callback_states = std::make_unique<HostCallbackStates>();
    for (int i = 0; i < num_computations; ++i) {
      auto& contexts = host_callback_states->contexts.emplace_back();
      auto& send_callbacks =
          host_callback_states->send_callbacks.emplace_back();
      auto& recv_callbacks =
          host_callback_states->recv_callbacks.emplace_back();

      for (const auto& host_send_recv_callback : host_send_recv_callbacks_) {
        contexts.push_back(CreateHostCallbackStateAndAppendSendRecvCallbacks(
            host_send_recv_callback->host_callback(),
            /*host_memory_for_device_manager=*/nullptr, send_callbacks,
            recv_callbacks, opts.use_major_to_minor_data_layout_for_callbacks));
      }
    }
    opts.send_callbacks = host_callback_states->send_callbacks;
    opts.recv_callbacks = host_callback_states->recv_callbacks;
  }

  // Execute the computation.
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_outputs;
  tsl::Future<> status;
  if (portable_execution) {
    std::optional<tsl::Future<>> returned_pjrt_future;
    TF_RET_CHECK(portable_execution_device->IsAddressable());
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<PjRtBuffer>> single_device_pjrt_results,
        pjrt_loaded_executable_->ExecutePortable(
            argument_handles.front(), portable_execution_device->pjrt_device(),
            opts, returned_pjrt_future,
            /*fill_future=*/true));

    pjrt_outputs.push_back(std::move(single_device_pjrt_results));
    status = *std::move(returned_pjrt_future);
  } else {
    std::optional<std::vector<tsl::Future<>>> returned_pjrt_futures;
    returned_pjrt_futures.emplace();

    TF_ASSIGN_OR_RETURN(
        pjrt_outputs, pjrt_loaded_executable_->Execute(argument_handles, opts,
                                                       returned_pjrt_futures));

    status = JoinFutures(absl::MakeSpan(*returned_pjrt_futures));
  }

  if (!all_loaded_host_callbacks_->empty()) {
    // For host callbacks to work, returned futures must be supported so that we
    // can use the futures to extend the lifetime of the host callbacks until
    // the execution finishes.
    status.OnReady([all_loaded_host_callbacks = all_loaded_host_callbacks_,
                    host_callback_states = std::move(host_callback_states),
                    context = std::move(context),
                    ffi_callbacks = std::move(ffi_callbacks),
                    callbacks = std::move(callbacks)](absl::Status) mutable {
      all_loaded_host_callbacks.reset();
    });
  }

  // Convert 2-level PjRtBuffer vectors into an Array vector.
  std::vector<ArrayRef> outputs;
  // TODO(hyeontaek): Check output dtype/shape consistency with the actual
  // output.
  if (pjrt_outputs.size() != num_computations) {
    return FailedPrecondition(
        "Unexpected number of computations in outputs: %d vs. %d",
        pjrt_outputs.size(), num_computations);
  }
  const int num_outputs = pjrt_outputs.empty()
                              ? common_metadata_.output_dtypes.size()
                              : pjrt_outputs.front().size();
  if (num_outputs != common_metadata_.output_dtypes.size()) {
    return FailedPrecondition("Unexpected number of outputs: %d vs. %d",
                              num_outputs,
                              common_metadata_.output_dtypes.size());
  }
  outputs.reserve(num_outputs);
  // Single-device Shardings for portable execution. Outputs with the same
  // memory_kind shares the same Sharding object.
  absl::flat_hash_map<MemoryKind, ShardingRef> single_device_shardings;

  std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
  layouts.reserve(num_outputs);
  if (common_metadata_.output_layouts.has_value()) {
    // TODO(hyeontaek): Once we can get `output_layouts_` reliably, only keep
    // this path.
    layouts = *common_metadata_.output_layouts;
  } else if (!pjrt_outputs.empty()) {
    for (int i = 0; i < num_outputs; ++i) {
      auto layout =
          common_metadata_.output_dtypes[i].kind() == xla::ifrt::DType::kToken
              ? std::make_shared<xla::PjRtLayout>(xla::Layout())
              : pjrt_outputs.front()[i]->layout();
      layouts.push_back(std::move(layout));
    }
  } else {
    auto maybe_layouts = GetOutputLayouts();
    // An unimplemented error is converted into all-default layouts.
    if (absl::IsUnimplemented(maybe_layouts.status())) {
      layouts.resize(/*size=*/num_outputs, /*value=*/nullptr);
    } else {
      TF_RETURN_IF_ERROR(maybe_layouts.status());
      layouts = *std::move(maybe_layouts);
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    PjRtArray::PjRtBuffers buffers;
    buffers.reserve(num_computations);
    const MemoryKind dst_memory_kind = output_shardings_[i]->memory_kind();
    const MemoryKind canonical_dst_memory_kind = CanonicalizeMemoryKind(
        dst_memory_kind, output_shardings_[i]->devices()->devices().front());

    for (int j = 0; j < num_computations; ++j) {
      if (j > 0) {
        if (auto memory_kind =
                MakeMemoryKindFromPjRtBuffer(pjrt_outputs[j][i].get());
            canonical_dst_memory_kind !=
            CanonicalizeMemoryKindWithPjRtDevice(
                memory_kind, pjrt_outputs[j][i]->device())) {
          return FailedPrecondition(
              "Memory kind mismatch. Got sharding with memory kind '%v' and "
              "buffer with memory_kind '%v'",
              dst_memory_kind, memory_kind);
        }
      }
      buffers.push_back(
          std::shared_ptr<PjRtBuffer>(pjrt_outputs[j][i].release()));
    }
    std::optional<ShardingRef> sharding;
    if (portable_execution) {
      if (auto it = single_device_shardings.find(dst_memory_kind);
          it == single_device_shardings.end()) {
        sharding = single_device_shardings
                       .insert({dst_memory_kind, SingleDeviceSharding::Create(
                                                     portable_execution_device,
                                                     dst_memory_kind)})
                       .first->second;
      } else {
        sharding = it->second;
      }
    } else {
      sharding = output_shardings_[i];
    }
    outputs.push_back(*PjRtArray::Create(
        client_, common_metadata_.output_dtypes[i],
        common_metadata_.output_shapes[i], *std::move(sharding),
        std::move(buffers), std::move(layouts[i])));
  }

  ExecuteResult result;
  if (options.fill_status) {
    result.status = status;
  }
  result.outputs = std::move(outputs);
  return result;
}

absl::StatusOr<std::optional<std::string>> PjRtLoadedExecutable::Fingerprint()
    const {
  DCHECK(this);
  absl::StatusOr<std::string> fingerprint =
      pjrt_loaded_executable_->FingerprintExecutable();
  if (fingerprint.ok()) {
    return {fingerprint.value()};
  }
  if (fingerprint.status().code() == absl::StatusCode::kUnimplemented) {
    // Return nullopt in case of unimplemented error.
    return std::nullopt;
  }
  return fingerprint.status();
}

absl::StatusOr<std::shared_ptr<const ExecutableVersion>>
PjRtLoadedExecutable::executable_version() const {
  DCHECK(this);
  // PjRt-IFRT currently does not track XLA executable versions.
  return std::make_shared<XlaExecutableVersion>();
}

absl::StatusOr<std::string> PjRtLoadedExecutable::Serialize() const {
  DCHECK(this);
  return common_metadata_.Serialize(pjrt_loaded_executable_->GetExecutable());
}

}  // namespace ifrt
}  // namespace xla
