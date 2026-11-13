/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_program.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/extract_callback.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/mlir/tfrt/utils/export.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/layout.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/dump.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_device_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

bool IsSingleDevice(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  return compile_metadata.num_replicas() == 1 &&
         compile_metadata.num_cores_per_replica() == 1;
}

absl::StatusOr<xla::DeviceAssignment> GetRuntimeXlaDeviceAssignment(
    const xla::ifrt::DeviceListRef& device_list, int num_replicas,
    int num_cores_per_replica) {
  const int num_devices = num_replicas * num_cores_per_replica;
  const absl::Span<xla::ifrt::Device* const> devices = device_list->devices();
  if (devices.size() != num_devices) {
    return absl::InternalError(
        absl::StrCat("Device assignment has ", devices.size(),
                     " devices, but expected ", num_devices));
  }
  xla::DeviceAssignment da(num_replicas, num_cores_per_replica);
  int device_index = 0;
  for (int replica_idx = 0; replica_idx < num_replicas; replica_idx++) {
    for (int core_idx = 0; core_idx < num_cores_per_replica;
         core_idx++, device_index++) {
      da(replica_idx, core_idx) = devices[device_index]->Id().value();
      VLOG(3) << "Added IFRT device id: " << da(replica_idx, core_idx);
    }
  }
  return da;
}

static constexpr absl::string_view kDeviceAssignmentAttr = "device_assignment";
static constexpr absl::string_view kEntryFuncName = "main";

absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedDevices(
    mlir::ModuleOp module, const xla::ifrt::Client& ifrt_client,
    int num_replicas, int num_cores_per_replica) {
  auto op = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!op) {
    return absl::InternalError("Could not find entry function in MLIR Module.");
  }

  auto device_assignment_attr =
      op->getAttrOfType<mlir::ArrayAttr>(kDeviceAssignmentAttr);
  std::optional<std::vector<int>> device_assignment_attr_val;

  if (device_assignment_attr && !device_assignment_attr.getValue().empty()) {
    std::vector<int> coords;
    coords.reserve(num_replicas * num_cores_per_replica);
    for (auto coord_attr : device_assignment_attr.getValue()) {
      auto coord_attr_val = mlir::dyn_cast<mlir::IntegerAttr>(coord_attr);
      if (!coord_attr_val) {
        return absl::InternalError(
            llvm::formatv("Device assignment attribute is not an integer: {0}",
                          device_assignment_attr)
                .str());
      }
      coords.push_back(coord_attr_val.getInt());
    }
    device_assignment_attr_val = std::move(coords);
  }
  return GetAssignedIfrtDevices(ifrt_client, num_replicas,
                                num_cores_per_replica,
                                device_assignment_attr_val);
}

absl::StatusOr<
    absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>>
GetHostCallbackModulesAndRemoveHostFuncs(mlir::ModuleOp module) {
  absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>
      host_callback_modules;
  llvm::DenseSet<mlir::TF::XlaHostComputeOp> xla_host_compute_ops;
  module->walk(
      [&](mlir::TF::XlaHostComputeOp op) { xla_host_compute_ops.insert(op); });
  for (auto& op : xla_host_compute_ops) {
    TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> host_callback_module,
                        ExtractCallbackModule(module, op.getKey().str()));
    auto [_, inserted] = host_callback_modules.insert(
        {op.getKey().str(), std::move(host_callback_module)});
    if (!inserted) {
      return absl::FailedPreconditionError(
          absl::StrCat("Duplicate host callback key: ", op.getKey().str()));
    }
    auto func = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
        module, op.getKeyAttr());
    if (!func) {
      return absl::InternalError(
          absl::StrCat("symbol not found: ", op.getKey().str()));
    }
    func->erase();
  }
  return host_callback_modules;
}

// Retrieves static shapes for the inputs. For arguments specified in
// `static_shape_arg_map`, it extracts the actual shapes from the
// corresponding input tensors.
absl::flat_hash_map<size_t, size_t> GetStaticShapeArgMap(
    mlir::ModuleOp module, absl::string_view signature_name) {
  absl::flat_hash_map<size_t, size_t> static_shape_arg_map;
  auto entry_func_op = module.lookupSymbol<mlir::func::FuncOp>(signature_name);
  if (!entry_func_op) {
    entry_func_op = module.lookupSymbol<mlir::func::FuncOp>("main");
  }
  if (entry_func_op) {
    for (size_t i = 0; i < entry_func_op.getNumArguments(); ++i) {
      if (auto arg_attr = entry_func_op.getArgAttrOfType<mlir::IntegerAttr>(
              i, "tf._static_shape_arg_idx")) {
        static_shape_arg_map[i] = arg_attr.getInt();
      }
    }
  }
  return static_shape_arg_map;
}

// If the input tensor's rank differs from `reshaped_shape` (e.g., due to
// compiler flattening), attempt to reshape the input to match the rank of
// `reshaped_shape`. The whole shape does not have to match, because the input
// is expected to be a prefix of the reshaped shape in case of static shape.
// Currently, only flattening to 1D is supported since it suffices the only use
// case we need to support.
}  // namespace

absl::StatusOr<std::vector<tensorflow::FunctionDef>> BuildFunctionDef(
    mlir::ModuleOp module) {
  std::vector<tensorflow::FunctionDef> function_defs;

  // Sets `export_tf_original_func_name` to false so that ExportFunctionDef
  // does not rename the function back to the original function name. This
  // allows calling the function by the function name in the MLIR module.
  TF_RETURN_IF_ERROR(ExportFunctionDefs(
      module,
      [&](tensorflow::FunctionDef function_def) {
        function_defs.push_back(std::move(function_def));
        return absl::OkStatus();
      },
      /*export_tf_original_func_name=*/false));

  return function_defs;
}

// Host callback info for one host callback.
struct HostCallbackBuilderInfo {
  tensorflow::tf2xla::HostTransferMetadata device_to_host;
  tensorflow::tf2xla::HostTransferMetadata host_to_device;
};

absl::StatusOr<absl::flat_hash_map<std::string, HostCallbackBuilderInfo>>
GroupHostCallbackByKey(
    const tf2xla::HostComputeMetadata& host_compute_metadata) {
  absl::flat_hash_map<std::string, HostCallbackBuilderInfo> host_callbacks;

  for (const auto& device_to_host : host_compute_metadata.device_to_host()) {
    auto& host_callback = host_callbacks[device_to_host.key()];
    host_callback.device_to_host = device_to_host;
  }
  for (const auto& host_to_device : host_compute_metadata.host_to_device()) {
    auto& host_callback = host_callbacks[host_to_device.key()];
    host_callback.host_to_device = host_to_device;
  }
  return host_callbacks;
}

// TODO: shape propagation in module
absl::StatusOr<xla::HostCallback> BuildHostCallback(
    absl::string_view key, const HostCallbackBuilderInfo& builder_info,
    mlir::ModuleOp callback_module, tensorflow::DeviceMgr* device_mgr,
    std::vector<std::unique_ptr<TfHostCallback>>& tf_host_callbacks) {
  VLOG(2) << "BuildHostCallback for key: " << key;

  DCHECK(device_mgr);
  xla::HostCallback host_callback;
  std::vector<DtypeAndShape> operand_type_and_shapes;
  std::vector<DtypeAndShape> result_type_and_shapes;

  auto to_xla_shape = [](tensorflow::DataType data_type,
                         const tensorflow::TensorShapeProto& shape)
      -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_ASSIGN_OR_RETURN(tensorflow::TensorShape tensor_shape,
                        tensorflow::TensorShape::BuildTensorShape(shape));

    if (absl::Status status = tensorflow::TensorShapeToXLAShape(
            data_type, tensor_shape, &xla_shape);
        status.ok()) {
      return xla_shape;
    } else {
      return status;
    }
  };

  operand_type_and_shapes.reserve(builder_info.device_to_host.metadata_size());
  result_type_and_shapes.reserve(builder_info.host_to_device.metadata_size());
  for (const auto& metadata : builder_info.device_to_host.metadata()) {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        to_xla_shape(metadata.type(), metadata.shape()));
    uint16_t channel_id = static_cast<uint16_t>(metadata.channel_id());
    VLOG(2) << "Channel id: " << channel_id;
    host_callback.operands.push_back(
        {.channel_id = channel_id, .shape = shape});
    operand_type_and_shapes.push_back(
        DtypeAndShape{.dtype = metadata.type(), .shape = metadata.shape()});
  }

  for (const auto& metadata : builder_info.host_to_device.metadata()) {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        to_xla_shape(metadata.type(), metadata.shape()));
    uint16_t channel_id = static_cast<uint16_t>(metadata.channel_id());
    VLOG(2) << "Channel id: " << channel_id;
    host_callback.results.push_back(
        {.channel_id = channel_id, .shape = std::move(shape)});
    result_type_and_shapes.push_back(
        DtypeAndShape{.dtype = metadata.type(), .shape = metadata.shape()});
  }

  TF_ASSIGN_OR_RETURN(std::vector<tensorflow::FunctionDef> function_defs,
                      BuildFunctionDef(callback_module));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfHostCallback> tf_host_callback,
      TfHostCallback::Create(function_defs, key, operand_type_and_shapes,
                             result_type_and_shapes, device_mgr));

  host_callback.callback = [tf_host_callback = tf_host_callback.get()](
                               void** output, void** input) {
    return tf_host_callback->Call(input, output);
  };

  tf_host_callbacks.push_back(std::move(tf_host_callback));
  return host_callback;
}

absl::StatusOr<std::vector<xla::HostCallback>> BuildHostCallbacks(
    const tf2xla::HostComputeMetadata& host_compute_metadata,
    absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>
        host_callback_modules,
    tensorflow::DeviceMgr* device_mgr,
    std::vector<std::unique_ptr<TfHostCallback>>& tf_host_callbacks) {
  TF_ASSIGN_OR_RETURN(auto host_callback_maps,
                      GroupHostCallbackByKey(host_compute_metadata));

  std::vector<xla::HostCallback> host_callbacks;
  host_callbacks.reserve(host_callback_maps.size());
  for (const auto& [entry_function, builder_info] : host_callback_maps) {
    auto host_callback_module_it = host_callback_modules.find(entry_function);
    if (host_callback_module_it == host_callback_modules.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Host callback module not found for key: ", entry_function));
    }
    TF_ASSIGN_OR_RETURN(auto host_callback,
                        BuildHostCallback(entry_function, builder_info,
                                          *host_callback_module_it->second,
                                          device_mgr, tf_host_callbacks));
    host_callbacks.push_back(std::move(host_callback));
  }

  return host_callbacks;
}

absl::Status EncodeLayout(absl::Span<const xla::Shape> xla_input_shapes,
                          mlir::ModuleOp module) {
  auto op = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!op) {
    return absl::InternalError("Could not find entry function in MLIR Module.");
  }

  if (op.getNumArguments() != xla_input_shapes.size()) {
    return absl::InternalError(
        absl::StrCat("Argument count mismatch: ", op.getNumArguments(), " vs ",
                     xla_input_shapes.size()));
  }

  mlir::OpBuilder builder(op.getContext());
  for (int i = 0; i < xla_input_shapes.size(); ++i) {
    const auto& shape = xla_input_shapes[i];
    if (shape.has_layout()) {
      auto attr = builder.getStringAttr(shape.layout().ToString());
      op.setArgAttr(i, "mhlo.layout_mode", attr);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<IfrtProgram>> IfrtProgram::Create(
    absl::string_view model_name, absl::string_view signature_name,
    mlir::OwningOpRef<mlir::ModuleOp> module,
    std::shared_ptr<xla::ifrt::Client> client,
    tensorflow::DeviceMgr* device_mgr,
    tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    IfrtServingCoreSelector* ifrt_serving_core_selector,
    std::variant<tsl::protobuf::Message*,
                 xla::CompileOptions::EnvironmentOptionOverrides>
        compilation_env_or_overrides,
    TfToHloCompiler* tf_to_hlo_compiler,
    IfrtPersistentCompilationCache* persistent_compilation_cache,
    bool use_undonatable_buffer_converter) {
  TF_ASSIGN_OR_RETURN(
      tensorflow::tpu::TPUCompileMetadataProto original_compile_metadata,
      GetCompileMetadata(*module, *client));

  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::Device*> assigned_devices,
      GetAssignedDevices(*module, *client,
                         original_compile_metadata.num_replicas(),
                         original_compile_metadata.num_cores_per_replica()));

  absl::flat_hash_map<size_t, size_t> static_shape_arg_map =
      GetStaticShapeArgMap(*module, signature_name);

  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      client->MakeDeviceList(assigned_devices));
  return absl::WrapUnique(new IfrtProgram(
      model_name, signature_name, std::move(module), std::move(client),
      device_mgr, std::move(shape_representation_fn),
      ifrt_serving_core_selector, std::move(original_compile_metadata),
      std::move(device_list), std::move(static_shape_arg_map),
      compilation_env_or_overrides, tf_to_hlo_compiler,
      persistent_compilation_cache, use_undonatable_buffer_converter));
}

absl::Status IfrtProgram::PopulateInvariantMetadata(
    const Tf2HloResult& tf2hlo_result,
    xla::ifrt::LoadedExecutableRef ifrt_executable,
    std::vector<std::unique_ptr<TfHostCallback>> host_callbacks,
    const xla::ifrt::Topology* topology, CompiledExecutable& compiled) {
  compiled.ifrt_input_dtypes.reserve(
      tf2hlo_result.compile_metadata.args().size());
  compiled.reshaped_input_tensors.reserve(
      tf2hlo_result.compile_metadata.args().size());
  compiled.ifrt_input_shapes.reserve(
      tf2hlo_result.compile_metadata.args().size());
  compiled.xla_input_shapes.reserve(
      tf2hlo_result.compile_metadata.args().size());
  compiled.xla_input_layouts.reserve(
      tf2hlo_result.compile_metadata.args().size());
  compiled.byte_strides.reserve(tf2hlo_result.compile_metadata.args().size());

  TF_ASSIGN_OR_RETURN(auto parameter_layouts,
                      ifrt_executable->GetParameterLayouts());

  const xla::PjRtTopologyDescription* pjrt_topology = nullptr;
  if (!tf2hlo_result.xla_input_shapes.empty() && topology) {
    pjrt_topology = topology->description().get();
  }

  for (int i = 0; i < tf2hlo_result.compile_metadata.args().size(); ++i) {
    const auto& arg = tf2hlo_result.compile_metadata.args(i);
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, ToIfrtDType(arg.dtype()));
    compiled.ifrt_input_dtypes.push_back(ifrt_dtype);
    TF_ASSIGN_OR_RETURN(auto reshaped_tensor,
                        tensorflow::TensorShape::BuildTensorShape(arg.shape()));

    xla::ifrt::Shape ifrt_shape = ToIfrtShape(reshaped_tensor);
    compiled.ifrt_input_shapes.push_back(
        std::make_shared<xla::ifrt::Shape>(std::move(ifrt_shape)));
    compiled.reshaped_input_tensors.push_back(std::move(reshaped_tensor));

    if (!tf2hlo_result.xla_input_shapes.empty()) {
      auto shape_ptr =
          std::make_shared<xla::Shape>(tf2hlo_result.xla_input_shapes[i]);
      if (pjrt_topology) {
        // Canonicalize the `xla::Shape`. For example, on TPUs, this could set
        // the layout or tile dimensions depending on the memory space.
        const xla::Shape& request_shape = tf2hlo_result.xla_input_shapes[i];
        if (auto shape = pjrt_topology->MakeCanonicalShapeForMemorySpace(
                pjrt_topology->GetDefaultMemorySpaceKindId(), request_shape,
                request_shape.has_layout() ? &request_shape.layout() : nullptr);
            shape.ok()) {
          shape_ptr = std::make_shared<xla::Shape>(*std::move(shape));
        }
      }
      compiled.xla_input_shapes.push_back(std::move(shape_ptr));
    } else {
      compiled.xla_input_shapes.push_back(nullptr);
    }

    compiled.byte_strides.push_back(
        GetByteStrides(arg.dtype(), compiled.reshaped_input_tensors.back())
            .value_or(absl::InlinedVector<int64_t, 4>()));

    // Create device shape with backend-optimized layout. The layouts from
    // `GetParameterLayouts()` are the physical formats expected by the
    // compiled program, which may include hardware-specific tiling or padding.
    compiled.xla_input_layouts.push_back(
        xla::ifrt::PjRtLayout::Create(parameter_layouts[i]));
  }

  if (use_undonatable_buffer_converter_ && ifrt_executable != nullptr) {
    absl::StatusOr<std::vector<bool>> input_may_be_donated =
        GetInputDonationMask(*ifrt_executable,
                             compiled.xla_input_layouts.size());
    if (input_may_be_donated.ok()) {
      compiled.input_may_be_donated = *std::move(input_may_be_donated);
    } else {
      // Missing donation info must disable the undonatable-buffer conversion,
      // never enable it: leave the mask empty so all buffers stay donatable.
      LOG(WARNING) << "Failed to determine donatable inputs for program "
                   << model_name_ << "/" << signature_name_
                   << "; loaded variables keep donatable "
                   << "buffers: " << input_may_be_donated.status();
    }
  }
  compiled.ifrt_executable = std::move(ifrt_executable);
  compiled.compile_metadata = std::move(tf2hlo_result.compile_metadata);
  compiled.host_callbacks = std::move(host_callbacks);

  compiled.arg_hlo_shardings.reserve(compiled.compile_metadata.args().size());

  for (const auto& arg : compiled.compile_metadata.args()) {
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(arg.sharding()));
    compiled.arg_hlo_shardings.push_back(hlo_sharding);
    TF_ASSIGN_OR_RETURN(
        auto ifrt_sharding,
        ToIfrtSharding(*ifrt_client_, hlo_sharding, assigned_device_list_));
    compiled.arg_ifrt_shardings.push_back(std::move(ifrt_sharding));
  }

  if (UsePortableExecution()) {
    // For core selection, the device is selected at runtime. We pre-calculate
    // the sharding for each addressable device to avoid doing it on the
    // critical path. The map is keyed by device ID.
    for (xla::ifrt::Device* device : ifrt_client_->addressable_devices()) {
      compiled.portable_single_device_shardings.emplace(
          device->Id(), xla::ifrt::SingleDeviceSharding::Create(
                            device, xla::ifrt::MemoryKind()));
    }
  }

  compiled.retval_hlo_shardings.reserve(
      compiled.compile_metadata.retvals().size());
  for (const auto& retvals : compiled.compile_metadata.retvals()) {
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(retvals.sharding()));
    compiled.retval_hlo_shardings.push_back(hlo_sharding);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<const IfrtProgram::CompiledExecutable>>
IfrtProgram::CreateExecutableSynchronously(
    mlir::OwningOpRef<mlir::ModuleOp> module_copy,
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices) {
  TF_ASSIGN_OR_RETURN(auto host_callback_modules,
                      GetHostCallbackModulesAndRemoveHostFuncs(*module_copy));
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("module_for_bridge_phase2", *module_copy);
  }

  Tf2HloArg tf2hlo_arg{
      .module = module_copy.get(),
      .input_dtypes_and_shapes = std::vector<DtypeAndShape>(
          dtypes_and_shapes.begin(), dtypes_and_shapes.end()),
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = signature_name(),
      .compile_metadata = compile_metadata,
      .shape_representation_fn = shape_representation_fn_,
      .platform_name = ifrt_client_->platform_name(),
  };

  // Only get device topology for clients that implement GetTopologyForDevices.
  if (tf2hlo_arg.platform_name != xla::CudaName() &&
      !absl::StartsWith(ifrt_client_->runtime_type(), "proxy/")) {
    TF_ASSIGN_OR_RETURN(
        tf2hlo_arg.topology,
        ifrt_client_->GetTopologyForDevices(assigned_device_list_));
  }

  TF_ASSIGN_OR_RETURN(Tf2HloResult tf2hlo_result,
                      persistent_compilation_cache_->LookupTf2HloResultOrCreate(
                          tf2hlo_arg, tf_to_hlo_compiler_));
  if (VLOG_IS_ON(1)) {
    xla::DumpHloModuleProtoIfEnabled(tf2hlo_result.hlo_module_proto,
                                     "before_ifrt_serialization");
  }

  if (xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(
          ifrt_client_.get()) == nullptr) {
    if (tf2hlo_result.hlo_module_proto.has_host_program_shape() &&
        tf2hlo_result.hlo_module_proto.host_program_shape().has_result()) {
      TF_ASSIGN_OR_RETURN(
          xla::Shape result_shape,
          xla::Shape::FromProto(
              tf2hlo_result.hlo_module_proto.host_program_shape().result()));
      if (result_shape.is_dynamic()) {
        return absl::UnimplementedError(
            "Dynamic output shape is not supported for non-PJRT IFRT client");
      }
    }
  }

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_hlo_module,
      ::xla::ConvertHloToStablehloWithOptions(
          *module_copy->getContext(), &tf2hlo_result.hlo_module_proto,
          /*import_all_computations=*/false));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_bridge_phase2",
                                 mlir_hlo_module.get());
  }

  if (!tf2hlo_result.xla_input_shapes.empty()) {
    TF_RETURN_IF_ERROR(
        EncodeLayout(tf2hlo_result.xla_input_shapes, mlir_hlo_module.get()));
  }

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_layout_encoding_layout_mode",
                                 mlir_hlo_module.get());
  }

  const int num_replicas = tf2hlo_result.compile_metadata.num_replicas();
  const int num_partitions =
      tf2hlo_result.compile_metadata.num_cores_per_replica();

  VLOG(2) << " Number of replcas is " << num_replicas
          << " and num_partitions is " << num_partitions;

  if (num_replicas > 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only support single replica, but replica number is ",
                     num_replicas, " and num_partitions is ", num_partitions));
  }

  xla::CompileOptions xla_compile_options;

  if (std::holds_alternative<tsl::protobuf::Message*>(
          compilation_env_or_overrides_)) {
    tsl::protobuf::Message* compilation_environment_proto_ =
        std::get<tsl::protobuf::Message*>(compilation_env_or_overrides_);
    if (compilation_environment_proto_) {
      tsl::protobuf::Message* comp_env_copy =
          compilation_environment_proto_->New();
      comp_env_copy->CopyFrom(*compilation_environment_proto_);
      TF_RETURN_IF_ERROR(
          xla_compile_options.executable_build_options.mutable_comp_envs()
              ->AddEnv(
                  absl::WrapUnique<tsl::protobuf::Message>(comp_env_copy)));
    }
  } else if (std::holds_alternative<
                 xla::CompileOptions::EnvironmentOptionOverrides>(
                 compilation_env_or_overrides_)) {
    xla_compile_options.env_option_overrides =
        std::get<xla::CompileOptions::EnvironmentOptionOverrides>(
            compilation_env_or_overrides_);
  } else {
    return absl::NotFoundError(
        "Either compilation_environment_proto or env_option_overrides is "
        "expected.");
  }

  xla_compile_options.executable_build_options.set_num_replicas(num_replicas);
  xla_compile_options.executable_build_options.set_num_partitions(
      num_partitions);

  xla_compile_options.executable_build_options.set_use_spmd_partitioning(
      original_compile_metadata_.use_spmd_for_xla_partitioning());
  xla_compile_options.executable_build_options.set_use_shardy_partitioner(
      compile_metadata.use_shardy_partitioner());
  xla_compile_options.parameter_is_tupled_arguments = false;
  // Use portable execution for single device + core selection.
  if (UsePortableExecution()) {
    xla_compile_options.compile_portable_executable = true;
  } else {
    TF_ASSIGN_OR_RETURN(
        xla::DeviceAssignment da,
        GetRuntimeXlaDeviceAssignment(assigned_device_list_, num_replicas,
                                      num_partitions));
    VLOG(2) << "Device assignment :" << da.ToString();
    xla_compile_options.executable_build_options.set_device_assignment(da);
  }

  std::vector<std::unique_ptr<TfHostCallback>> tf_host_callbacks;
  TF_ASSIGN_OR_RETURN(auto host_callbacks,
                      BuildHostCallbacks(tf2hlo_result.host_compute_metadata,
                                         std::move(host_callback_modules),
                                         device_mgr_, tf_host_callbacks));

  std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
      loaded_host_callbacks;
  loaded_host_callbacks.reserve(host_callbacks.size());
  for (const auto& host_callback : host_callbacks) {
    loaded_host_callbacks.push_back(
        tsl::MakeRef<xla::ifrt::PjRtHostSendAndRecvLoadedHostCallback>(
            ifrt_client_.get(),
            std::make_unique<xla::HostCallback>(host_callback)));
  }
  auto hlo_program =
      std::make_unique<xla::ifrt::HloProgram>(mlir_hlo_module.get());
  auto compiled = std::make_shared<CompiledExecutable>();

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutableRef ifrt_executable,
      persistent_compilation_cache_->LookupLoadedExecutableOrCreate(
          std::move(hlo_program), assigned_device_list_, xla_compile_options,
          loaded_host_callbacks, ifrt_client_.get(),
          [&](std::unique_ptr<xla::ifrt::Program> program,
              std::unique_ptr<xla::ifrt::CompileOptions> options)
              -> absl::StatusOr<xla::ifrt::LoadedExecutableRef> {
            return ifrt_client_->GetDefaultCompiler()
                ->CompileAndLoad(std::move(program), std::move(options))
                .Await();
          }));

  TF_RETURN_IF_ERROR(PopulateInvariantMetadata(
      tf2hlo_result, std::move(ifrt_executable), std::move(tf_host_callbacks),
      tf2hlo_arg.topology.get(), *compiled));

  return compiled;
}

absl::StatusOr<
    tsl::Future<std::shared_ptr<const IfrtProgram::CompiledExecutable>>>
IfrtProgram::GetOrCompile(absl::Span<const DtypeAndShape> dtypes_and_shapes,
                          absl::Span<const int> variable_arg_indices) {
  using CompiledFuture = tsl::Future<std::shared_ptr<const CompiledExecutable>>;
  tsl::Promise<std::shared_ptr<const CompiledExecutable>> promise;
  CompiledFuture future;
  mlir::OwningOpRef<mlir::ModuleOp> module_copy;
  {
    absl::MutexLock lock(compile_mutex_);

    const auto it = compiled_executables_.find(KeyView{dtypes_and_shapes});
    if (it != compiled_executables_.end()) {
      return it->second;
    }

    const bool xla_compilation_disabled =
        tf_to_hlo_compiler_->IsXlaCompilationDisabled();
    if (is_frozen_ || xla_compilation_disabled) {
      // Build a description of the requested (offending) input shapes.
      std::string requested_shapes_str;
      for (size_t i = 0; i < dtypes_and_shapes.size(); ++i) {
        absl::StrAppend(
            &requested_shapes_str, "[", i,
            "]: ", dtypes_and_shapes[i].GetShapeForCompilation().DebugString());
        if (i + 1 < dtypes_and_shapes.size()) {
          absl::StrAppend(&requested_shapes_str, ", ");
        }
      }
      // Build a description of all cached (already compiled) shape sets.
      std::string cached_shapes_str;
      int key_idx = 0;
      for (const auto& [key, unused_future] : compiled_executables_) {
        absl::StrAppend(&cached_shapes_str, "  compiled[", key_idx++, "]: {");
        for (size_t i = 0; i < key.input_shapes.size(); ++i) {
          absl::StrAppend(&cached_shapes_str,
                          key.input_shapes[i].DebugString());
          if (i + 1 < key.input_shapes.size()) {
            absl::StrAppend(&cached_shapes_str, ", ");
          }
        }
        absl::StrAppend(&cached_shapes_str, "}\n");
      }
      // When compilation is disabled, prepend the marker
      // `kXlaCompilationDisabledErrorMarker` so that callers (e.g. model
      // servers) can identify this as a compilation cache miss while
      // compilation is disabled. Note that the failure is intentionally not
      // cached in `compiled_executables_`, so a retry with compilation enabled
      // can still compile the executable if it is not frozen.
      return absl::FailedPreconditionError(absl::StrCat(
          xla_compilation_disabled
              ? absl::StrCat(kXlaCompilationDisabledErrorMarker, ": ")
              : "",
          "Cannot compile for new input shapes. Either the executable is "
          "already frozen: ",
          is_frozen_,
          " or XLA compilation disabled by ScopedTpuCompileDisabler: ",
          xla_compilation_disabled, ". Requested input shapes: {",
          requested_shapes_str, "}. Number of already compiled shape sets: ",
          compiled_executables_.size(),
          cached_shapes_str.empty()
              ? ""
              : absl::StrCat(". Already compiled:\n", cached_shapes_str)));
    }

    // Only create promise and future when cache missed.
    std::tie(promise, future) =
        tsl::MakePromise<std::shared_ptr<const CompiledExecutable>>();

    std::vector<tensorflow::TensorShape> input_shapes;
    input_shapes.reserve(dtypes_and_shapes.size());
    for (const auto& dtype_and_shape : dtypes_and_shapes) {
      input_shapes.push_back(dtype_and_shape.GetShapeForCompilation());
    }
    Key key = {.input_shapes = std::move(input_shapes)};

    compiled_executables_.emplace(std::move(key), future);
    // Clone the module to avoid race condition between Freeze() and
    // compilation.
    module_copy = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
  }

  LOG(INFO) << "Cache missed. Building executable";

  auto compile =
      [&]() -> absl::StatusOr<std::shared_ptr<const CompiledExecutable>> {
    tensorflow::tpu::TPUCompileMetadataProto compile_metadata =
        original_compile_metadata_;

    // b/469105465: Add test coverage for core selection in execution.
    if (UsePortableExecution()) {
      // Clear device_assignment because portable execution doesn't allow device
      // assignment.
      compile_metadata.clear_device_assignment();
    }

    TF_RETURN_IF_ERROR(
        UpdateCompileMetadata(compile_metadata, dtypes_and_shapes));

    return CreateExecutableSynchronously(std::move(module_copy),
                                         compile_metadata, dtypes_and_shapes,
                                         variable_arg_indices);
  }();

  if (!compile.ok()) {
    promise.Set(compile.status());
    return compile.status();
  }
  promise.Set(*std::move(compile));

  // Here is an immediate return as promise is already set.
  return future;
}

void IfrtProgram::Freeze() {
  absl::MutexLock lock(compile_mutex_);
  is_frozen_ = true;
  module_ = nullptr;
}

bool IfrtProgram::UsePortableExecution() const {
  // TODO(b/335247101) Add a check that the core selector must be non-null if
  // it is a single-device program after core selection in Ifrt is stable.
  return IsSingleDevice(original_compile_metadata_) &&
         ifrt_serving_core_selector_;
}

absl::StatusOr<std::vector<bool>> GetInputDonationMask(
    const xla::ifrt::LoadedExecutable& ifrt_executable, int num_inputs) {
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<xla::HloModule>> hlo_modules,
                      ifrt_executable.GetHloModules());
  std::vector<bool> input_may_be_donated(num_inputs, false);
  absl::Status status = absl::OkStatus();
  auto mark_donated = [&](int64_t parameter_number) {
    // An out-of-range parameter means our input numbering does not match the
    // program's; the mask cannot be trusted, so fail instead of guessing.
    if (parameter_number < 0 || parameter_number >= num_inputs) {
      status.Update(absl::InternalError(absl::StrCat(
          "Donated parameter ", parameter_number,
          " is out of range for an executable with ", num_inputs, " inputs")));
      return;
    }
    input_may_be_donated[parameter_number] = true;
  };
  for (const std::shared_ptr<xla::HloModule>& hlo_module : hlo_modules) {
    // Both kMayAlias and kMustAlias permit the runtime to donate the input.
    hlo_module->input_output_alias_config().ForEachAlias(
        [&](const xla::ShapeIndex& output_index,
            const xla::HloInputOutputAliasConfig::Alias& alias) {
          mark_donated(alias.parameter_number);
        });
    for (const auto& donor : hlo_module->buffer_donor_config().buffer_donor()) {
      mark_donated(donor.param_number);
    }
  }
  TF_RETURN_IF_ERROR(status);
  return input_may_be_donated;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
