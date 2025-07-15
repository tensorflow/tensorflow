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

#include "tensorflow/core/tfrt/ifrt/ifrt_serving_executable.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
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
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "shardy/dialect/sdy/transforms/import/passes.h"  // from @shardy
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/extract_callback.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/mlir/tfrt/utils/export.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dump.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_device_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/platform/tstring.h"
#include "tsl/profiler/lib/traceme.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {
namespace {

bool IsSingleDevice(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  return compile_metadata.num_replicas() == 1 &&
         compile_metadata.num_cores_per_replica() == 1;
}

absl::StatusOr<std::vector<DtypeAndShape>> BuildDtypeAndShape(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry) {
  std::vector<DtypeAndShape> dtypes_and_shapes;
  dtypes_and_shapes.reserve(inputs.size());

  int variable_arg_index = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      // Get already loaded variable tensor.
      TF_ASSIGN_OR_RETURN(auto dtype_and_shape,
                          ifrt_restore_tensor_registry.GetDtypeAndShape(
                              inputs[i].scalar<tsl::tstring>()()));
      dtypes_and_shapes.push_back(std::move(dtype_and_shape));

      variable_arg_index++;
    } else {
      dtypes_and_shapes.push_back(DtypeAndShape{.dtype = inputs[i].dtype(),
                                                .shape = inputs[i].shape()});
    }
  }
  return dtypes_and_shapes;
}

// Returns the device assignment from the given IFRT devices list.
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

absl::StatusOr<bool> GetUseShardyPartitioner(mlir::ModuleOp module) {
  std::optional<bool> use_shardy_partitioner;
  mlir::WalkResult result = module->walk([&](mlir::TF::XlaCallModuleOp op) {
    if (!use_shardy_partitioner.has_value()) {
      use_shardy_partitioner = op.getUseShardyPartitioner();
    } else if (*use_shardy_partitioner != op.getUseShardyPartitioner()) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return absl::FailedPreconditionError(
        "use_shardy_partitioner is not consistent across XlaCallModuleOps");
  }

  if (!use_shardy_partitioner.has_value()) {
    // If the module doesn't contain any XlaCallModuleOp, disable Shardy.
    use_shardy_partitioner = false;
  }
  VLOG(2) << "use_shardy_partitioner: " << *use_shardy_partitioner;
  return *use_shardy_partitioner;
}

// We first convert mhlo.sharding to sdy.sharding. Then, we call
// the SdyRoundTrip import pass to convert the
// `mhlo.frontend_attributes={xla.sdy.sharding...}` to sdy.sharding. After
// that we lift the meshes that were inlined when we built the module for the
// cluster. We don't need to invoke SdyRoundTrip export here as MLIR to HLO will
// perform that.
absl::Status ImportShardingsAndLiftInlinedMeshes(mlir::ModuleOp module) {
  mlir::PassManager sdy_roundtrip(module->getContext());
  sdy_roundtrip.addPass(xla::sdy::createImportShardingsPass(
      /*allowPropagationToArgs=*/false, /*allowPropagationToResults=*/false));
  xla::sdy::addSdyRoundTripImportPipeline(sdy_roundtrip,
                                          /*enableConstantImport=*/false);
  sdy_roundtrip.addPass(mlir::sdy::createLiftInlinedMeshesPass());

  tsl::StatusScopedDiagnosticHandler diagnosticHandler(module->getContext());
  absl::Status status =
      diagnosticHandler.consumeStatus(sdy_roundtrip.run(module));
  if (status.ok() && VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_bridge_phase2_sdy", module);
  }
  return status;
}

}  // namespace

absl::StatusOr<std::unique_ptr<IfrtServingExecutable>>
IfrtServingExecutable::Create(
    int64_t program_id, absl::string_view model_name,
    absl::string_view signature_name, mlir::OwningOpRef<mlir::ModuleOp> module,
    std::shared_ptr<xla::ifrt::Client> client,
    tsl::thread::ThreadPool* thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    const IfrtRestoreTensorRegistry* ifrt_restore,
    tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
    tensorflow::DeviceMgr* device_mgr,
    tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    IfrtServingCoreSelector* ifrt_serving_core_selector,
    tsl::protobuf::Message* compilation_environment_proto,
    TfToHloCompiler* tf_to_hlo_compiler,
    IfrtPersistentCompilationCache* persistent_compilation_cache) {
  TF_ASSIGN_OR_RETURN(
      tensorflow::tpu::TPUCompileMetadataProto original_compile_metadata,
      GetCompileMetadata(*module, *client));

  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::Device*> assigned_devices,
      GetAssignedDevices(*module, *client,
                         original_compile_metadata.num_replicas(),
                         original_compile_metadata.num_cores_per_replica()));

  auto executable = absl::WrapUnique(new IfrtServingExecutable(
      program_id, model_name, signature_name, std::move(module), client,
      thread_pool, ifrt_loaded_variable_registry, ifrt_restore,
      checkpoint_loader_queue, device_mgr, std::move(shape_representation_fn),
      ifrt_serving_core_selector, std::move(original_compile_metadata),
      client->MakeDeviceList(assigned_devices), compilation_environment_proto,
      tf_to_hlo_compiler, persistent_compilation_cache));

  return executable;
}

absl::StatusOr<xla::ifrt::ArrayRef> IfrtServingExecutable::ConvertTensorToArray(
    const tensorflow::Tensor& tensor,
    const xla::ifrt::DeviceListRef& device_list,
    const xla::OpSharding& sharding) {
  xla::ifrt::Shape input_shape = ToIfrtShape(tensor.shape());
  VLOG(2) << "Converting tensor of shape " << input_shape;

  TF_ASSIGN_OR_RETURN(auto hlo_sharding, xla::HloSharding::FromProto(sharding));

  return MakeArrayFromTensor(*ifrt_client_, tensor, device_list,
                             std::move(hlo_sharding), thread_pool_);
}

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
GroupHostCallbackByKey(const Tf2HloResult& tf2hlo_result) {
  absl::flat_hash_map<std::string, HostCallbackBuilderInfo> host_callbacks;

  for (const auto& device_to_host :
       tf2hlo_result.host_compute_metadata.device_to_host()) {
    auto& host_callback = host_callbacks[device_to_host.key()];
    host_callback.device_to_host = device_to_host;
  }
  for (const auto& host_to_device :
       tf2hlo_result.host_compute_metadata.host_to_device()) {
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
    const Tf2HloResult& tf2hlo_result,
    absl::flat_hash_map<std::string, mlir::OwningOpRef<mlir::ModuleOp>>
        host_callback_modules,
    tensorflow::DeviceMgr* device_mgr,
    std::vector<std::unique_ptr<TfHostCallback>>& tf_host_callbacks) {
  TF_ASSIGN_OR_RETURN(auto host_callback_maps,
                      GroupHostCallbackByKey(tf2hlo_result));

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

absl::StatusOr<IfrtServingExecutable::SharedCachedExecutableBundle>
IfrtServingExecutable::CreateExecutableSynchronously(
    mlir::OwningOpRef<mlir::ModuleOp> module_copy,
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices) {
  TF_ASSIGN_OR_RETURN(auto host_callback_modules,
                      GetHostCallbackModulesAndRemoveHostFuncs(*module_copy));
  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("module_for_bridge_phase2", *module_copy);
  }

  TF_ASSIGN_OR_RETURN(bool use_shardy_partitioner,
                      GetUseShardyPartitioner(module_copy.get()));

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

  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_hlo_module,
      ::xla::ConvertHloToStablehloWithOptions(
          *module_copy->getContext(), &tf2hlo_result.hlo_module_proto,
          /*import_all_computations=*/false));

  if (VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile("ifrt_after_bridge_phase2",
                                 mlir_hlo_module.get());
  }

  if (use_shardy_partitioner) {
    // We have inlined meshes to build the module for the cluster, but Shardy
    // expects lifted meshes.
    TF_RETURN_IF_ERROR(
        ImportShardingsAndLiftInlinedMeshes(mlir_hlo_module.get()));
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
  if (compilation_environment_proto_) {
    tsl::protobuf::Message* comp_env_copy =
        compilation_environment_proto_->New();
    comp_env_copy->CopyFrom(*compilation_environment_proto_);
    TF_RETURN_IF_ERROR(
        xla_compile_options.executable_build_options.mutable_comp_envs()
            ->AddEnv(absl::WrapUnique<tsl::protobuf::Message>(comp_env_copy)));
  }

  xla_compile_options.executable_build_options.set_num_replicas(num_replicas);
  xla_compile_options.executable_build_options.set_num_partitions(
      num_partitions);

  xla_compile_options.executable_build_options.set_use_spmd_partitioning(
      original_compile_metadata_.use_spmd_for_xla_partitioning());
  xla_compile_options.executable_build_options.set_use_shardy_partitioner(
      use_shardy_partitioner);
  xla_compile_options.parameter_is_tupled_arguments = false;
  // Use portable execution for single device + core selection.
  if (UsePortableExecution(compile_metadata)) {
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
  TF_ASSIGN_OR_RETURN(
      auto host_callbacks,
      BuildHostCallbacks(tf2hlo_result, std::move(host_callback_modules),
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
  SharedCachedExecutableBundle executable_bundle =
      std::make_shared<CachedExecutableBundle>();

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutableRef ifrt_executable,
      persistent_compilation_cache_->LookupLoadedExecutableOrCreate(
          std::move(hlo_program), assigned_device_list_, xla_compile_options,
          loaded_host_callbacks, ifrt_client_.get(),
          [&](std::unique_ptr<xla::ifrt::Program> program,
              std::unique_ptr<xla::ifrt::CompileOptions> options)
              -> absl::StatusOr<xla::ifrt::LoadedExecutableRef> {
            return ifrt_client_->GetDefaultCompiler()->CompileAndLoad(
                std::move(program), std::move(options));
          }));

  executable_bundle->ifrt_executable = std::move(ifrt_executable);
  executable_bundle->compile_metadata =
      std::move(tf2hlo_result.compile_metadata);
  executable_bundle->host_callbacks = std::move(tf_host_callbacks);

  return executable_bundle;
}

xla::ifrt::Future<IfrtServingExecutable::SharedCachedExecutableBundle>
IfrtServingExecutable::LookUpOrCreateExecutable(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices) {
  std::vector<tensorflow::TensorShape> input_shapes;
  for (const auto& dtype_and_shape : dtypes_and_shapes) {
    input_shapes.push_back(dtype_and_shape.shape);
  }
  Key key = {.input_shapes = std::move(input_shapes)};

  xla::ifrt::Promise<SharedCachedExecutableBundle> promise;
  xla::ifrt::Future<SharedCachedExecutableBundle> future;
  mlir::OwningOpRef<mlir::ModuleOp> module_copy;
  {
    absl::MutexLock lock(&mutex_);

    const auto it = executable_bundles_.find(key);
    if (it != executable_bundles_.end()) {
      return it->second;
    }

    if (is_frozen_) {
      xla::ifrt::Future<SharedCachedExecutableBundle> frozen_future(
          absl::FailedPreconditionError(
              "Cannot compile for new input shapes after the executable is "
              "already frozen."));
      return frozen_future;
    }

    // Only create promise and future when cache missed.
    promise = xla::ifrt::Future<SharedCachedExecutableBundle>::CreatePromise();
    future = xla::ifrt::Future<SharedCachedExecutableBundle>(promise);

    executable_bundles_.emplace(key, future);
    // Clone the module to avoid race condition between Freeze() and
    // compilation.
    module_copy = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
  }

  LOG(INFO) << "Cache missed. Building executable";
  absl::StatusOr<SharedCachedExecutableBundle> executable_bundle =
      CreateExecutableSynchronously(std::move(module_copy), compile_metadata,
                                    dtypes_and_shapes, variable_arg_indices);
  promise.Set(std::move(executable_bundle));
  return future;
}

void IfrtServingExecutable::Freeze() {
  LOG(INFO) << "Freezing executable. Program id: " << program_id_;
  absl::MutexLock lock(&mutex_);
  is_frozen_ = true;
  module_ = nullptr;
}

bool IfrtServingExecutable::UsePortableExecution(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  // TODO(b/335247101) Add a check that the core selector must be non-null if
  // it is a single-device program after core selection in Ifrt is stable.
  return IsSingleDevice(compile_metadata) && ifrt_serving_core_selector_;
}

absl::StatusOr<std::vector<tensorflow::Tensor>> IfrtServingExecutable::Execute(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices) {
  tsl::profiler::TraceMe traceme("IfrtServingExecutable::Execute");
  for (int i = 1; i < variable_arg_indices.size(); i++) {
    if (variable_arg_indices[i] <= variable_arg_indices[i - 1]) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Expected variable_arg_indices in ascending order. But subsequence "
          "starting at ",
          i - 1, ": (", variable_arg_indices[i - 1], ", ",
          variable_arg_indices[i], ")", " is not in ascending order"));
    }
  }

  if (!variable_arg_indices.empty() &&
      inputs.size() <= variable_arg_indices.back()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected at most ", inputs.size(), " inputs, but got up to ",
        variable_arg_indices.back(), " variables."));
  }

  // Ensure the variable tensor holds a valid key: a scalar string tensor.
  for (const int i : variable_arg_indices) {
    if (inputs[i].dtype() != tensorflow::DT_STRING ||
        !tensorflow::TensorShapeUtils::IsScalar(inputs[i].shape())) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected a scalar tensor as loaded variable array key, "
                       "but got type ",
                       inputs[i].dtype(), " and shape ",
                       inputs[i].shape().DebugString(), " at index ", i));
    }
  }

  TF_ASSIGN_OR_RETURN(std::vector<DtypeAndShape> dtypes_and_shapes,
                      BuildDtypeAndShape(inputs, variable_arg_indices,
                                         ifrt_restore_tensor_registry_));

  tensorflow::tpu::TPUCompileMetadataProto compile_metadata =
      original_compile_metadata_;
  TF_RETURN_IF_ERROR(
      UpdateCompileMetadata(compile_metadata, dtypes_and_shapes));

  // `device_reservation` should be alive before the end of the execution.
  tsl::DeviceReservation device_reservation(kNoCoreSelectedIndex, nullptr);
  xla::ifrt::DeviceListRef device_list;
  if (UsePortableExecution(compile_metadata)) {
    device_reservation =
        ifrt_serving_core_selector_->ReserveDevice(program_id_);
    // Clear device_assignment because portable execution doesn't allow device
    // assignment.
    compile_metadata.clear_device_assignment();
    TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(
                            device_reservation.device_index())));
    device_list = ifrt_client_->MakeDeviceList({device});
  } else {
    device_list = assigned_device_list_;
  }
  TF_ASSIGN_OR_RETURN(
      SharedCachedExecutableBundle executable_bundle,
      LookUpOrCreateExecutable(compile_metadata, dtypes_and_shapes,
                               variable_arg_indices)
          .Await());

  if (executable_bundle->compile_metadata.args().size() !=
      dtypes_and_shapes.size()) {
    return absl::InternalError(absl::StrCat(
        "Expected ", executable_bundle->compile_metadata.args().size(),
        " but got ", dtypes_and_shapes.size(), " arguments"));
  }

  {
    tsl::profiler::TraceMe traceme("AsyncRestoreVariables");
    absl::ReaderMutexLock lock(&mutex_);
    if (!is_frozen_) {
      // Asynchronously load the restored variable tensors to Ifrt array.
      TF_RETURN_IF_ERROR(AsyncLoadIfrtArray(inputs, variable_arg_indices,
                                            *executable_bundle, device_list));
    }
  }

  VLOG(2) << "Completed AsyncLoadIfrtArray";

  std::vector<int> device_ids;
  device_ids.reserve(device_list->size());
  for (xla::ifrt::Device* device : device_list->devices()) {
    device_ids.push_back(device->Id().value());
  }
  std::vector<xla::ifrt::ArrayRef> args;
  args.reserve(inputs.size());
  int variable_arg_index = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      TF_ASSIGN_OR_RETURN(
          xla::HloSharding hlo_sharding,
          xla::HloSharding::FromProto(
              executable_bundle->compile_metadata.args()[i].sharding()));
      IfrtLoadedVariableRegistry::Key key{
          .device_ids = device_ids,
          .input_name = inputs[i].scalar<tsl::tstring>()(),
          .hlo_sharding = std::move(hlo_sharding),
      };
      TF_ASSIGN_OR_RETURN(
          auto loaded_variable,
          ifrt_loaded_variable_registry_.GetLoadedVariable(key));
      TF_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef single_array,
                          loaded_variable.array.Await());
      args.push_back(std::move(single_array));
      variable_arg_index++;
    } else {
      // If the input shape is not the same as the shape after Tf2Hlo
      // compilation, reshape the input tensor to the expected shape. Note that
      // the tensor assignment here won't create a copy.
      tensorflow::Tensor reshaped = inputs[i];
      TF_ASSIGN_OR_RETURN(
          tensorflow::TensorShape reshaped_shape,
          tensorflow::TensorShape::BuildTensorShape(
              executable_bundle->compile_metadata.args()[i].shape()));
      if (reshaped.shape() != reshaped_shape &&
          !reshaped.CopyFrom(inputs[i], reshaped_shape)) {
        return absl::InternalError("Failed to reshape tensor");
      }

      TF_ASSIGN_OR_RETURN(
          auto single_array,
          ConvertTensorToArray(
              reshaped, device_list,
              executable_bundle->compile_metadata.args()[i].sharding()));
      args.push_back(single_array);
    }
  }
  DCHECK_EQ(args.size(), executable_bundle->compile_metadata.args().size());

  VLOG(2) << "Start Execution";

  std::optional<xla::ifrt::DeviceListRef> execution_device_list;
  if (UsePortableExecution(compile_metadata)) {
    execution_device_list = device_list;
  }

  absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult> execution_result;
  {
    tsl::profiler::TraceMe traceme("Execute");
    execution_result = executable_bundle->ifrt_executable->Execute(
        absl::MakeSpan(args), /*options=*/{.fill_status = true},
        std::move(execution_device_list));
    TF_RETURN_IF_ERROR(execution_result.status());
  }

  auto status = execution_result->status.Await();
  TF_RETURN_IF_ERROR(status);

  if (executable_bundle->compile_metadata.retvals().size() !=
      execution_result->outputs.size()) {
    return absl::InternalError(absl::StrCat(
        "Expect ", executable_bundle->compile_metadata.retvals().size(),
        " but got ", execution_result->outputs.size(), " outputs"));
  }

  std::vector<xla::ifrt::Future<tensorflow::Tensor>> output_futures;
  output_futures.reserve(execution_result->outputs.size());
  for (int i = 0; i < execution_result->outputs.size(); ++i) {
    tensorflow::TensorShape tensor_shape;
    const xla::ifrt::ArrayRef& array_for_copy = execution_result->outputs[i];
    const tpu::TPUCompileMetadataProto::Retval& metadata_retval =
        executable_bundle->compile_metadata.retvals()[i];

    // IFRT's return does not contain sufficient information; so we use
    // sharding spec from metadata.
    VLOG(2) << "Output sharding: " << array_for_copy->sharding().DebugString();

    TF_ASSIGN_OR_RETURN(auto hlo_sharding, xla::HloSharding::FromProto(
                                               metadata_retval.sharding()));
    output_futures.push_back(MakeTensorFromArray(*ifrt_client_, *array_for_copy,
                                                 hlo_sharding, device_list,
                                                 thread_pool_));
  }

  std::vector<tensorflow::Tensor> outputs;
  outputs.reserve(output_futures.size());
  for (auto& output_future : output_futures) {
    TF_ASSIGN_OR_RETURN(auto tensor, output_future.Await());
    outputs.push_back(std::move(tensor));
  }
  return outputs;
}

absl::Status IfrtServingExecutable::AsyncLoadIfrtArray(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const CachedExecutableBundle& executable_bundle,
    const xla::ifrt::DeviceListRef& devices) {
  for (const int i : variable_arg_indices) {
    if (inputs[i].dtype() != tensorflow::DT_STRING ||
        !tensorflow::TensorShapeUtils::IsScalar(inputs[i].shape())) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected a scalar tensor as loaded variable array key, "
                       "but got type ",
                       inputs[i].dtype(), " and shape ",
                       inputs[i].shape().DebugString(), " at index ", i));
    }
    std::string tensor_name = inputs[i].scalar<tsl::tstring>()();
    // TODO(b/339521818): Add test cases for OpSharding on variables.
    TF_ASSIGN_OR_RETURN(
        xla::HloSharding hlo_sharding,
        xla::HloSharding::FromProto(
            executable_bundle.compile_metadata.args()[i].sharding()));
    VariableDeviceShardingConfig sharding_config{
        .hlo_sharding = std::move(hlo_sharding),
    };
    for (xla::ifrt::Device* device : devices->devices()) {
      sharding_config.device_ids.push_back(device->Id().value());
    }

    TF_RETURN_IF_ERROR(
        ifrt_serving::AsyncLoadRestoredTensorAsIfrtLoadedVariable(
            tensor_name, ifrt_client_, thread_pool_,
            ifrt_restore_tensor_registry_, ifrt_loaded_variable_registry_,
            checkpoint_loader_queue_, sharding_config));
  }
  return absl::OkStatus();
}
}  // namespace ifrt_serving
}  // namespace tensorflow
