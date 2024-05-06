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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/extract_callback.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/mlir/tfrt/utils/export.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_tensor_utils.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/framework/serving_device_selector.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tstring.h"

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

  int variable_index = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (variable_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_index]) {
      // Get already loaded variable tensor.
      TF_ASSIGN_OR_RETURN(auto dtype_and_shape,
                          ifrt_restore_tensor_registry.GetDtypeAndShape(
                              inputs[i].scalar<tsl::tstring>()()));
      dtypes_and_shapes.push_back(std::move(dtype_and_shape));

      variable_index++;
    } else {
      dtypes_and_shapes.push_back(DtypeAndShape{.dtype = inputs[i].dtype(),
                                                .shape = inputs[i].shape()});
    }
  }
  return dtypes_and_shapes;
}

absl::StatusOr<xla::DeviceAssignment> GetXlaDeviceAssignment(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  if (!compile_metadata.has_device_assignment()) {
    return absl::InternalError("No device assignment found.");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::DeviceAssignment> da,
      xla::DeviceAssignment::Deserialize(compile_metadata.device_assignment()));
  return *da;
}

absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedDevices(
    const xla::ifrt::Client& ifrt_client,
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata) {
  TF_ASSIGN_OR_RETURN(auto device_assignment,
                      GetXlaDeviceAssignment(compile_metadata));
  const int num_devices =
      device_assignment.replica_count() * device_assignment.computation_count();
  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(num_devices);
  for (int replica_idx = 0; replica_idx < device_assignment.replica_count();
       replica_idx++) {
    for (int computation_idx = 0;
         computation_idx < device_assignment.computation_count();
         computation_idx++) {
      auto device_id = device_assignment(replica_idx, computation_idx);
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Device * device,
          ifrt_client.LookupDevice(xla::ifrt::DeviceId(device_id)));
      devices.push_back(device);
    }
  }
  return devices;
}

}  // namespace

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
IfrtServingExecutable::ConvertTensorToArray(
    const tensorflow::Tensor& tensor, const xla::ifrt::DeviceList& device_list,
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
    mlir::ModuleOp module, tensorflow::StaticDeviceMgr* device_mgr,
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

  // TODO(b/332774825): reuse functions in BEF/MLRT once we switch to
  // GraphExecutor.
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> callback_module,
                      ExtractCallbackModule(module, key));

  TF_ASSIGN_OR_RETURN(std::vector<tensorflow::FunctionDef> function_defs,
                      BuildFunctionDef(*callback_module));

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
    const Tf2HloResult& tf2hlo_result, mlir::ModuleOp module,
    tensorflow::StaticDeviceMgr* device_mgr,
    std::vector<std::unique_ptr<TfHostCallback>>& tf_host_callbacks) {
  TF_ASSIGN_OR_RETURN(auto host_callback_maps,
                      GroupHostCallbackByKey(tf2hlo_result));

  std::vector<xla::HostCallback> host_callbacks;
  host_callbacks.reserve(host_callback_maps.size());
  for (const auto& [entry_function, builder_info] : host_callback_maps) {
    TF_ASSIGN_OR_RETURN(auto host_callback,
                        BuildHostCallback(entry_function, builder_info, module,
                                          device_mgr, tf_host_callbacks));
    host_callbacks.push_back(std::move(host_callback));
  }

  return host_callbacks;
}

absl::StatusOr<IfrtServingExecutable::SharedCachedExecutableBundle>
IfrtServingExecutable::CreateExecutableSynchronously(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
    absl::Span<const DtypeAndShape> dtypes_and_shapes) {
  // Clone the module b/c CompileTfToHlo serialize the module and may lead to
  // race condition.
  mlir::OwningOpRef<mlir::ModuleOp> module_copy(module_->clone());
  TF_ASSIGN_OR_RETURN(
      Tf2HloResult tf2hlo_result,
      CompileTfToHlo(*module_copy, dtypes_and_shapes, signature_name(),
                     *ifrt_client_, compile_metadata,
                     shape_representation_fn_));
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
  // TODO(b/304839793): populate xla_compile_options.argument_layouts.
  // TODO(b/316071625): per model config in TFRT + IFRT.
  xla_compile_options.executable_build_options.set_num_replicas(num_replicas);
  xla_compile_options.executable_build_options.set_num_partitions(
      num_partitions);

  xla_compile_options.executable_build_options.set_use_spmd_partitioning(true);
  xla_compile_options.parameter_is_tupled_arguments = false;
  // Use portable execution for single device + core selection.
  if (UsePortableExecution(compile_metadata)) {
    xla_compile_options.compile_portable_executable = true;
  } else {
    TF_ASSIGN_OR_RETURN(xla::DeviceAssignment da,
                        GetXlaDeviceAssignment(tf2hlo_result.compile_metadata));
    VLOG(2) << "Device assignment :" << da.ToString();
    xla_compile_options.executable_build_options.set_device_assignment(da);
  }

  std::vector<std::unique_ptr<TfHostCallback>> tf_host_callbacks;
  TF_ASSIGN_OR_RETURN(auto host_callbacks,
                      BuildHostCallbacks(tf2hlo_result, *module_, device_mgr_,
                                         tf_host_callbacks));

  std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
      loaded_host_callbacks;
  loaded_host_callbacks.reserve(host_callbacks.size());
  for (const auto& host_callback : host_callbacks) {
    loaded_host_callbacks.push_back(
        tsl::MakeRef<xla::ifrt::PjRtHostSendAndRecvLoadedHostCallback>(
            ifrt_client_.get(),
            std::make_unique<xla::HostCallback>(host_callback)));
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::LoadedExecutable> ifrt_executable,
      ifrt_client_->GetDefaultCompiler()->Compile(
          std::make_unique<xla::ifrt::XlaProgram>(
              tf2hlo_result.mlir_hlo_module.get()),
          std::make_unique<xla::ifrt::XlaCompileOptions>(
              xla_compile_options, loaded_host_callbacks)));

  SharedCachedExecutableBundle executable_bundle =
      std::make_shared<CachedExecutableBundle>();
  executable_bundle->ifrt_executable = std::move(ifrt_executable);
  executable_bundle->compile_metadata =
      std::move(tf2hlo_result.compile_metadata);
  executable_bundle->host_callbacks = std::move(tf_host_callbacks);

  return executable_bundle;
}

xla::ifrt::Future<IfrtServingExecutable::SharedCachedExecutableBundle>
IfrtServingExecutable::LookUpOrCreateExecutable(
    const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
    absl::Span<const DtypeAndShape> dtypes_and_shapes) {
  std::vector<tensorflow::TensorShape> input_shapes;
  for (const auto& dtype_and_shape : dtypes_and_shapes) {
    input_shapes.push_back(dtype_and_shape.shape);
  }
  Key key = {.input_shapes = std::move(input_shapes)};

  xla::ifrt::Promise<SharedCachedExecutableBundle> promise;
  xla::ifrt::Future<SharedCachedExecutableBundle> future;

  {
    absl::MutexLock lock(&mutex_);

    const auto it = executable_bundles_.find(key);
    if (it != executable_bundles_.end()) {
      return it->second;
    }

    // Only create promise and future when cache missed.
    promise = xla::ifrt::Future<SharedCachedExecutableBundle>::CreatePromise();
    future = xla::ifrt::Future<SharedCachedExecutableBundle>(promise);

    executable_bundles_.emplace(key, future);
  }

  LOG(INFO) << "Cache missed. Building executable";
  absl::StatusOr<SharedCachedExecutableBundle> executable_bundle =
      CreateExecutableSynchronously(compile_metadata, dtypes_and_shapes);
  promise.Set(std::move(executable_bundle));
  return future;
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

  TF_ASSIGN_OR_RETURN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(*module_, dtypes_and_shapes, *ifrt_client_));

  // `device_reservation` should be alive before the end of the execution.
  tsl::DeviceReservation device_reservation(kNoCoreSelectedIndex, nullptr);
  std::vector<xla ::ifrt::Device*> devices;
  if (UsePortableExecution(compile_metadata)) {
    device_reservation =
        ifrt_serving_core_selector_->ReserveDevice(program_id_);
    // Clear device_assignment because portable execution doesn't allow device
    // assignment.
    compile_metadata.clear_device_assignment();
    TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(
                            device_reservation.device_index())));
    devices.push_back(device);
  } else {
    TF_ASSIGN_OR_RETURN(devices,
                        GetAssignedDevices(*ifrt_client_, compile_metadata));
  }
  TF_ASSIGN_OR_RETURN(SharedCachedExecutableBundle executable_bundle,
                      LookUpOrCreateExecutable(
                          compile_metadata, absl::MakeSpan(dtypes_and_shapes))
                          .Await());

  xla::ifrt::DeviceList device_list(
      xla::ifrt::DeviceList::Devices(devices.begin(), devices.end()));

  if (executable_bundle->compile_metadata.args().size() !=
      dtypes_and_shapes.size()) {
    return absl::InternalError(absl::StrCat(
        "Expected ", executable_bundle->compile_metadata.args().size(),
        " but got ", dtypes_and_shapes.size(), " arguments"));
  }

  // Asynchronously load the restored variable tensors to Ifrt array.
  TF_RETURN_IF_ERROR(AsyncLoadIfrtArray(inputs, variable_arg_indices,
                                        *executable_bundle, devices));

  std::vector<tsl::RCReference<xla::ifrt::Array>> args;
  args.reserve(inputs.size());
  int variable_index = 0;
  for (int i = 0; i < inputs.size(); i++) {
    if (variable_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_index]) {
      absl::flat_hash_set<int> device_ids;
      for (const auto& device : devices) {
        device_ids.insert(device->Id().value());
      }
      IfrtLoadedVariableRegistry::Key key{
          .device_ids = std::move(device_ids),
          .input_name = inputs[i].scalar<tsl::tstring>()(),
      };
      TF_ASSIGN_OR_RETURN(
          auto loaded_variable,
          ifrt_loaded_variable_registry_.GetLoadedVariable(key));
      TF_ASSIGN_OR_RETURN(tsl::RCReference<xla::ifrt::Array> single_array,
                          loaded_variable.array.Await());
      args.push_back(std::move(single_array));
      variable_index++;
    } else {
      TF_ASSIGN_OR_RETURN(
          auto single_array,
          ConvertTensorToArray(
              inputs[i], device_list,
              executable_bundle->compile_metadata.args()[i].sharding()));
      args.push_back(single_array);
    }
  }
  DCHECK_EQ(args.size(), dtypes_and_shapes.size());

  VLOG(2) << "Start Execution";

  std::optional<xla::ifrt::DeviceList> execution_device_list;
  if (UsePortableExecution(compile_metadata)) {
    execution_device_list = device_list;
  }
  TF_ASSIGN_OR_RETURN(
      auto execution_result,
      executable_bundle->ifrt_executable->Execute(
          absl::MakeSpan(args),
          /*options=*/
          {.untuple_result = true,
           .use_major_to_minor_data_layout_for_callbacks = true},
          std::move(execution_device_list)));

  auto status = execution_result.status.Await();
  TF_RETURN_IF_ERROR(status);

  std::vector<tensorflow::Tensor> outputs;

  if (executable_bundle->compile_metadata.retvals().size() !=
      execution_result.outputs.size()) {
    return absl::InternalError(absl::StrCat(
        "Expect ", executable_bundle->compile_metadata.retvals().size(),
        " but got ", execution_result.outputs.size(), " outputs"));
  }
  for (int i = 0; i < execution_result.outputs.size(); ++i) {
    tensorflow::TensorShape tensor_shape;
    const tsl::RCReference<xla::ifrt::Array>& array_for_copy =
        execution_result.outputs[i];
    const tpu::TPUCompileMetadataProto::Retval& metadata_retval =
        executable_bundle->compile_metadata.retvals()[i];

    // IFRT's return does not contain sufficient information; so we use
    // sharding spec from metadata.
    VLOG(2) << "Output sharding: " << array_for_copy->sharding().DebugString();

    TF_ASSIGN_OR_RETURN(auto hlo_sharding, xla::HloSharding::FromProto(
                                               metadata_retval.sharding()));
    TF_ASSIGN_OR_RETURN(
        tensorflow::Tensor tensor,
        MakeTensorFromArray(*ifrt_client_, *array_for_copy, hlo_sharding,
                            device_list, thread_pool_));
    outputs.push_back(std::move(tensor));
  }

  return outputs;
}

absl::Status IfrtServingExecutable::AsyncLoadIfrtArray(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const CachedExecutableBundle& executable_bundle,
    const std::vector<xla::ifrt::Device*>& devices) {
  for (const int i : variable_arg_indices) {
    if (inputs[i].dtype() != tensorflow::DT_STRING ||
        !tensorflow::TensorShapeUtils::IsScalar(inputs[i].shape())) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected a scalar tensor as loaded variable array key, "
                       "but got type ",
                       inputs[i].dtype(), " and shape ",
                       inputs[i].shape().DebugString(), " at index ", i));
    }
    std::string runtime_name = inputs[i].scalar<tsl::tstring>()();
    // TODO(b/330360798): Add test cases for OpSharding on variables.
    VariableDeviceShardingConfigProto sharding_config;
    *sharding_config.mutable_sharding() =
        executable_bundle.compile_metadata.args()[i].sharding();
    for (const auto& device : devices) {
      sharding_config.add_device_ids(device->Id().value());
    }

    TF_RETURN_IF_ERROR(
        ifrt_serving::AsyncLoadRestoredTensorAsIfrtLoadedVariable(
            runtime_name, ifrt_client_, thread_pool_,
            ifrt_restore_tensor_registry_, ifrt_loaded_variable_registry_,
            checkpoint_loader_queue_, sharding_config));
  }
  return absl::OkStatus();
}
}  // namespace ifrt_serving
}  // namespace tensorflow
