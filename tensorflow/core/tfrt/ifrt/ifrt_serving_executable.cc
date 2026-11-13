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
#include "absl/container/flat_hash_set.h"
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
#include "mlir/Pass/PassManager.h"  // from @llvm-project
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
#include "xla/python/ifrt/array.h"
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
#include "xla/python/ifrt/topology.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/dump.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "xla/tsl/lib/monitoring/counter.h"
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

auto* ifrt_execution_counter = tsl::monitoring::Counter<1>::New(
    "/tensorflow/tfrt/ifrt/execution_count",
    "Count of IfrtServingExecutable::ExecuteCore executions.", "model_name");

using StaticShapeMap =
    absl::flat_hash_map<size_t /*original_arg_idx*/,
                        tensorflow::TensorShape /*static_shape*/>;

absl::StatusOr<std::vector<DtypeAndShape>> BuildDtypeAndShape(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const StaticShapeMap& static_shapes_map,
    const IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry) {
  std::vector<DtypeAndShape> dtypes_and_shapes;
  dtypes_and_shapes.reserve(inputs.size());

  const bool has_static_shapes = !static_shapes_map.empty();
  int variable_arg_index = 0;
  for (int i = 0; i < inputs.size(); i++) {
    std::optional<tensorflow::TensorShape> static_shape;
    if (has_static_shapes) {
      auto it = static_shapes_map.find(i);
      if (it != static_shapes_map.end()) {
        static_shape = it->second;
      }
    }
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      // Get already loaded variable tensor.
      TF_ASSIGN_OR_RETURN(auto dtype_and_shape,
                          ifrt_restore_tensor_registry.GetDtypeAndShape(
                              inputs[i].scalar<tsl::tstring>()()));
      dtype_and_shape.static_shape = std::move(static_shape);
      dtypes_and_shapes.push_back(std::move(dtype_and_shape));
      variable_arg_index++;
    } else {
      dtypes_and_shapes.push_back(
          DtypeAndShape{.dtype = inputs[i].dtype(),
                        .shape = inputs[i].shape(),
                        .static_shape = std::move(static_shape)});
    }
  }
  return dtypes_and_shapes;
}

// Retrieves static shapes for the inputs. For arguments specified in
// `static_shape_arg_map`, it extracts the actual shapes from the
// corresponding input tensors.
absl::StatusOr<StaticShapeMap> GetStaticShapesFromInputs(
    absl::Span<const tensorflow::Tensor> inputs,
    const absl::flat_hash_map<size_t, size_t>& static_shape_arg_map) {
  StaticShapeMap static_shapes;
  for (const auto& [original_arg_idx, static_shape_arg_idx] :
       static_shape_arg_map) {
    if (static_shape_arg_idx >= inputs.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Static shape arg index out of bound: got index ",
          static_shape_arg_idx, " with inputs size ", inputs.size()));
    }
    const auto& static_shape_tensor = inputs[static_shape_arg_idx];
    if (static_shape_tensor.dims() != 1) {
      return absl::InvalidArgumentError("Static shape tensor must be 1D");
    }
    if (static_shape_tensor.NumElements() > 2) {
      return absl::InvalidArgumentError("Static shape must be 1D or 2D");
    }
    tensorflow::TensorShape static_shape;
    if (static_shape_tensor.dtype() == tensorflow::DT_INT32) {
      auto flat = static_shape_tensor.flat<int32_t>();
      for (size_t k = 0; k < flat.size(); ++k) {
        static_shape.AddDim(flat(k));
      }
    } else if (static_shape_tensor.dtype() == tensorflow::DT_INT64) {
      auto flat = static_shape_tensor.flat<int64_t>();
      for (size_t k = 0; k < flat.size(); ++k) {
        static_shape.AddDim(flat(k));
      }
    } else {
      return absl::InternalError("Static shape tensor must be int32 or int64");
    }
    static_shapes.insert({original_arg_idx, std::move(static_shape)});
  }
  return static_shapes;
}

// If the input tensor's rank differs from `reshaped_shape` (e.g., due to
// compiler flattening), attempt to reshape the input to match the rank of
// `reshaped_shape`. The whole shape does not have to match, because the input
// is expected to be a prefix of the reshaped shape in case of static shape.
// Currently, only flattening to 1D is supported since it suffices the only use
// case we need to support.
absl::StatusOr<tensorflow::Tensor> MaybeReshapeInputForStaticShape(
    const tensorflow::Tensor& input,
    const tensorflow::TensorShape& reshaped_shape) {
  if (input.dims() > 2 || reshaped_shape.dims() > 2) {
    return absl::UnimplementedError(absl::StrCat(
        "MaybeReshapeInputForStaticShape only supports input and reshaped "
        "shapes with at most 2 dimensions. Got input dims: ",
        input.dims(), ", reshaped_shape dims: ", reshaped_shape.dims()));
  }
  if (input.dims() == reshaped_shape.dims()) {
    return input;
  }
  if (input.dims() == 1 && reshaped_shape.dims() == 2) {
    return absl::UnimplementedError(absl::StrCat(
        "Input shape rank is 1 but reshaped shape rank is 2. Got input shape: ",
        input.shape().DebugString(),
        ", reshaped_shape: ", reshaped_shape.DebugString()));
  }
  tensorflow::TensorShape flattened_shape;
  flattened_shape.AddDim(input.NumElements());
  tensorflow::Tensor flattened;
  if (!flattened.CopyFrom(input, flattened_shape)) {
    return absl::InternalError(absl::StrCat(
        "Failed to flatten input tensor to match static shape rank 1. ",
        "Input shape: ", input.shape().DebugString()));
  }
  return flattened;
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
    std::variant<tsl::protobuf::Message*,
                 xla::CompileOptions::EnvironmentOptionOverrides>
        compilation_env_or_overrides,
    TfToHloCompiler* tf_to_hlo_compiler,
    IfrtPersistentCompilationCache* persistent_compilation_cache,
    H2DTransferExecutorFactory* h2d_transfer_executor_factory,
    bool use_output_arena, bool use_undonatable_buffer_converter) {
  if (h2d_transfer_executor_factory == nullptr) {
    return absl::InvalidArgumentError("H2DTransferExecutorFactory is null.");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<IfrtProgram> program,
      IfrtProgram::Create(
          model_name, signature_name, std::move(module), client, device_mgr,
          std::move(shape_representation_fn), ifrt_serving_core_selector,
          compilation_env_or_overrides, tf_to_hlo_compiler,
          persistent_compilation_cache, use_undonatable_buffer_converter));
  return Create(program_id, std::move(program), std::move(client), thread_pool,
                ifrt_loaded_variable_registry, ifrt_restore,
                checkpoint_loader_queue, ifrt_serving_core_selector,
                h2d_transfer_executor_factory, use_output_arena,
                use_undonatable_buffer_converter);
}

absl::StatusOr<std::unique_ptr<IfrtServingExecutable>>
IfrtServingExecutable::Create(
    int64_t program_id, std::shared_ptr<IfrtProgram> program,
    std::shared_ptr<xla::ifrt::Client> client,
    tsl::thread::ThreadPool* thread_pool,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    const IfrtRestoreTensorRegistry* ifrt_restore,
    tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
    IfrtServingCoreSelector* ifrt_serving_core_selector,
    H2DTransferExecutorFactory* h2d_transfer_executor_factory,
    bool use_output_arena, bool use_undonatable_buffer_converter) {
  if (h2d_transfer_executor_factory == nullptr) {
    return absl::InvalidArgumentError("H2DTransferExecutorFactory is null.");
  }
  if (program == nullptr) {
    return absl::InvalidArgumentError("IfrtProgram is null.");
  }
  return absl::WrapUnique(new IfrtServingExecutable(
      program_id, std::move(program), std::move(client), thread_pool,
      ifrt_loaded_variable_registry, ifrt_restore, checkpoint_loader_queue,
      ifrt_serving_core_selector, h2d_transfer_executor_factory,
      use_output_arena, use_undonatable_buffer_converter));
}

absl::Status IfrtServingExecutable::LoadAndRegisterVariableOnExecutable(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const xla::ifrt::DeviceListRef& device_list,
    IfrtServingExecutable::CachedExecutableBundle* executable_bundle) {
  TF_RETURN_IF_ERROR(AsyncLoadIfrtArray(
      inputs, variable_arg_indices, *executable_bundle->compiled, device_list));

  std::vector<int> device_ids;
  device_ids.reserve(device_list->size());
  for (xla::ifrt::Device* device : device_list->devices()) {
    device_ids.push_back(device->Id().value());
  }

  const bool is_portable_execution = UsePortableExecution();
  std::vector<IfrtLoadedVariableRegistry::LoadedVariable>* slot_vec = nullptr;
  if (is_portable_execution) {
    if (!std::holds_alternative<
            CachedExecutableBundle::DeviceLoadedVariableListMap>(
            executable_bundle->variable_arrays)) {
      executable_bundle->variable_arrays
          .emplace<CachedExecutableBundle::DeviceLoadedVariableListMap>();
    }
    xla::ifrt::DeviceId device_id = device_list->devices()[0]->Id();
    std::vector<IfrtLoadedVariableRegistry::LoadedVariable>& variable_arrays =
        std::get<CachedExecutableBundle::DeviceLoadedVariableListMap>(
            executable_bundle->variable_arrays)[device_id];
    variable_arrays.reserve(variable_arg_indices.size());
    slot_vec = &variable_arrays;
  } else {
    if (!std::holds_alternative<CachedExecutableBundle::LoadedVariableList>(
            executable_bundle->variable_arrays)) {
      executable_bundle->variable_arrays
          .emplace<CachedExecutableBundle::LoadedVariableList>();
    }
    std::vector<IfrtLoadedVariableRegistry::LoadedVariable>& variable_arrays =
        std::get<CachedExecutableBundle::LoadedVariableList>(
            executable_bundle->variable_arrays);
    variable_arrays.reserve(variable_arg_indices.size());
    slot_vec = &variable_arrays;
  }
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
    IfrtLoadedVariableRegistry::Key key{
        .device_ids = device_ids,
        .input_name = tensor_name,
        .hlo_sharding = executable_bundle->compiled->arg_hlo_shardings[i],
        .shape_on_device = executable_bundle->compiled->xla_input_shapes[i],
    };
    TF_ASSIGN_OR_RETURN(auto loaded_variable,
                        ifrt_loaded_variable_registry_.GetLoadedVariable(key));
    slot_vec->push_back(std::move(loaded_variable));
  }
  return absl::OkStatus();
}

absl::StatusOr<tsl::Future<IfrtServingExecutable::SharedCachedExecutableBundle>>
IfrtServingExecutable::LookUpOrCreateExecutable(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const DtypeAndShape> dtypes_and_shapes,
    absl::Span<const int> variable_arg_indices,
    const xla::ifrt::DeviceListRef& device_list) {
  {
    absl::MutexLock lock(mutex_);
    const auto it = executable_bundles_.find(KeyView{dtypes_and_shapes});
    if (it != executable_bundles_.end()) {
      return it->second;
    }
  }

  // Compile (or wait for compilation) before taking `mutex_`, so that the
  // failure to compile is not cached as a bundle.
  TF_ASSIGN_OR_RETURN(
      tsl::Future<std::shared_ptr<const CompiledExecutable>> compiled_future,
      program_->GetOrCompile(dtypes_and_shapes, variable_arg_indices));

  std::vector<tensorflow::TensorShape> input_shapes;
  input_shapes.reserve(dtypes_and_shapes.size());
  for (const auto& dtype_and_shape : dtypes_and_shapes) {
    input_shapes.push_back(dtype_and_shape.GetShapeForCompilation());
  }
  auto [promise, future] = tsl::MakePromise<SharedCachedExecutableBundle>();
  {
    absl::MutexLock lock(mutex_);
    // Another call may have created the bundle while `mutex_` was released.
    const auto [it, inserted] = executable_bundles_.try_emplace(
        Key{.input_shapes = std::move(input_shapes)}, future);
    if (!inserted) {
      return it->second;
    }
  }

  auto create_bundle = [&]() -> absl::StatusOr<SharedCachedExecutableBundle> {
    auto executable_bundle = std::make_shared<CachedExecutableBundle>();
    TF_ASSIGN_OR_RETURN(executable_bundle->compiled, compiled_future.Await());

    if (UsePortableExecution()) {
      // If core selector is enabled, we load variables on all cores.
      for (const auto& device : ifrt_client_->addressable_devices()) {
        TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef selected_device_list,
                            ifrt_client_->MakeDeviceList({device}));
        TF_RETURN_IF_ERROR(LoadAndRegisterVariableOnExecutable(
            inputs, variable_arg_indices, selected_device_list,
            executable_bundle.get()));
      }
    } else {
      TF_RETURN_IF_ERROR(LoadAndRegisterVariableOnExecutable(
          inputs, variable_arg_indices, device_list, executable_bundle.get()));
    }

    return executable_bundle;
  }();

  if (!create_bundle.ok()) {
    promise.Set(create_bundle.status());
    return create_bundle.status();
  }

  promise.Set(*create_bundle);

  // Here is an immediate return as promise is already set.
  return future;
}

void IfrtServingExecutable::Freeze() {
  LOG(INFO) << "Freezing executable. Program id: " << program_id_;
  program_->Freeze();
}

absl::StatusOr<IfrtServingExecutable::ExecutionInfo>
IfrtServingExecutable::ExecuteCore(absl::Span<const tensorflow::Tensor> inputs,
                                   absl::Span<const int> variable_arg_indices) {
  ifrt_execution_counter->GetCell(std::string(model_name()))->IncrementBy(1);
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

  StaticShapeMap static_shapes_map;
  const absl::flat_hash_map<size_t, size_t>& static_shape_arg_map =
      program_->static_shape_arg_map();
  if (!static_shape_arg_map.empty()) {
    TF_ASSIGN_OR_RETURN(static_shapes_map, GetStaticShapesFromInputs(
                                               inputs, static_shape_arg_map));
  }
  TF_ASSIGN_OR_RETURN(
      std::vector<DtypeAndShape> dtypes_and_shapes,
      BuildDtypeAndShape(inputs, variable_arg_indices, static_shapes_map,
                         ifrt_restore_tensor_registry_));

  // `device_reservation` should be alive before the end of the execution.
  tsl::DeviceReservation device_reservation(kNoCoreSelectedIndex, nullptr);
  xla::ifrt::DeviceListRef device_list;
  bool use_portable_execution = UsePortableExecution();
  if (use_portable_execution) {
    device_reservation =
        ifrt_serving_core_selector_->ReserveDevice(program_id_);
    TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(
                            device_reservation.device_index())));
    TF_ASSIGN_OR_RETURN(device_list, ifrt_client_->MakeDeviceList({device}));
  } else {
    device_list = program_->assigned_device_list();
  }
  TF_ASSIGN_OR_RETURN(
      tsl::Future<SharedCachedExecutableBundle> executable_bundle_future,
      LookUpOrCreateExecutable(inputs, dtypes_and_shapes, variable_arg_indices,
                               device_list));
  TF_ASSIGN_OR_RETURN(SharedCachedExecutableBundle executable_bundle,
                      executable_bundle_future.Await());

  if (executable_bundle->compiled->compile_metadata.args().size() !=
      dtypes_and_shapes.size()) {
    return absl::InternalError(absl::StrCat(
        "Expected ",
        executable_bundle->compiled->compile_metadata.args().size(),
        " but got ", dtypes_and_shapes.size(), " arguments"));
  }

  int variable_arg_index = 0;
  std::vector<tsl::Future<xla::ifrt::ArrayRef>> variable_args;
  variable_args.reserve(variable_arg_indices.size());

  std::vector<InputHandle> input_handles;
  std::vector<int> input_handle_result_indices;
  input_handles.reserve(inputs.size() - variable_arg_indices.size());
  input_handle_result_indices.reserve(inputs.size() -
                                      variable_arg_indices.size());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<H2DTransferExecutor> user_inputs_h2d_transfer_executor,
      h2d_transfer_executor_factory_->CreateH2DTransferExecutor(*ifrt_client_));

  const std::vector<IfrtLoadedVariableRegistry::LoadedVariable>*
      variable_arrays = nullptr;
  if (use_portable_execution) {
    xla::ifrt::DeviceId device_id = device_list->devices().front()->Id();
    const auto& device_map =
        std::get<CachedExecutableBundle::DeviceLoadedVariableListMap>(
            executable_bundle->variable_arrays);
    auto it = device_map.find(device_id);
    if (it == device_map.end()) {
      return absl::InternalError(
          absl::StrCat("Variable arrays not found for device id: ", device_id));
    }
    variable_arrays = &it->second;
  } else {
    variable_arrays = &std::get<CachedExecutableBundle::LoadedVariableList>(
        executable_bundle->variable_arrays);
  }

  xla::ifrt::ShardingRef portable_sharding;
  if (use_portable_execution) {
    auto it =
        executable_bundle->compiled->portable_single_device_shardings.find(
            device_list->devices().front()->Id());
    if (it ==
        executable_bundle->compiled->portable_single_device_shardings.end()) {
      return absl::InternalError(absl::StrCat(
          "Portable single device sharding not found for device id: ",
          device_list->devices().front()->Id()));
    }
    portable_sharding = it->second;
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (variable_arg_index < variable_arg_indices.size() &&
        i == variable_arg_indices[variable_arg_index]) {
      variable_args.push_back((*variable_arrays)[variable_arg_index].array);
      variable_arg_index++;
    } else {
      // If the input shape is not the same as the shape after Tf2Hlo
      // compilation, reshape the input tensor to the expected shape. Note that
      // the tensor assignment here won't create a copy.
      tensorflow::Tensor reshaped = inputs[i];
      const tensorflow::TensorShape& reshaped_shape =
          executable_bundle->compiled->reshaped_input_tensors[i];
      if (reshaped.shape() != reshaped_shape) {
        if (dtypes_and_shapes[i].static_shape.has_value()) {
          TF_ASSIGN_OR_RETURN(reshaped, MaybeReshapeInputForStaticShape(
                                            reshaped, reshaped_shape));
        } else {
          if (!reshaped.CopyFrom(inputs[i], reshaped_shape)) {
            return absl::InternalError("Failed to reshape tensor");
          }
        }
      }
      xla::ifrt::LayoutRef layout_ref =
          executable_bundle->compiled->xla_input_layouts[i];

      xla::ifrt::ShardingRef ifrt_sharding =
          use_portable_execution
              ? portable_sharding
              : executable_bundle->compiled->arg_ifrt_shardings[i];

      input_handles.push_back(
          {.tensor = reshaped,
           .ifrt_dtype = executable_bundle->compiled->ifrt_input_dtypes[i],
           .ifrt_shape = executable_bundle->compiled->ifrt_input_shapes[i],
           .input_xla_shape = executable_bundle->compiled->xla_input_shapes[i],
           .device_list = device_list,
           .ifrt_sharding = std::move(ifrt_sharding),
           .xla_input_layout = std::move(layout_ref),
           .byte_strides = executable_bundle->compiled->byte_strides[i]});
      input_handle_result_indices.push_back(i);
    }
  }

  TF_ASSIGN_OR_RETURN(auto input_futures,
                      user_inputs_h2d_transfer_executor->ScheduledH2DTransfers(
                          absl::MakeSpan(input_handles), thread_pool_));
  TF_RETURN_IF_ERROR(user_inputs_h2d_transfer_executor->RunH2DTransfers());

  std::vector<xla::ifrt::ArrayRef> transfer_result;
  transfer_result.resize(inputs.size());

  for (int i = 0; i < variable_args.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto array_ref, variable_args[i].Await());
    transfer_result[variable_arg_indices[i]] = std::move(array_ref);
  }

  TF_ASSIGN_OR_RETURN(auto input_arrays, input_futures.Await());
  if (input_arrays.size() != input_handles.size()) {
    return absl::InternalError(absl::StrCat("Expected ", input_handles.size(),
                                            " input arrays but got ",
                                            input_arrays.size()));
  }

  for (int i = 0; i < input_arrays.size(); ++i) {
    transfer_result[input_handle_result_indices[i]] =
        std::move(input_arrays[i]);
  }

  VLOG(2) << "Start Execution";

  std::optional<xla::ifrt::DeviceListRef> execution_device_list;
  if (UsePortableExecution()) {
    execution_device_list = device_list;
  }
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult execution_result,
      [&]() -> absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult> {
        tsl::profiler::TraceMe traceme("Execute");
        return executable_bundle->compiled->ifrt_executable->Execute(
            absl::MakeSpan(transfer_result), execute_options_,
            std::move(execution_device_list));
      }());

  if (executable_bundle->compiled->compile_metadata.retvals().size() !=
      execution_result.outputs.size()) {
    return absl::InternalError(absl::StrCat(
        "Expect ",
        executable_bundle->compiled->compile_metadata.retvals().size(),
        " but got ", execution_result.outputs.size(), " outputs"));
  }

  return ExecutionInfo{
      .execution_result = std::move(execution_result),
      .executable_bundle = std::move(executable_bundle),
      .device_list = std::move(device_list),
      .transfer_result = std::move(transfer_result),
      .device_reservation = std::make_shared<tsl::DeviceReservation>(
          std::move(device_reservation)),
  };
}

absl::StatusOr<std::vector<tensorflow::Tensor>> IfrtServingExecutable::Execute(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices) {
  TF_ASSIGN_OR_RETURN(ExecutionInfo exec_info,
                      ExecuteCore(inputs, variable_arg_indices));

  TF_RETURN_IF_ERROR(exec_info.execution_result.status.Await());

  std::vector<tsl::Future<tensorflow::Tensor>> output_futures;
  output_futures.reserve(exec_info.execution_result.outputs.size());
  for (int i = 0; i < exec_info.execution_result.outputs.size(); ++i) {
    output_futures.push_back(MakeTensorFromArray(
        *ifrt_client_, *exec_info.execution_result.outputs[i],
        exec_info.executable_bundle->compiled->retval_hlo_shardings[i],
        exec_info.device_list, thread_pool_));
  }

  tsl::Future<std::vector<tensorflow::Tensor>> joined_outputs =
      tsl::JoinFutures(absl::MakeSpan(output_futures));

  // If there are no outputs, JoinFutures returns an invalid future.
  // Return early to avoid calling Await() on it, which would crash.
  if (!joined_outputs.IsValid()) {
    return std::vector<tensorflow::Tensor>();
  }

  return joined_outputs.Await();
}

absl::StatusOr<tsl::Future<std::vector<tensorflow::Tensor>>>
IfrtServingExecutable::ExecuteAsync(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices) {
  TF_ASSIGN_OR_RETURN(ExecutionInfo exec_info,
                      ExecuteCore(inputs, variable_arg_indices));

  std::vector<tsl::Future<tensorflow::Tensor>> output_futures;
  output_futures.reserve(exec_info.execution_result.outputs.size());
  for (int i = 0; i < exec_info.execution_result.outputs.size(); ++i) {
    output_futures.push_back(MakeTensorFromArray(
        *ifrt_client_, *exec_info.execution_result.outputs[i],
        exec_info.executable_bundle->compiled->retval_hlo_shardings[i],
        exec_info.device_list, thread_pool_));
  }

  if (output_futures.empty()) {
    // Special case for no returns: we must wait for status.
    // Map creates a promise internally, but it's only for this rare case.
    tsl::Future<> exec_status = exec_info.execution_result.status;
    return exec_status.Map<std::vector<tensorflow::Tensor>>(
        [exec_info =
             std::move(exec_info)]() -> std::vector<tensorflow::Tensor> {
          return std::vector<tensorflow::Tensor>();
        });
  }

  tsl::Future<> status_future = std::move(exec_info.execution_result.status);
  tsl::Future<std::vector<tensorflow::Tensor>> final_future =
      tsl::JoinFutures(absl::MakeSpan(output_futures));

  // Fast path: if both are ready, offload cleanup to thread pool immediately.
  // The execution status must still be checked, like the slow path below and
  // `Execute()` do: a failed execution must not be reported as successful just
  // because the output futures are already available.
  if (final_future.IsReady() && status_future.IsReady()) {
    absl::Status status = status_future.Await();
    thread_pool_.Schedule([exec_info = std::move(exec_info)]() mutable {});
    if (!status.ok()) {
      return tsl::Future<std::vector<tensorflow::Tensor>>(std::move(status));
    }
    return final_future;
  }

  return status_future.Map([final_future = std::move(final_future),
                            exec_info = std::move(exec_info),
                            thread_pool = &thread_pool_]() mutable
                               -> tsl::Future<std::vector<tensorflow::Tensor>> {
    return final_future.Map(
        [exec_info = std::move(exec_info),
         thread_pool](std::vector<tensorflow::Tensor> outputs) mutable {
          // Offload cleanup to thread pool to avoid blocking execution
          // thread.
          thread_pool->Schedule(
              [exec_info = std::move(exec_info)]() mutable {});
          return outputs;
        });
  });
}

absl::Status IfrtServingExecutable::AsyncLoadIfrtArray(
    absl::Span<const tensorflow::Tensor> inputs,
    absl::Span<const int> variable_arg_indices,
    const CompiledExecutable& compiled,
    const xla::ifrt::DeviceListRef& devices) {
  if (compiled.xla_input_shapes.size() != inputs.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", compiled.xla_input_shapes.size(),
                     " input shapes, but got ", inputs.size(), " inputs"));
  }
  for (const int i : variable_arg_indices) {
    // Validation for variable inputs is handled upstream in the Execute()
    // method.
    std::string tensor_name = inputs[i].scalar<tsl::tstring>()();
    // TODO(b/339521818): Add test cases for OpSharding on variables.
    VariableDeviceShardingConfig sharding_config{
        .hlo_sharding = compiled.arg_hlo_shardings[i],
    };
    for (xla::ifrt::Device* device : devices->devices()) {
      sharding_config.device_ids.push_back(device->Id().value());
    }
    xla::ifrt::LayoutRef layout_ref = compiled.xla_input_layouts[i];
    std::shared_ptr<const xla::Shape> shape_on_device =
        compiled.xla_input_shapes[i];

    const std::vector<bool>& may_be_donated = compiled.input_may_be_donated;
    // Convert only on positive evidence that the compiled program never
    // donates this input. An empty/short mask means donation info is missing:
    // assume the input may be donated and keep its buffers donatable.
    bool use_undonatable_buffer_converter =
        use_undonatable_buffer_converter_ &&
        static_cast<size_t>(i) < may_be_donated.size() && !may_be_donated[i];

    TF_RETURN_IF_ERROR(
        ifrt_serving::AsyncLoadRestoredTensorAsIfrtLoadedVariable(
            tensor_name, ifrt_client_, thread_pool_,
            ifrt_restore_tensor_registry_, ifrt_loaded_variable_registry_,
            checkpoint_loader_queue_, sharding_config, std::move(layout_ref),
            std::move(shape_on_device), devices,
            use_undonatable_buffer_converter));
  }
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
