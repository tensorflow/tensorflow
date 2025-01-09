/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/ifrt/checkpoint_loader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/future.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel_runner_utils.h"
#include "tensorflow/core/tfrt/mlrt/kernel/shard_restore_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tstring.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

namespace {

static constexpr int kNumRestoreClusters = 4;

// A shard of variables to be restored.
struct RestoreVariableShard {
  tensorflow::Tensor prefix;
  tensorflow::Tensor tensor_names;
  tensorflow::Tensor shape_and_slices;
  std::vector<tensorflow::tfrt_stub::FallbackTensor> var_handles;
  tensorflow::AttrValue dtypes_attr_value;
  std::vector<tensorflow::DataType> restored_dtypes;
  std::vector<bool> truncate_in_cast;
};

struct AsyncState {
  explicit AsyncState(
      const std::vector<tensorflow::TensorValue>& input_tf_tensor_values,
      const OpKernelContext::Params& params, int num_outputs,
      const tensorflow::DeviceMgr& device_manager,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime)
      : run_state(input_tf_tensor_values, params),
        context(&run_state.params, num_outputs),
        device_manager(device_manager),
        process_function_library_runtime(process_function_library_runtime) {}

  tfrt_stub::OpKernelRunState run_state;
  OpKernelContext context;
  const tensorflow::DeviceMgr& device_manager;
  const tensorflow::ProcessFunctionLibraryRuntime&
      process_function_library_runtime;

  std::vector<xla::ifrt::Promise<tensorflow::Tensor>> results;
};

// Returns a casted tensor if successful.
absl::StatusOr<tensorflow::Tensor> Cast(
    tensorflow::Tensor& in_tensor, tensorflow::DataType restored_dtype,
    tensorflow::DataType cast_dtype, bool truncate_in_cast,
    const tensorflow::DeviceMgr& device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime,
    OpKernelContext::Params& params) {
  auto runner =
      tfrt_stub::OpKernelRunner::Create(
          /*op_name=*/
          "Cast", /*node_name=*/"Cast", params.device->name(),
          /*num_args=*/1,
          [&](tensorflow::AttrValueMap* attr_value_map) {
            tensorflow::AttrValue restored_dtype_attr_value;
            restored_dtype_attr_value.set_type(restored_dtype);
            attr_value_map->insert({"SrcT", restored_dtype_attr_value});

            tensorflow::AttrValue cast_dtype_attr_value;
            cast_dtype_attr_value.set_type(cast_dtype);
            attr_value_map->insert({"DstT", cast_dtype_attr_value});

            tensorflow::AttrValue truncate_attr_value;
            truncate_attr_value.set_b(truncate_in_cast);
            attr_value_map->insert({"Truncate", truncate_attr_value});
            return absl::OkStatus();
          },
          device_manager, process_function_library_runtime)
          .value();

  std::vector<tensorflow::TensorValue> input_tf_tensor_values;
  input_tf_tensor_values.push_back(tensorflow::TensorValue(&in_tensor));

  tf_mlrt::SetUpParams(runner, input_tf_tensor_values, params);
  // Use persistent device instead of the per request device.

  OpKernelContext op_kernel_context(&params, /*num_outputs=*/1);

  runner.Run(&op_kernel_context);

  if (!op_kernel_context.status().ok()) {
    return op_kernel_context.status();
  }
  DCHECK_EQ(op_kernel_context.num_outputs(), 1);
  return *(op_kernel_context.mutable_output(0));
}

void RunShardHelper(const tfrt_stub::OpKernelRunner& runner,
                    AsyncState* async_state, RestoreVariableShard shard) {
  // Keep input tensor alive in `shard`.
  auto* op_kernel_context_ptr = &async_state->context;
  runner.Run(op_kernel_context_ptr);

  auto& op_kernel_context = async_state->context;
  if (!op_kernel_context.status().ok()) {
    for (auto& result : async_state->results) {
      std::move(result).Set(op_kernel_context.status());
    }
    return;
  }
  DCHECK_EQ(shard.var_handles.size(), op_kernel_context.num_outputs());
  DCHECK_EQ(shard.truncate_in_cast.size(), op_kernel_context.num_outputs());

  // TODO(b/343964091): consider to run multiple casts in parallel.
  for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
    DCHECK(op_kernel_context.mutable_output(i));

    if (op_kernel_context.mutable_output(i)->dtype() !=
        shard.restored_dtypes[i]) {
      std::move(async_state->results[i])
          .Set(absl::InvalidArgumentError(
              absl::StrCat("The restored tensor has a different dtype than the "
                           "variable handle: ",
                           op_kernel_context.mutable_output(i)->dtype(),
                           " vs. ", shard.restored_dtypes[i])));
      return;
    }
    const ResourceHandle& var_handle =
        shard.var_handles[i].tensor().scalar<tensorflow::ResourceHandle>()();

    if (shard.restored_dtypes[i] == var_handle.dtypes_and_shapes()[0].dtype) {
      std::move(async_state->results[i])
          .Set(*std::move(op_kernel_context.mutable_output(i)));
    } else {
      absl::StatusOr<tensorflow::Tensor> cast_output =
          Cast(*op_kernel_context.mutable_output(i), shard.restored_dtypes[i],
               var_handle.dtypes_and_shapes()[0].dtype,
               shard.truncate_in_cast[i], async_state->device_manager,
               async_state->process_function_library_runtime,
               async_state->run_state.params);
      if (!cast_output.ok()) {
        std::move(async_state->results[i]).Set(cast_output.status());
      } else {
        std::move(async_state->results[i]).Set(*std::move(cast_output));
      }
    }
  }
}

absl::Status RunShard(RestoreVariableShard shard,
                      IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry,
                      tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue,
                      tf_mlrt::Context& context, bool use_async_restore) {
  if (!ifrt_restore_tensor_registry) {
    return absl::InternalError("ifrt_restore_tensor_registry must not be null");
  }
  if (!checkpoint_loader_work_queue) {
    return absl::InternalError("checkpoint_loader_work_queue must not be null");
  }
  const int num_outputs = shard.var_handles.size();
  DCHECK_EQ(num_outputs, shard.tensor_names.NumElements());
  auto& fallback_request_state = context.fallback_request_state();

  // Use `tf.RestoreV2` to restore tensor. This will also populate
  // tensorflow::ResourceManager.
  // TODO(b/319045348): avoid populating tensorflow::ResourceManager if the
  // variable is only used by device/IFRT.
  // TODO(b/319045348): consider directly calling restore function such as that
  // in /tensorflow/core/kernels/save_restore_v2_ops.cc
  auto runner =
      tfrt_stub::OpKernelRunner::Create(
          /*op_name=*/
          "RestoreV2", /*node_name=*/"RestoreV2",
          context.params().device->name(),
          /*num_args=*/3,
          [&](tensorflow::AttrValueMap* attr_value_map) {
            attr_value_map->insert({"dtypes", shard.dtypes_attr_value});
            return absl::OkStatus();
          },
          fallback_request_state.device_manager(),
          fallback_request_state.process_function_library_runtime())
          .value();

  // Prepare the input tensors.
  std::vector<tensorflow::TensorValue> input_tf_tensor_values;
  static constexpr int kNumInputArgs = 3;
  input_tf_tensor_values.resize(kNumInputArgs);
  // We need to keep these tensor alive
  input_tf_tensor_values[0].tensor = &shard.prefix;
  input_tf_tensor_values[1].tensor = &shard.tensor_names;
  input_tf_tensor_values[2].tensor = &shard.shape_and_slices;

  auto& params = context.params();
  tf_mlrt::SetUpParams(runner, input_tf_tensor_values, params);
  // Use persistent device instead of the per request device.
  params.device = context.fallback_request_state().device_manager().HostCPU();

  auto async_state = std::make_unique<AsyncState>(
      input_tf_tensor_values, params, num_outputs,
      fallback_request_state.device_manager(),
      fallback_request_state.process_function_library_runtime());

  for (int i = 0; i < num_outputs; ++i) {
    auto promise = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
    auto future = xla::ifrt::Future<tensorflow::Tensor>(promise);
    const ResourceHandle& var_handle =
        shard.var_handles[i].tensor().scalar<tensorflow::ResourceHandle>()();

    TF_ASSIGN_OR_RETURN(ifrt_serving::DtypeAndShape dtype_and_shape,
                        ifrt_serving::GetDtypeAndShape(var_handle));

    std::string runtime_name =
        ifrt_serving::GetRuntimeNameFromVarHandle(var_handle);

    ifrt_serving::IfrtRestoreTensorRegistry::RestoredTensorInfo
        restored_tensor_info = {false, std::move(dtype_and_shape),
                                std::move(future)};
    if (auto status = ifrt_restore_tensor_registry->TryRegister(
            runtime_name, restored_tensor_info);
        !status.ok()) {
      // Propagate errors so that if already-registered futures are being waited
      // on, they can be unblocked.
      for (auto& result : async_state->results) {
        std::move(result).Set(status);
      };
      return status;
    }
    async_state->results.push_back(std::move(promise));
  }
  // Run the shard synchronously.
  if (!use_async_restore) {
    RunShardHelper(runner, async_state.get(), shard);
  } else {
    tensorflow::Context bg_context(tensorflow::ContextKind::kThread);
    // Use dedicated work queue for restore operation.
    checkpoint_loader_work_queue->AddTask(
        [runner = std::move(runner), async_state = std::move(async_state),
         shard = std::move(shard), bg_context = std::move(bg_context)]() {
          tensorflow::WithContext wc(bg_context);
          RunShardHelper(runner, async_state.get(), shard);
        });
  }

  return absl::OkStatus();
}

int64_t GetSizeFromVarHandle(const ResourceHandle& handle) {
  int size = 0;
  for (auto& dtype_and_shape : handle.dtypes_and_shapes()) {
    size += DataTypeSize(dtype_and_shape.dtype) *
            dtype_and_shape.shape.num_elements();
  }
  return size;
}

}  // namespace

absl::Status CheckpointLoader::PrepareRestore(const PrepareRestoreArgs& args) {
  VLOG(1) << "Skip CheckpointLoader::PrepareRestore";
  return absl::OkStatus();
}

absl::Status CheckpointLoader::Load(
    const tensorflow::tfrt_stub::FallbackTensor& prefix,
    const std::vector<tensorflow::tfrt_stub::FallbackTensor>& var_handles,
    const tensorflow::tfrt_stub::FallbackTensor& tensor_names,
    const tensorflow::tfrt_stub::FallbackTensor& shape_and_slices,
    absl::Span<const tensorflow::DataType> restored_dtypes,
    const std::vector<bool>& truncate_in_cast, tf_mlrt::Context& context) {
  std::vector<int64_t> variable_sizes;
  variable_sizes.reserve(var_handles.size());
  for (auto& handle : var_handles) {
    variable_sizes.push_back(GetSizeFromVarHandle(
        handle.tensor().scalar<tensorflow::ResourceHandle>()()));
  }

  std::vector<std::vector<int>> sharded_indices = tf_mlrt::ShardVariables(
      kNumRestoreClusters, absl::MakeSpan(variable_sizes));

  // Converts the names and slices back to the tensor.
  auto vector_to_tensor = [](const std::vector<tsl::tstring>& vec) {
    tensorflow::Tensor tensor(tensorflow::DT_STRING,
                              TensorShape({static_cast<int>(vec.size())}));
    for (int i = 0; i < vec.size(); ++i) {
      tensor.flat<tsl::tstring>()(i) = vec[i];
    }
    return tensor;
  };

  const auto& tensor_names_flat = tensor_names.tensor().flat<tsl::tstring>();
  const auto& shape_and_slices_flat =
      shape_and_slices.tensor().flat<tsl::tstring>();

  std::vector<RestoreVariableShard> shards;
  shards.reserve(sharded_indices.size());
  for (auto& sharded_index : sharded_indices) {
    RestoreVariableShard shard;
    shard.var_handles.reserve(sharded_index.size());
    shard.truncate_in_cast.reserve(sharded_index.size());
    shard.restored_dtypes.reserve(sharded_index.size());
    std::vector<tsl::tstring> tensor_names;
    std::vector<tsl::tstring> shape_and_slices;
    shape_and_slices.reserve(sharded_index.size());
    tensor_names.reserve(sharded_index.size());
    for (int index : sharded_index) {
      tensor_names.push_back(tensor_names_flat(index));
      shape_and_slices.push_back(shape_and_slices_flat(index));
      shard.dtypes_attr_value.mutable_list()->add_type(restored_dtypes[index]);
      shard.var_handles.push_back(var_handles[index]);
      shard.restored_dtypes.push_back(restored_dtypes[index]);
      shard.truncate_in_cast.push_back(truncate_in_cast[index]);
    }
    shard.prefix = prefix.tensor();
    shard.tensor_names = vector_to_tensor(tensor_names);
    shard.shape_and_slices = vector_to_tensor(shape_and_slices);
    shards.push_back(std::move(shard));
  }
  for (const auto& shard : shards) {
    TF_RETURN_IF_ERROR(RunShard(shard, ifrt_restore_tensor_registry_,
                                checkpoint_loader_work_queue_, context,
                                use_async_restore_));
  }
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
