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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/future.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel_runner_utils.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/tstring.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

using tensorflow::ifrt_serving::IfrtModelContext;

namespace tensorflow {
namespace tf_mlrt {

namespace {

struct MlrtIfrtRestoreVariableKernel : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.ifrt_restore_variable";

  tensorflow::tfrt_stub::FallbackTensor prefix() const {
    DCHECK_GT(arguments().size(), 3);
    return arguments()[0].Get<tensorflow::tfrt_stub::FallbackTensor>();
  }
  tensorflow::tfrt_stub::FallbackTensor tensor_names() const {
    DCHECK_GT(arguments().size(), 3);
    return arguments()[1].Get<tensorflow::tfrt_stub::FallbackTensor>();
  }
  tensorflow::tfrt_stub::FallbackTensor shape_and_slices() const {
    DCHECK_GT(arguments().size(), 3);
    return arguments()[2].Get<tensorflow::tfrt_stub::FallbackTensor>();
  }

  mlrt::bc::Vector<tensorflow::DataType> restored_dtypes() const {
    return attributes().GetAs<mlrt::bc::Vector<tensorflow::DataType>>(0);
  }

  std::vector<tensorflow::tfrt_stub::FallbackTensor> var_handles() const {
    DCHECK_GT(arguments().size(), 3);
    std::vector<tensorflow::tfrt_stub::FallbackTensor> result;
    result.reserve(arguments().size() - 3);
    for (int i = 3; i < arguments().size(); ++i) {
      result.push_back(
          arguments()[i].Get<tensorflow::tfrt_stub::FallbackTensor>());
    }
    return result;
  }

  Context& context() { return execution_context().GetUserContext<Context>(); }
  void Invoke();
};

void MlrtIfrtRestoreVariableKernel::Invoke() {
  std::optional<IfrtModelContext*> ifrt_model_context =
      context().resource_context().GetResource<IfrtModelContext>(
          "IfrtModelContext");
  if (!ifrt_model_context.has_value()) {
    execution_context().Fail(absl::FailedPreconditionError(
        "RestoreVariableOp: failed to fetch IfrtModelContext"));
    return;
  }
  const int num_outputs = var_handles().size();
  DCHECK_EQ(num_outputs, tensor_names().tensor().NumElements());
  auto& fallback_request_state = context().fallback_request_state();
  tensorflow::AttrValue dtypes_attr_value;
  for (const auto& dtype : restored_dtypes()) {
    dtypes_attr_value.mutable_list()->mutable_type()->Add(dtype);
  }
  // Use `tf.RestoreV2` to restore tensor. This will also populate
  // tensorflow::ResourceManager.
  // TODO(b/319045348): avoid populating tensorflow::ResourceManager if the
  // variable is only used by device/IFRT.
  // TODO(b/319045348): consider directly calling restore function such as that
  // in /tensorflow/core/kernels/save_restore_v2_ops.cc
  auto runner = tfrt_stub::OpKernelRunner::Create(
                    /*op_name=*/
                    "RestoreV2", /*node_name=*/"RestoreV2",
                    context().params().device->name(),
                    /*num_args=*/3,
                    [&](tensorflow::AttrValueMap* attr_value_map) {
                      attr_value_map->insert({"dtypes", dtypes_attr_value});
                      return absl::OkStatus();
                    },
                    fallback_request_state.device_manager(),
                    fallback_request_state.process_function_library_runtime())
                    .value();

  // Prepare the input tensors.
  std::vector<tensorflow::TensorValue> input_tf_tensor_values;
  input_tf_tensor_values.resize(arguments().size());
  for (int i = 0; i < arguments().size(); ++i) {
    auto& fallback_tensor =
        arguments()[i].Get<tensorflow::tfrt_stub::FallbackTensor>();
    input_tf_tensor_values[i].tensor = &fallback_tensor.tensor();
  }

  auto& params = context().params();
  SetUpParams(runner, input_tf_tensor_values, params);

  struct AsyncState {
    explicit AsyncState(
        const std::vector<tensorflow::TensorValue>& input_tf_tensor_values,
        const OpKernelContext::Params& params, int num_outputs)
        : run_state(input_tf_tensor_values, params),
          context(&run_state.params, num_outputs) {}

    tfrt_stub::OpKernelRunState run_state;
    OpKernelContext context;
    std::vector<xla::ifrt::Promise<tensorflow::Tensor>> results;
  };
  auto async_state =
      std::make_unique<AsyncState>(input_tf_tensor_values, params, num_outputs);

  async_state->results.reserve(num_outputs);

  ifrt_serving::IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry =
      (*ifrt_model_context)->GetRestoreTensorRegistry();
  for (int i = 0; i < num_outputs; ++i) {
    auto promise = xla::ifrt::Future<tensorflow::Tensor>::CreatePromise();
    auto future = xla::ifrt::Future<tensorflow::Tensor>(promise);
    const ResourceHandle& var_handle =
        var_handles()[i].tensor().scalar<ResourceHandle>()();
    absl::StatusOr<ifrt_serving::DtypeAndShape> dtype_and_shape =
        ifrt_serving::GetDtypeAndShape(var_handle);
    if (!dtype_and_shape.ok()) {
      // TODO(b/330360798) Refactor Invoke() to have less usage on
      // execution_context().Fail.
      execution_context().Fail(dtype_and_shape.status());
      return;
    }
    std::string runtime_name =
        ifrt_serving::GetRuntimeNameFromVarHandle(var_handle);
    ifrt_serving::IfrtRestoreTensorRegistry::RestoredTensorInfo
        restored_tensor_info = {false, *std::move(dtype_and_shape),
                                std::move(future)};
    if (auto status = ifrt_restore_tensor_registry.TryRegister(
            runtime_name, restored_tensor_info);
        !status.ok()) {
      // Propagate errors so that if already-registered futures are being waited
      // on, they can be unblocked.
      for (auto& result : async_state->results) {
        std::move(result).Set(status);
      }
      execution_context().Fail(std::move(status));
      return;
    }
    async_state->results.push_back(std::move(promise));
  }

  // Use dedicated work queue for restore operation.
  DCHECK((*ifrt_model_context)->checkpoint_loader_queue() != nullptr);
  (*ifrt_model_context)
      ->checkpoint_loader_queue()
      ->AddTask(
          [runner = std::move(runner), async_state = std::move(async_state)]() {
            auto* op_kernel_context_ptr = &async_state->context;
            runner.Run(op_kernel_context_ptr);

            auto& op_kernel_context = async_state->context;
            if (!op_kernel_context.status().ok()) {
              for (auto& result : async_state->results) {
                std::move(result).Set(op_kernel_context.status());
              }
              return;
            }
            for (int i = 0; i < op_kernel_context.num_outputs(); ++i) {
              DCHECK(op_kernel_context.mutable_output(i));
              std::move(async_state->results[i])
                  .Set(std::move(*op_kernel_context.mutable_output(i)));
            }
          });
}

class MlrtIfrtLoadVariableKernel : public mlrt::KernelFrame {
 public:
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.ifrt_load_variable";

  const tensorflow::Tensor& variable_handler_tensor() const {
    DCHECK_GE(arguments().size(), 1);
    const tensorflow::Tensor& ret =
        arguments()[0].Get<tensorflow::tfrt_stub::FallbackTensor>().tensor();
    DCHECK_EQ(ret.NumElements(), 1);
    return ret;
  }

  absl::string_view sharding_config_proto_text() const {
    DCHECK_EQ(attributes().size(), 3);
    return attributes().GetAs<mlrt::bc::String>(0).Get();
  }

  bool used_by_host() const {
    DCHECK_EQ(attributes().size(), 3);
    return attributes().GetAs<bool>(2);
  }

  Context& context() { return execution_context().GetUserContext<Context>(); }
  void Invoke();

 private:
  absl::Status InvokeHelper();
};

void MlrtIfrtLoadVariableKernel::Invoke() {
  absl::Status status = InvokeHelper();
  if (!status.ok()) {
    execution_context().Fail(std::move(status));
    return;
  }
}

absl::Status MlrtIfrtLoadVariableKernel::InvokeHelper() {
  DCHECK_EQ(2, results().size());
  std::optional<IfrtModelContext*> ifrt_model_context =
      context().resource_context().GetResource<IfrtModelContext>(
          "IfrtModelContext");
  if (!ifrt_model_context.has_value()) {
    return absl::FailedPreconditionError(
        "LoadVariableOp: failed to fetch IfrtModelContext: ");
  }
  auto tensor_promise =
      mlrt::Promise::Allocate<tensorflow::tfrt_stub::FallbackTensor>();
  auto tensor_future = tensor_promise.GetFuture();

  ifrt_serving::IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry =
      (*ifrt_model_context)->GetRestoreTensorRegistry();

  std::string runtime_name = ifrt_serving::GetRuntimeNameFromVarHandle(
      variable_handler_tensor().scalar<ResourceHandle>()());

  if (used_by_host()) {
    TF_RETURN_IF_ERROR(
        ifrt_restore_tensor_registry.SetUsedByHost(runtime_name));
  }

  xla::ifrt::Future<tensorflow::Tensor> restored_tensor_future =
      ifrt_restore_tensor_registry.GetRestoredTensor(runtime_name);

  restored_tensor_future.OnReady(
      [tensor_promise = std::move(tensor_promise)](
          absl::StatusOr<tensorflow::Tensor> restored_tensor) mutable {
        if (!restored_tensor.ok()) {
          std::move(tensor_promise).SetError(restored_tensor.status());
          return;
        }
        std::move(tensor_promise)
            .Set<tensorflow::tfrt_stub::FallbackTensor>(
                tensorflow::tfrt_stub::FallbackTensor(*restored_tensor));
      });
  // Return the name as the key
  tensorflow::Tensor key_tensor(tensorflow::DT_STRING, {});
  key_tensor.scalar<tsl::tstring>()() = runtime_name;
  results()[0].Set(tensorflow::tfrt_stub::FallbackTensor(key_tensor));
  results()[1].Set(std::move(tensor_future));
  return absl::OkStatus();
}

void RegisterTfMlrtIfrtKernels(mlrt::KernelRegistry& registry) {
  registry.Register<MlrtIfrtLoadVariableKernel>();
  registry.Register<MlrtIfrtRestoreVariableKernel>();
}

}  // namespace

const bool kUnused = [] {
  RegisterTfMlrtIfrtKernels(GetTfMlrtOptionalKernelRegistry());
  return true;
}();

}  // namespace tf_mlrt
}  // namespace tensorflow
