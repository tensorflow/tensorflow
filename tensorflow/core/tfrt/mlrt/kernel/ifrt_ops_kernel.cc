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

#include <cstdint>
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
#include "xla/tsl/concurrency/future.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/tfrt/ifrt/checkpoint_loader.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_restore_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/tstring.h"

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

  mlrt::bc::Vector<bool> truncate_in_cast() const {
    return attributes().GetAs<mlrt::bc::Vector<bool>>(1);
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

 private:
  // TODO(b/335247101): Consider exposing this as an option for tuning or
  // dynamically decide it based on the size of the variables.
  static constexpr int kNumRestoreClusters = 4;

  absl::Status InvokeHelper();

  absl::Status ValidateInput();
};

void MlrtIfrtRestoreVariableKernel::Invoke() {
  absl::Status status = InvokeHelper();
  if (!status.ok()) {
    execution_context().Fail(std::move(status));
    return;
  }
}

absl::Status MlrtIfrtRestoreVariableKernel::ValidateInput() {
  if (prefix().tensor().NumElements() != 1) {
    return absl::InvalidArgumentError(
        "The prefix tensor must be a scalar tensor.");
  }
  if (!TensorShapeUtils::IsVector(tensor_names().tensor().shape()) ||
      !TensorShapeUtils::IsVector(shape_and_slices().tensor().shape())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input tensor_names and shape_and_slices "
                     "should be an 1-D tensors, got ",
                     tensor_names().tensor().shape().DebugString(), " and ",
                     shape_and_slices().tensor().shape().DebugString()));
  }

  if (tensor_names().tensor().NumElements() !=
      shape_and_slices().tensor().NumElements()) {
    return absl::InvalidArgumentError(
        "The tensor_names and shape_and_slices tensors must have the same "
        "number of elements.");
  }

  if (tensor_names().tensor().NumElements() != var_handles().size()) {
    return absl::InvalidArgumentError(
        "The tensor_names and var_handles must have the same number of "
        "elements.");
  }
  if (tensor_names().tensor().NumElements() != restored_dtypes().size()) {
    return absl::InvalidArgumentError(
        "The tensor_names and restored_dtypes must have the same number of "
        "elements.");
  }

  if (tensor_names().tensor().NumElements() != truncate_in_cast().size()) {
    return absl::InvalidArgumentError(
        "The tensor_names and truncate_in_cast must have the same number of "
        "elements.");
  }

  return absl::OkStatus();
}

absl::Status MlrtIfrtRestoreVariableKernel::InvokeHelper() {
  std::optional<ifrt_serving::IfrtModelRestoreContext*> model_restore_context =
      context()
          .resource_context()
          .GetResource<ifrt_serving::IfrtModelRestoreContext>(
              ifrt_serving::kIfrtModelRestoreContextName);
  if (!model_restore_context.has_value()) {
    return absl::InternalError(
        "Did not find IfrtModelRestoreContext resource.");
  }
  if (*model_restore_context == nullptr) {
    return absl::InternalError("IfrtModelRestoreContext must not be null.");
  }
  ifrt_serving::CheckpointLoader* checkpoint_loader =
      (*model_restore_context)->checkpoint_loader();
  if (!checkpoint_loader) {
    return absl::InternalError("CheckpointLoader must not be null.");
  }
  TF_RETURN_IF_ERROR(ValidateInput());

  std::vector<tensorflow::DataType> restored_dtypes_vec(
      restored_dtypes().begin(), restored_dtypes().end());
  std::vector<bool> truncate_in_cast_vec(truncate_in_cast().begin(),
                                         truncate_in_cast().end());
  return checkpoint_loader->Load(prefix(), var_handles(), tensor_names(),
                                 shape_and_slices(), restored_dtypes_vec,
                                 truncate_in_cast_vec, context());
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

  bool used_by_host() const {
    DCHECK_EQ(attributes().size(), 1);
    return attributes().GetAs<bool>(0);
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

  auto& resource_handle = variable_handler_tensor().scalar<ResourceHandle>()();
  std::string runtime_name =
      ifrt_serving::GetRuntimeNameFromVarHandle(resource_handle);

  if (used_by_host()) {
    if (ifrt_restore_tensor_registry.SetUsedByHost(runtime_name).ok()) {
      tsl::Future<tensorflow::Tensor> restored_tensor_future =
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
    } else {
      // If not at IfrtRestoreTensorRegistry, try ResourceManager
      auto resource_manager = context()
                                  .fallback_request_state()
                                  .device_manager()
                                  .HostCPU()
                                  ->resource_manager();
      DCHECK(resource_manager);
      Var* variable;
      TF_RETURN_IF_ERROR(resource_manager->Lookup(
          resource_handle.container(), resource_handle.name(), &variable));
      if (tensorflow::Tensor* t = variable->tensor(); t != nullptr) {
        std::move(tensor_promise)
            .Set<tensorflow::tfrt_stub::FallbackTensor>(
                tensorflow::tfrt_stub::FallbackTensor(*t));
      } else {
        std::move(tensor_promise)
            .SetError(absl::InternalError(
                absl::StrCat("Variable ", resource_handle.name(),
                             " is not found in either "
                             "IfrtRestoreTensorRegistry or ResourceManager")));
      }
    }
  } else {
    // If not used by host, set the future to be ready immediately with an empty
    // tensor so that it does not block the graph execution.
    std::move(tensor_promise)
        .Set<tensorflow::tfrt_stub::FallbackTensor>(
            tensorflow::tfrt_stub::FallbackTensor());
  }
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
