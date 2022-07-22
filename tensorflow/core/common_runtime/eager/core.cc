/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/platform/errors.h"

namespace {

bool IsCPU(tensorflow::Device* d) {
  return d == nullptr || d->tensorflow_accelerator_device_info() == nullptr;
}

}  // namespace

namespace tensorflow {

// TODO(b/152902651): This should not depend on EagerContext. This can be
// resolved by storing ctx->HostCPU() in the TensorHandle class.
AbstractTensorInterface* TensorHandle::Resolve(Status* status) {
  *status = WaitUnknownDevice();
  if (!status->ok()) {
    return nullptr;
  }
  if (Type() == REMOTE) {
    const tensorflow::Tensor* t = nullptr;
    TensorHandle* h_cpu = nullptr;
    *status = EagerCopyToDevice(this, ctx_, &ctx_->Executor(), ctx_->HostCPU(),
                                false, &h_cpu);
    if (!status->ok()) {
      return nullptr;
    }
    *status = h_cpu->Tensor(&t);
    if (!status->ok()) {
      h_cpu->Unref();
      return nullptr;
    }
    // TODO(b/153052876): Change TF_TensorFromTensor to just return an
    // AbstractTensorInterface
    TF_Tensor* tf_tensor = TF_TensorFromTensor(*t, status);
    AbstractTensorInterface* retval = tf_tensor->tensor;
    h_cpu->Unref();
    delete tf_tensor;
    return retval;
  } else if (Type() == LOCAL) {
    tensorflow::Tensor tensor;
    if (IsCPU(device()) || HasLocalMirror(nullptr)) {
      const tensorflow::Tensor* src = nullptr;
      if (HasLocalMirror(nullptr)) {
        *status = TensorFromDevice(nullptr, &src);
      } else {
        *status = Tensor(&src);
      }
      if (!status->ok()) return nullptr;

      tensor = *src;
    } else {
      *status = CopyToDevice(*ctx_, ctx_->HostCPU(), &tensor);
      if (!status->ok()) return nullptr;

      tensorflow::Tensor mirror = tensor;
      *status = AddLocalMirror(std::move(mirror), nullptr);
      if (!status->ok()) {
        // If a mirror was added since we called HasLocalMirror then drop the
        // newly copied tensor and use the previously added mirror.
        if (status->code() != error::Code::ALREADY_EXISTS) {
          return nullptr;
        }
        const tensorflow::Tensor* src = nullptr;
        *status = TensorFromDevice(nullptr, &src);
        if (!status->ok()) return nullptr;

        tensor = *src;
      }
    }
    // TODO(b/153052876): Change TF_TensorFromTensor to just return an
    // AbstractTensorInterface
    TF_Tensor* tf_tensor = TF_TensorFromTensor(tensor, status);
    AbstractTensorInterface* retval = tf_tensor->tensor;
    delete tf_tensor;
    return retval;
  } else {
    *status = errors::InvalidArgument(
        "Resolve() is not supported on packed TensorHandles.");
    return nullptr;
  }
}

ImmediateExecutionTensorHandle* EagerContext::CopyTensorHandleToDevice(
    ImmediateExecutionTensorHandle* handle, const char* device_name,
    Status* status) {
  ImmediateExecutionTensorHandle* result = nullptr;
  Device* device;
  *status = this->FindDeviceFromName(device_name, &device);
  if (!status->ok()) {
    *status =
        tensorflow::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }

  TensorHandle* input = TensorHandleFromInterface(handle);
  *status =
      EagerCopyToDevice(input, this, &this->Executor(), device, false,
                        reinterpret_cast<tensorflow::TensorHandle**>(&result));
  if (status->ok()) {
    return result;
  }
  return nullptr;
}

// TODO(b/152902651): We unfortunately need to put this EagerContext function
// here to a circular BUILD dep issue. If we move this to context.cc, then we
// will have the circular dependency of:
//   context -> tensor_handle -> remote_tensor_handle_data -> context
ImmediateExecutionTensorHandle* EagerContext::CreateLocalHandle(
    AbstractTensorInterface* t) {
  Tensor tensor = TensorFromInterface(t);
  return TensorHandle::CreateLocalHandle(std::move(tensor), /*d=*/HostCPU(),
                                         /*op_device=*/nullptr, this);
}

ImmediateExecutionTensorHandle* EagerContext::CreateLocalHandleFromTFTensor(
    tensorflow::Tensor& t, const char* d_name) {
  // If device name is not specified, create the TensorHandle on host cpu.
  if (d_name == nullptr)
    return TensorHandle::CreateLocalHandle(std::move(t), /*d=*/HostCPU(),
                                           /*op_device=*/nullptr, this);
  Device* d = nullptr;
  auto status = FindDeviceFromName(d_name, &d);
  if (!status.ok()) return nullptr;
  return TensorHandle::CreateLocalHandle(std::move(t), /*d=*/d,
                                         /*op_device=*/nullptr, this);
}

ImmediateExecutionTensorHandle* EagerContext::TFTensorHandleFromInterface(
    ImmediateExecutionTensorHandle* handle) {
  return handle;
}

// TODO(b/152902651): We have to keep this function here since EagerOperation
// depends on EagerContext. Thus, the context build target can't depend on
// EagerOperation.
ImmediateExecutionOperation* EagerContext::CreateOperation() {
  return new EagerOperation(this);
}

Status EagerContext::RegisterFunction(AbstractFunction* f) {
  FunctionDef* fdef;
  TF_RETURN_IF_ERROR(f->GetFunctionDef(&fdef));
  if (!fdef) {
    return errors::InvalidArgument("GetFunctionDef returned nullptr.");
  }
  return AddFunctionDef(*fdef);
}

// TODO(b/152902651): Once we move many execute.cc functions into
// eager_operation.cc we can avoid a circular dependency between them.
Status EagerOperation::Execute(absl::Span<AbstractTensorHandle*> retvals,
                               int* num_retvals) {
  for (ImmediateExecutionTensorHandle* handle : inputs_) {
    if (TensorHandle::classof(handle)) {
      TF_RETURN_IF_ERROR(down_cast<TensorHandle*>(handle)->WaitUnknownDevice());
    }
  }

  // Run eager placement logic.
  class Device* device = absl::get<class Device*>(Device());
  if (device == nullptr) {
    TF_RETURN_IF_ERROR(eager::MaybePinToResourceDevice(&device, *this));
  }
  if (device == nullptr && ctx_.PinSmallOpsToCPU()) {
    bool pin_to_cpu;
    TF_RETURN_IF_ERROR(eager::MaybePinSmallOpsToCpu(
        &pin_to_cpu, Name(), GetInputs(), ctx_.HostCPU()->name()));
    if (pin_to_cpu) {
      device = ctx_.HostCPU();
    }
  }

  if (device != nullptr) {
    SetDevice(device);
  }
  // At this point all inputs and outputs are TensorHandles associated with
  // physical devices.
  tensorflow::TensorHandle** retval_array =
      reinterpret_cast<tensorflow::TensorHandle**>(retvals.data());
  return EagerExecute(this, retval_array, num_retvals);
}

}  //  namespace tensorflow
