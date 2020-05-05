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
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

namespace {

bool IsCPU(tensorflow::VariantDevice variant) {
  if (VariantDeviceIsCustom(variant)) {
    return false;
  }
  tensorflow::Device* d = absl::get<tensorflow::Device*>(variant);
  return d == nullptr || d->tensorflow_gpu_device_info() == nullptr;
}

}  // namespace

namespace tensorflow {

// TODO(b/152902651): This should not depend on EagerContext. This can be
// resolved by storing ctx->HostCPU() in the TensorHandle class.
AbstractTensorInterface* TensorHandle::Resolve(Status* status) {
  if (VariantDeviceIsCustom(device())) {
    auto* custom_device = absl::get<CustomDevice*>(device());
    TensorHandle* copy;
    *status = custom_device->CopyTensorFromDevice(
        this, "/job:localhost/replica:0/task:0/device:CPU:0", &copy);
    if (status->ok()) {
      return copy->Resolve(status);
    } else {
      return nullptr;
    }
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
        "Resolve() is not supoorted on packed TensorHandles.");
    return nullptr;
  }
}

AbstractTensorHandleInterface* EagerContext::CopyTensorHandleToDevice(
    AbstractTensorHandleInterface* handle, const char* device_name,
    Status* status) {
  TensorHandle* input = TensorHandleFromInterface(handle);
  TensorHandle* result = nullptr;
  Device* device;
  *status = this->FindDeviceFromName(device_name, &device);
  if (!status->ok()) {
    tensorflow::CustomDevice* dev;
    *status = this->FindCustomDeviceFromName(device_name, &dev);
    if (status->ok()) {
      *status = dev->CopyTensorToDevice(input, &result);
      if (status->ok()) {
        return result;
      }
    }
    return nullptr;
  }
  // Handle tensor handles currently in custom devices
  const char* handle_device_name = input->DeviceName(status);
  if (!status->ok()) {
    return nullptr;
  }
  tensorflow::CustomDevice* dev;
  *status = this->FindCustomDeviceFromName(handle_device_name, &dev);
  if (status->ok()) {
    *status = dev->CopyTensorFromDevice(input, device_name, &result);
    if (status->ok()) {
      return result;
    }
    return nullptr;
  }

  // Handle regular case.
  *status =
      EagerCopyToDevice(input, this, &this->Executor(), device, false, &result);
  if (status->ok()) {
    return result;
  }
  return nullptr;
}

// TODO(b/152902651): We unfortunately need to put this EagerContext function
// here to a circular BUILD dep issue. If we move this to context.cc, then we
// will have the circular dependency of:
//   context -> tensor_handle -> remote_tensor_handle_data -> context
AbstractTensorHandleInterface* EagerContext::CreateLocalHandle(
    AbstractTensorInterface* t) {
  Tensor tensor = TensorFromInterface(t);
  return TensorHandle::CreateLocalHandle(std::move(tensor), /*d=*/HostCPU(),
                                         /*op_device=*/nullptr, this);
}

// TODO(b/152902651): We have to keep this function here since EagerOperation
// depends on EagerContext. Thus, the context build target can't depend on
// EagerOperation.
AbstractOperationInterface* EagerContext::CreateOperation() {
  return new EagerOperation(this);
}

// TODO(b/152902651): Once we move many execute.cc functions into
// eager_operation.cc we can avoid a circular dependency between them.
Status EagerOperation::Execute(
    absl::Span<AbstractTensorHandleInterface*> retvals, int* num_retvals) {
  return EagerExecute(
      this, reinterpret_cast<tensorflow::TensorHandle**>(retvals.data()),
      num_retvals);
}

}  //  namespace tensorflow
