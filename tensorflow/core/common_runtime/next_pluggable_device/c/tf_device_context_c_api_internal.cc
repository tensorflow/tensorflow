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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api_internal.h"

#include <cstring>
#include <functional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/status.h"

namespace tensorflow {

namespace {

TF_StatusCallback* ToC(tsl::StatusCallback callback) {
  TF_StatusCallback* c_callback = new TF_StatusCallback();
  tsl::StatusCallback* callback_ptr =
      new tsl::StatusCallback(std::move(callback));
  c_callback->context = static_cast<void*>(callback_ptr);
  c_callback->callback = [](void* context, TF_Status* tf_status) {
    absl::Status status = tsl::StatusFromTF_Status(tf_status);
    tsl::StatusCallback* callback = static_cast<tsl::StatusCallback*>(context);
    (*callback)(status);
  };
  return c_callback;
}

void Destroy(TF_StatusCallback* callback) {
  if (callback == nullptr) {
    return;
  }
  if (callback->context != nullptr) {
    auto func = static_cast<tsl::StatusCallback*>(callback->context);
    delete func;
    callback->context = nullptr;
  }
}

void Destroy(TF_DeviceContext_CopyCPUTensorToDevice_Params* params) {
  if (params == nullptr) {
    return;
  }
  TF_DeleteTensor(params->cpu_tensor);
  params->cpu_tensor = nullptr;

  TF_DeleteTensor(params->device_tensor);
  params->device_tensor = nullptr;

  Destroy(params->done);
  delete params->done;
  params->done = nullptr;
}

void Destroy(TF_DeviceContext_CopyDeviceTensorToCPU_Params* params) {
  if (params == nullptr) {
    return;
  }
  TF_DeleteTensor(params->device_tensor);
  params->device_tensor = nullptr;

  delete[] params->tensor_name;
  params->tensor_name = nullptr;

  TF_DeleteTensor(params->cpu_tensor);
  params->cpu_tensor = nullptr;

  Destroy(params->done);
  delete params->done;
  params->done = nullptr;
}

void Destroy(TF_DeviceContext_CopyTensorInSameDevice_Params* params) {
  if (params == nullptr) {
    return;
  }
  TF_DeleteTensor(params->input_tensor);
  params->input_tensor = nullptr;

  TF_DeleteTensor(params->output_tensor);
  params->output_tensor = nullptr;

  Destroy(params->done);
  delete params->done;
  params->done = nullptr;
}

class TfCThunkDeviceContext final : public DeviceContext {
 public:
  explicit TfCThunkDeviceContext(const TF_DeviceContext& thunk)
      : thunk_(thunk) {}

  ~TfCThunkDeviceContext() override = default;

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, tsl::StatusCallback done,
                             bool sync_dst_compute) const override {
    auto* params = new TF_DeviceContext_CopyCPUTensorToDevice_Params();
    params->cpu_tensor = nullptr;
    params->device_tensor = nullptr;
    params->done = nullptr;
    params->sync_dst_compute = false;

    done = [params, done = std::move(done),
            device_tensor](const absl::Status& status) -> void {
      // TODO: Find a way to convert device tensor.
      done(status);
      // Make a copy of params on local variable, since this std::function will
      // be destructed in Destroy() below.
      auto param_to_delete = params;
      Destroy(params);
      delete param_to_delete;
    };

    absl::Status tensor_status;
    params->cpu_tensor = TF_TensorFromTensor(*cpu_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->done = ToC(done);
    params->sync_dst_compute = sync_dst_compute;
    thunk_.cpu_to_device_func(thunk_.device_context, params);
  }
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor,
                             tsl::StatusCallback done) override {
    auto* params = new TF_DeviceContext_CopyDeviceTensorToCPU_Params();
    params->device_tensor = nullptr;
    params->tensor_name = nullptr;
    params->tensor_name_len = 0;
    params->cpu_tensor = nullptr;
    params->done = nullptr;

    done = [params, done = std::move(done),
            cpu_tensor](const absl::Status& status) -> void {
      absl::Status tensor_status = status;
      if (tensor_status.ok()) {
        tensor_status = TF_TensorToTensor(params->cpu_tensor, cpu_tensor);
      }
      done(tensor_status);
      // Make a copy of params on local variable, since this std::function will
      // be destructed in Destroy() below.
      auto param_to_delete = params;
      Destroy(params);
      delete param_to_delete;
    };

    absl::Status tensor_status;
    params->device_tensor = TF_TensorFromTensor(*device_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->done = ToC(done);
    params->tensor_name_len = tensor_name.size();
    params->tensor_name = new char[params->tensor_name_len + 1];
    strncpy(params->tensor_name, tensor_name.data(), params->tensor_name_len);
    params->tensor_name[params->tensor_name_len] = 0;
    thunk_.device_to_cpu_func(thunk_.device_context, params);
  }
  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              tsl::StatusCallback done) const override {
    auto* params = new TF_DeviceContext_CopyTensorInSameDevice_Params();
    params->input_tensor = nullptr;
    params->output_tensor = nullptr;
    params->done = nullptr;

    done = [params, done = std::move(done),
            output_tensor](const absl::Status& status) -> void {
      absl::Status tensor_status = status;
      if (tensor_status.ok()) {
        tensor_status = TF_TensorToTensor(params->output_tensor, output_tensor);
      }
      done(tensor_status);
      // Make a copy of params on local variable, since this std::function will
      // be destructed in Destroy() below.
      auto param_to_delete = params;
      Destroy(params);
      delete param_to_delete;
    };

    absl::Status tensor_status;
    params->input_tensor = TF_TensorFromTensor(*input_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->done = ToC(done);
    thunk_.same_device_func(thunk_.device_context, params);
  }

 private:
  const TF_DeviceContext thunk_;
};

}  // namespace

DeviceContext* DeviceContext_FromC(TF_DeviceContext* c_device_context) {
  if (c_device_context == nullptr) {
    return nullptr;
  }
  return new TfCThunkDeviceContext(*c_device_context);
}

}  // namespace tensorflow
