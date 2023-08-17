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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api_conversions.h"

#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

namespace {

char* ToC(absl::string_view str) {
  char* cstr = new char[str.size() + 1];
  strncpy(cstr, str.data(), str.size());
  return cstr;
}

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

tsl::StatusCallback FromC(TF_StatusCallback* callback) {
  return [callback](absl::Status status) {
    TF_Status* c_status = TF_NewStatus();
    Set_TF_Status_from_Status(c_status, status);
    callback->callback(callback->context, c_status);
  };
}

void Destroy(TF_StatusCallback* callback) {
  if (callback == nullptr) {
    return;
  }
  if (callback->context != nullptr) {
    auto func = static_cast<tsl::StatusCallback*>(callback->context);
    delete func;
  }
}

void Destroy(TF_DeviceContext_CopyCPUTensorToDevice_Params* params) {
  if (params == nullptr) {
    return;
  }
  TF_DeleteTensor(params->cpu_tensor);
  TF_DeleteTensor(params->device_tensor);
  Destroy(params->done);
  delete params->done;
}

struct CopyCPUTensorToDeviceParamsDeleter {
  void operator()(TF_DeviceContext_CopyCPUTensorToDevice_Params* params) {
    Destroy(params);
    delete params;
  }
};

void Destroy(TF_DeviceContext_CopyDeviceTensorToCPU_Params* params) {
  if (params == nullptr) {
    return;
  }
  TF_DeleteTensor(params->device_tensor);
  delete params->tensor_name;
  TF_DeleteTensor(params->cpu_tensor);
  Destroy(params->done);
  delete params->done;
}

struct CopyDeviceTensorToCPU_ParamsDeleter {
  void operator()(TF_DeviceContext_CopyDeviceTensorToCPU_Params* params) {
    Destroy(params);
    delete params;
  }
};

void Destroy(TF_DeviceContext_CopyTensorInSameDevice_Params* params) {
  if (params == nullptr) {
    return;
  }
  TF_DeleteTensor(params->input_tensor);
  TF_DeleteTensor(params->output_tensor);
  Destroy(params->done);
  delete params->done;
}

struct CopyTensorInSameDevice_ParamsDeleter {
  void operator()(TF_DeviceContext_CopyTensorInSameDevice_Params* params) {
    Destroy(params);
    delete params;
  }
};

class TfCThunkDeviceContext final : public DeviceContext {
 public:
  explicit TfCThunkDeviceContext(const TF_DeviceContext& thunk)
      : thunk_(thunk) {}

  ~TfCThunkDeviceContext() override = default;

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, tsl::StatusCallback done,
                             bool sync_dst_compute) const override {
    auto params = std::unique_ptr<TF_DeviceContext_CopyCPUTensorToDevice_Params,
                                  CopyCPUTensorToDeviceParamsDeleter>(
        new TF_DeviceContext_CopyCPUTensorToDevice_Params());
    params->cpu_tensor = nullptr;
    params->device_tensor = nullptr;
    params->done = nullptr;
    params->sync_dst_compute = false;

    absl::Status tensor_status;
    params->cpu_tensor = TF_TensorFromTensor(*cpu_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->device_tensor = TF_TensorFromTensor(*device_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->done = ToC(std::move(done));
    params->sync_dst_compute = sync_dst_compute;
    const TF_DeviceContext_CopyCPUTensorToDevice_Impl& func =
        thunk_.cpu_to_device;
    func.cpu_to_device_func(func.context, params.get());
  }
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor,
                             tsl::StatusCallback done) override {
    auto params = std::unique_ptr<TF_DeviceContext_CopyDeviceTensorToCPU_Params,
                                  CopyDeviceTensorToCPU_ParamsDeleter>(
        new TF_DeviceContext_CopyDeviceTensorToCPU_Params());
    params->device_tensor = nullptr;
    params->tensor_name = nullptr;
    params->tensor_name_len = 0;
    params->cpu_tensor = nullptr;
    params->done = nullptr;

    absl::Status tensor_status;
    params->device_tensor = TF_TensorFromTensor(*device_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->tensor_name = ToC(tensor_name);
    params->cpu_tensor = TF_TensorFromTensor(*cpu_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->done = ToC(done);
    const TF_DeviceContext_CopyDeviceTensorToCPU_Impl& func =
        thunk_.device_to_cpu;
    func.device_to_cpu_func(func.context, params.get());
  }
  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              tsl::StatusCallback done) const override {
    auto params =
        std::unique_ptr<TF_DeviceContext_CopyTensorInSameDevice_Params,
                        CopyTensorInSameDevice_ParamsDeleter>(
            new TF_DeviceContext_CopyTensorInSameDevice_Params());
    params->input_tensor = nullptr;
    params->output_tensor = nullptr;
    params->done = nullptr;

    absl::Status tensor_status;
    params->input_tensor = TF_TensorFromTensor(*input_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->output_tensor = TF_TensorFromTensor(*output_tensor, &tensor_status);
    if (!tensor_status.ok()) {
      done(tensor_status);
      return;
    }
    params->done = ToC(done);
    const TF_DeviceContext_CopyTensorInSameDevice_Impl& func =
        thunk_.same_device;
    func.same_device_func(func.context, params.get());
  }

 private:
  const TF_DeviceContext thunk_;
};

void CpuToDeviceThunk(void* context,
                      TF_DeviceContext_CopyCPUTensorToDevice_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  Tensor cpu_tensor, device_tensor;
  tsl::StatusCallback done = FromC(params->done);
  absl::Status tensor_status;
  tensor_status = TF_TensorToTensor(params->cpu_tensor, &cpu_tensor);
  if (!tensor_status.ok()) {
    done(tensor_status);
    return;
  }
  tensor_status = TF_TensorToTensor(params->device_tensor, &device_tensor);
  if (!tensor_status.ok()) {
    done(tensor_status);
    return;
  }
  bool sync_dst_compute = params->sync_dst_compute;
  device_context->CopyCPUTensorToDevice(&cpu_tensor, /* device = */ nullptr,
                                        &device_tensor, std::move(done),
                                        sync_dst_compute);
}

void DeviceToCpuThunk(void* context,
                      TF_DeviceContext_CopyDeviceTensorToCPU_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  Tensor cpu_tensor, device_tensor;
  tsl::StatusCallback done = FromC(params->done);
  absl::Status tensor_status;
  tensor_status = TF_TensorToTensor(params->device_tensor, &device_tensor);
  if (!tensor_status.ok()) {
    done(tensor_status);
    return;
  }
  std::string_view tensor_name(params->tensor_name, params->tensor_name_len);
  tensor_status = TF_TensorToTensor(params->cpu_tensor, &cpu_tensor);
  if (!tensor_status.ok()) {
    done(tensor_status);
    return;
  }
  device_context->CopyDeviceTensorToCPU(&device_tensor, tensor_name,
                                        /* device = */ nullptr, &cpu_tensor,
                                        std::move(done));
}

void SameDeviceThunk(void* context,
                     TF_DeviceContext_CopyTensorInSameDevice_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  tsl::StatusCallback done = FromC(params->done);
  Tensor input_tensor, output_tensor;
  absl::Status tensor_status;
  tensor_status = TF_TensorToTensor(params->input_tensor, &input_tensor);
  if (!tensor_status.ok()) {
    done(tensor_status);
    return;
  }
  tensor_status = TF_TensorToTensor(params->output_tensor, &output_tensor);
  if (!tensor_status.ok()) {
    done(tensor_status);
    return;
  }
  device_context->CopyTensorInSameDevice(&input_tensor, /* device = */ nullptr,
                                         &output_tensor, std::move(done));
}

TF_DeviceContext_CopyCPUTensorToDevice_Impl BindCpuToDevice(
    DeviceContext* device_context) {
  TF_DeviceContext_CopyCPUTensorToDevice_Impl cpu_to_device_func;
  cpu_to_device_func.context = static_cast<void*>(device_context);
  cpu_to_device_func.cpu_to_device_func = CpuToDeviceThunk;
  return cpu_to_device_func;
}

TF_DeviceContext_CopyDeviceTensorToCPU_Impl BindDeviceToCpu(
    DeviceContext* device_context) {
  TF_DeviceContext_CopyDeviceTensorToCPU_Impl device_to_cpu_func;
  device_to_cpu_func.context = static_cast<void*>(device_context);
  device_to_cpu_func.device_to_cpu_func = DeviceToCpuThunk;
  return device_to_cpu_func;
}

TF_DeviceContext_CopyTensorInSameDevice_Impl BindSameDevice(
    DeviceContext* device_context) {
  TF_DeviceContext_CopyTensorInSameDevice_Impl same_device_func;
  same_device_func.context = static_cast<void*>(device_context);
  same_device_func.same_device_func = SameDeviceThunk;
  return same_device_func;
}

}  // namespace

void Destroy(TF_DeviceContext* c_device_context) {}

void TF_DeviceContext_Deleter::operator()(TF_DeviceContext* c_device_context) {
  Destroy(c_device_context);
  delete c_device_context;
}

TF_DeviceContext* ToC(DeviceContext* device_context) {
  TF_DeviceContext* c_device_context = new TF_DeviceContext();
  c_device_context->cpu_to_device = BindCpuToDevice(device_context);
  c_device_context->device_to_cpu = BindDeviceToCpu(device_context);
  c_device_context->same_device = BindSameDevice(device_context);
  return c_device_context;
}

DeviceContext* FromC(TF_DeviceContext* c_device_context) {
  if (c_device_context == nullptr) {
    return nullptr;
  }
  return new TfCThunkDeviceContext(*c_device_context);
}

}  // namespace tensorflow
