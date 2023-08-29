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

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {

void CopyTF_TensorToTensor(const TF_Tensor* src, Tensor* dst) {
  // TODO: Convert through a lookup table for better API compatibility.
  DataType dtype = static_cast<DataType>(TF_TensorType(src));
  TensorShape tensor_shape;
  int dim = TF_NumDims(src);
  for (int i = 0; i < dim; ++i) {
    tensor_shape.AddDim(TF_Dim(src, i));
  }
  *dst = Tensor(dtype, tensor_shape);

  std::memcpy(dst->data(), TF_TensorData(src), TF_TensorByteSize(src));
}

TF_Tensor* CopyTensorToTF_Tensor(const Tensor& src) {
  // TODO: Convert through a lookup table for better API compatibility.
  TF_DataType dtype = static_cast<TF_DataType>(src.dtype());
  const TensorShape& shape = src.shape();
  auto dims = std::make_unique<int64_t[]>(shape.dims());
  size_t len = TF_DataTypeSize(dtype);
  for (int i = 0; i < shape.dims(); ++i) {
    dims[i] = shape.dim_size(i);
    len *= dims[i];
  }
  TF_Tensor* tf_tensor =
      TF_AllocateTensor(dtype, dims.get(), shape.dims(), len);
  void* data = TF_TensorData(tf_tensor);
  std::memcpy(data, src.data(), len);
  return tf_tensor;
}

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

tsl::StatusCallback FromC(TF_StatusCallback* callback) {
  return [callback](absl::Status status) {
    TF_Status* c_status = TF_NewStatus();
    Set_TF_Status_from_Status(c_status, status);
    callback->callback(callback->context, c_status);
    TF_DeleteStatus(c_status);
  };
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

void CpuToDeviceThunk(void* context,
                      TF_DeviceContext_CopyCPUTensorToDevice_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  Tensor* cpu_tensor = new Tensor();
  Tensor* device_tensor = new Tensor();
  tsl::StatusCallback done = [params, device_tensor,
                              cpu_tensor](absl::Status status) {
    delete cpu_tensor;
    // TODO: find a way to convert device tensor.
    // params->device_tensor = TF_TensorFromTensor(*device_tensor,
    //                                             &tensor_status);
    delete device_tensor;
    FromC(params->done)(status);
  };
  CopyTF_TensorToTensor(params->cpu_tensor, cpu_tensor);
  bool sync_dst_compute = params->sync_dst_compute;
  device_context->CopyCPUTensorToDevice(cpu_tensor, /* device = */ nullptr,
                                        device_tensor, std::move(done),
                                        sync_dst_compute);
}

void DeviceToCpuThunk(void* context,
                      TF_DeviceContext_CopyDeviceTensorToCPU_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  Tensor* cpu_tensor = new Tensor();
  Tensor* device_tensor = new Tensor();
  tsl::StatusCallback done = [params, device_tensor,
                              cpu_tensor](absl::Status status) {
    delete device_tensor;
    params->cpu_tensor = CopyTensorToTF_Tensor(*cpu_tensor);
    delete cpu_tensor;
    FromC(params->done)(status);
  };
  std::string_view tensor_name(params->tensor_name, params->tensor_name_len);
  // TODO: Find a way to convert device tensor.
  device_context->CopyDeviceTensorToCPU(
      /* device_tensor = */ nullptr, tensor_name,
      /* device = */ nullptr, cpu_tensor, std::move(done));
}

void SameDeviceThunk(void* context,
                     TF_DeviceContext_CopyTensorInSameDevice_Params* params) {
  LOG(FATAL) << "Unimplemented";  // Crash OK
}

}  // namespace

void Destroy(TF_DeviceContext* c_device_context) {}

void TF_DeviceContext_Deleter::operator()(TF_DeviceContext* c_device_context) {
  Destroy(c_device_context);
  delete c_device_context;
}

TF_DeviceContext* ToC(DeviceContext* device_context) {
  TF_DeviceContext* c_device_context = new TF_DeviceContext();
  c_device_context->device_context = static_cast<void*>(device_context);
  c_device_context->cpu_to_device_func = CpuToDeviceThunk;
  c_device_context->device_to_cpu_func = DeviceToCpuThunk;
  c_device_context->same_device_func = SameDeviceThunk;
  return c_device_context;
}

DeviceContext* FromC(TF_DeviceContext* c_device_context) {
  if (c_device_context == nullptr) {
    return nullptr;
  }
  return new TfCThunkDeviceContext(*c_device_context);
}

}  // namespace tensorflow
