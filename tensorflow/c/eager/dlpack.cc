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

#include "tensorflow/c/eager/dlpack.h"

#include <string>

#include "include/dlpack/dlpack.h"  // from @dlpack
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

// Managing context for the DLManagedTensor, will manage the lifetime of
// DLManagedTensor. When calling DLManagedTensor::deleter, it will notify the
// original framework of destruction, and this context will be deleted also.
struct TfDlManagedTensorCtx {
  TensorReference reference;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;

  explicit TfDlManagedTensorCtx(const TensorReference& ref) : reference(ref) {}
};

// Gets tensor from eager tensor handle.
const Tensor* GetTensorFromHandle(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::TensorHandleFromInterface(tensorflow::unwrap(h));
  if (handle->Type() != TensorHandle::LOCAL) {
    status->status = tensorflow::errors::InvalidArgument(
        "DLPack doesn't support ", handle->TypeString(), " tensor");
    return nullptr;
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return tensor;
}

// Deleter for DLManagedTensor
void DLManagedTensorDeleter(DLManagedTensor* arg) {
  TfDlManagedTensorCtx* owner =
      static_cast<TfDlManagedTensorCtx*>(arg->manager_ctx);
  owner->reference.Unref();
  delete owner;
}

// Converts TF_DATAType to DLPack data type.
DLDataType GetDlDataType(TF_DataType data_type, TF_Status* status) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = TF_DataTypeSize(data_type) * 8;
  switch (data_type) {
    case TF_DataType::TF_HALF:
    case TF_DataType::TF_FLOAT:
    case TF_DataType::TF_DOUBLE:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case TF_DataType::TF_INT8:
    case TF_DataType::TF_INT16:
    case TF_DataType::TF_INT32:
    case TF_DataType::TF_INT64:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case TF_DataType::TF_BOOL:
    case TF_DataType::TF_UINT8:
    case TF_DataType::TF_UINT16:
    case TF_DataType::TF_UINT32:
    case TF_DataType::TF_UINT64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case TF_DataType::TF_BFLOAT16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case TF_DataType::TF_COMPLEX64:
    case TF_DataType::TF_COMPLEX128:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    default:
      status->status = tensorflow::errors::InvalidArgument(
          DataType_Name(static_cast<DataType>(data_type)),
          " is not supported by dlpack");
      break;
  }
  return dtype;
}

// Gets DLPack's DLDevice from eager tensor handle.
DLDevice GetDlContext(TFE_TensorHandle* h, TF_Status* status) {
  DLDevice ctx;
  const char* device_name =
      tensorflow::unwrap(h)->BackingDeviceName(&status->status);
  DeviceNameUtils::ParsedName parsed_name;
  tensorflow::DeviceNameUtils::ParseFullName(device_name, &parsed_name);
  std::string device_type = parsed_name.type;
  int device_id = 0;
  if (parsed_name.has_id) {
    device_id = parsed_name.id;
  }

  ctx.device_id = device_id;
  if (device_type == "CPU") {
    ctx.device_type = DLDeviceType::kDLCPU;
  } else if (device_type == "GPU") {
    ctx.device_type = DLDeviceType::kDLCUDA;
  } else {
    status->status = tensorflow::errors::InvalidArgument(
        "Unsupported Device Type for dlpack");
  }

  return ctx;
}

// Converts DLDevice to TF device name.
absl::optional<std::string> DeviceNameFromDlContext(const DLDevice& ctx,
                                                    TF_Status* status) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      return "CPU:0";
    case DLDeviceType::kDLCUDA:
      return absl::StrCat("GPU:", ctx.device_id);
    default:
      return absl::nullopt;
  }
}

// Converts DLPack data type to TF_DATATYPE.
Status TfDataTypeFormDlDataType(const DLDataType& dtype,
                                TF_DataType* tf_dtype) {
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          *tf_dtype = TF_DataType::TF_UINT8;
          return OkStatus();
        case 16:
          *tf_dtype = TF_DataType::TF_UINT16;
          return OkStatus();
        case 32:
          *tf_dtype = TF_DataType::TF_UINT32;
          return OkStatus();
        case 64:
          *tf_dtype = TF_DataType::TF_UINT64;
          return OkStatus();
        default:
          return tensorflow::errors::InvalidArgument("Unsupported UInt bits: ",
                                                     dtype.bits);
      }
      return OkStatus();
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          *tf_dtype = TF_DataType::TF_INT8;
          return OkStatus();
        case 16:
          *tf_dtype = TF_DataType::TF_INT16;
          return OkStatus();
        case 32:
          *tf_dtype = TF_DataType::TF_INT32;
          return OkStatus();
        case 64:
          *tf_dtype = TF_DataType::TF_INT64;
          return OkStatus();
        default:
          return tensorflow::errors::InvalidArgument("Unsupported Int bits: ",
                                                     dtype.bits);
      }
      return OkStatus();
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          *tf_dtype = TF_DataType::TF_HALF;
          return OkStatus();
        case 32:
          *tf_dtype = TF_DataType::TF_FLOAT;
          return OkStatus();
        case 64:
          *tf_dtype = TF_DataType::TF_DOUBLE;
          return OkStatus();
        default:
          return tensorflow::errors::InvalidArgument("Unsupported Float bits: ",
                                                     dtype.bits);
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          *tf_dtype = TF_DataType::TF_BFLOAT16;
          return OkStatus();
        default:
          return tensorflow::errors::InvalidArgument(
              "Unsupported BFloat bits: ", dtype.bits);
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 64:
          *tf_dtype = TF_DataType::TF_COMPLEX64;
          return OkStatus();
        case 128:
          *tf_dtype = TF_DataType::TF_COMPLEX128;
          return OkStatus();
        default:
          return tensorflow::errors::InvalidArgument(
              "Unsupported Complex bits: ", dtype.bits);
      }
      break;
    default:
      return tensorflow::errors::InvalidArgument("Unsupported Type Codes: ",
                                                 dtype.code);
  }
}

// Wraps the deleter function of DLManagedTensor to match the function signature
// TFE_NewTensorHandleFromDeviceMemory.
void DeallocatorWrapperFunc(void* data, size_t len, void* dlmt_vptr) {
  TFE_CallDLManagedTensorDeleter(dlmt_vptr);
}

// Checks whether the stride array matches the layout of compact, row-majored
// data.
bool IsValidStrideCompactRowMajorData(int64_t* shape_arr, int64_t* stride_arr,
                                      int ndim) {
  if (ndim >= 1 && stride_arr[ndim - 1] != 1) {
    return false;
  }
  for (int i = ndim - 2; i >= 0; --i) {
    if (stride_arr[i] != shape_arr[i + 1] * stride_arr[i + 1]) {
      return false;
    }
  }
  return true;
}
}  // namespace

void TFE_CallDLManagedTensorDeleter(void* dlm_ptr) {
  DLManagedTensor* dlMTensor = static_cast<DLManagedTensor*>(dlm_ptr);
  if (dlMTensor->deleter != nullptr) {
    dlMTensor->deleter(dlMTensor);
  }
}

void* TFE_HandleToDLPack(TFE_TensorHandle* h, TF_Status* status) {
  auto tf_dlm_context = GetDlContext(h, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  auto* tf_dlm_data = TFE_TensorHandleDevicePointer(h, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  const Tensor* tensor = GetTensorFromHandle(h, status);
  TF_DataType data_type = static_cast<TF_DataType>(tensor->dtype());

  auto tf_dlm_type = GetDlDataType(data_type, status);
  if (!status->status.ok()) {
    return nullptr;
  }

  TensorReference tensor_ref(*tensor);  // This will call buf_->Ref()
  auto* tf_dlm_tensor_ctx = new TfDlManagedTensorCtx(tensor_ref);
  tf_dlm_tensor_ctx->reference = tensor_ref;

  DLManagedTensor* dlm_tensor = &tf_dlm_tensor_ctx->tensor;
  dlm_tensor->manager_ctx = tf_dlm_tensor_ctx;
  dlm_tensor->deleter = &DLManagedTensorDeleter;
  dlm_tensor->dl_tensor.device = tf_dlm_context;
  int ndim = tensor->dims();
  dlm_tensor->dl_tensor.ndim = ndim;
  dlm_tensor->dl_tensor.data = tf_dlm_data;
  dlm_tensor->dl_tensor.dtype = tf_dlm_type;

  std::vector<int64_t>* shape_arr = &tf_dlm_tensor_ctx->shape;
  std::vector<int64_t>* stride_arr = &tf_dlm_tensor_ctx->strides;
  shape_arr->resize(ndim);
  stride_arr->resize(ndim, 1);
  for (int i = 0; i < ndim; i++) {
    (*shape_arr)[i] = tensor->dim_size(i);
  }
  for (int i = ndim - 2; i >= 0; --i) {
    (*stride_arr)[i] = (*shape_arr)[i + 1] * (*stride_arr)[i + 1];
  }

  dlm_tensor->dl_tensor.shape = shape_arr->data();
  // There are two ways to represent compact row-major data
  // 1) nullptr indicates tensor is compact and row-majored.
  // 2) fill in the strides array as the real case for compact row-major data.
  // Here we choose option 2, since some frameworks didn't handle the strides
  // argument properly.
  dlm_tensor->dl_tensor.strides = stride_arr->data();

  dlm_tensor->dl_tensor.byte_offset =
      0;  // TF doesn't handle the strides and byte_offsets here
  return static_cast<void*>(dlm_tensor);
}

TFE_TensorHandle* TFE_HandleFromDLPack(void* dlm, TF_Status* status,
                                       TFE_Context* ctx) {
  DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(dlm);
  DLTensor* dl_tensor = &dlmt->dl_tensor;
  absl::optional<std::string> device_name =
      DeviceNameFromDlContext(dl_tensor->device, status);
  if (!device_name.has_value()) {
    status->status =
        tensorflow::errors::InvalidArgument("Unsupported Device Type");
    return nullptr;
  }
  TF_DataType dtype;
  Status s = TfDataTypeFormDlDataType(dl_tensor->dtype, &dtype);
  if (!s.ok()) {
    status->status = std::move(s);
    return nullptr;
  }
  int num_dims = dl_tensor->ndim;
  const int64_t* dims = dl_tensor->shape;
  void* data = dl_tensor->data;

  size_t total_bytes = dl_tensor->dtype.bits / 8;
  for (int i = 0; i < num_dims; i++) {
    total_bytes *= dims[i];
  }

  if (dl_tensor->strides != nullptr &&
      !IsValidStrideCompactRowMajorData(dl_tensor->shape, dl_tensor->strides,
                                        num_dims)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid strides array from DLPack");
    return nullptr;
  }

  TFE_TensorHandle* handle = TFE_NewTensorHandleFromDeviceMemory(
      ctx, device_name.value().c_str(), dtype, dims, num_dims, data,
      total_bytes, &DeallocatorWrapperFunc, dlmt, status);

  return handle;
}

}  // namespace tensorflow
