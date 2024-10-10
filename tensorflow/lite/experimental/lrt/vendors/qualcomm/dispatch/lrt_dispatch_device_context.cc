// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/dispatch/lrt_dispatch_device_context.h"

#include <cstddef>
#include <cstdint>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/qairt/include/QNN/HTP/QnnHtpMem.h"
#include "third_party/qairt/include/QNN/QnnBackend.h"
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnMem.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn.h"

namespace {

absl::StatusOr<Qnn_DataType_t> ToQnnDataType(LrtElementType element_type) {
  switch (element_type) {
    case kLrtElementTypeBool:
      return QNN_DATATYPE_BOOL_8;
    case kLrtElementTypeInt4:
      return QNN_DATATYPE_SFIXED_POINT_4;
    case kLrtElementTypeInt8:
      return QNN_DATATYPE_INT_8;
    case kLrtElementTypeInt16:
      return QNN_DATATYPE_INT_16;
    case kLrtElementTypeInt32:
      return QNN_DATATYPE_INT_32;
    case kLrtElementTypeInt64:
      return QNN_DATATYPE_INT_64;
    case kLrtElementTypeUInt8:
      return QNN_DATATYPE_UINT_8;
    case kLrtElementTypeUInt16:
      return QNN_DATATYPE_UINT_16;
    case kLrtElementTypeUInt32:
      return QNN_DATATYPE_UINT_32;
    case kLrtElementTypeUInt64:
      return QNN_DATATYPE_UINT_64;
    case kLrtElementTypeFloat16:
      return QNN_DATATYPE_FLOAT_16;
    case kLrtElementTypeFloat32:
      return QNN_DATATYPE_FLOAT_32;
    case kLrtElementTypeFloat64:
      return QNN_DATATYPE_FLOAT_64;
    default:
      return absl::InvalidArgumentError("Element type is not supported by QNN");
  }
}

}  // namespace

LrtDispatchDeviceContextT::~LrtDispatchDeviceContextT() {
  if (auto status = qnn_.qnn_interface().backendFree(backend_handle_);
      status != QNN_SUCCESS) {
    ABSL_LOG(ERROR) << "Failed to free backend: " << status;
  }
}

absl::StatusOr<LrtDispatchDeviceContextT::Ptr>
LrtDispatchDeviceContextT::Create(const lrt::qnn::Qnn& qnn) {
  const QnnBackend_Config_t* backend_configs[] = {nullptr};
  Qnn_BackendHandle_t backend_handle = nullptr;
  if (auto status = qnn.qnn_interface().backendCreate(
          qnn.log_handle(), backend_configs, &backend_handle);
      status != QNN_SUCCESS) {
    return absl::InternalError("Failed to create backend");
  }

  return Ptr(new LrtDispatchDeviceContextT(qnn, backend_handle));
}

absl::StatusOr<LrtTensorBuffer> LrtDispatchDeviceContextT::GetTensorBuffer(
    LrtTensorBufferHandle tensor_buffer_handle) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry.ok()) {
    return registry_entry.status();
  }

  return (*registry_entry)->tensor_buffer;
}

absl::StatusOr<Qnn_MemHandle_t> LrtDispatchDeviceContextT::GetMemHandle(
    LrtTensorBufferHandle tensor_buffer_handle, const Qnn_Tensor_t& tensor,
    Qnn_ContextHandle_t context_handle) {
  auto registry_entry = tensor_buffer_registry_.Get(tensor_buffer_handle);
  if (!registry_entry.ok()) {
    return registry_entry.status();
  }

  if (!(*registry_entry)->qnn_mem_handle) {
    auto qnn_mem_handle = RegisterTensorBuffer((*registry_entry)->tensor_buffer,
                                               tensor, context_handle);
    if (!qnn_mem_handle.ok()) {
      return qnn_mem_handle.status();
    }
    (*registry_entry)->qnn_mem_handle = *qnn_mem_handle;
  }

  return (*registry_entry)->qnn_mem_handle;
}

absl::StatusOr<Qnn_MemHandle_t> LrtDispatchDeviceContextT::RegisterTensorBuffer(
    LrtTensorBuffer tensor_buffer, const Qnn_Tensor_t& tensor,
    Qnn_ContextHandle_t context_handle) {
  LrtTensorBufferType tensor_buffer_type;
  if (auto status = LrtGetTensorBufferType(tensor_buffer, &tensor_buffer_type);
      status != kLrtStatusOk) {
    return absl::InternalError("Failed to get tensor buffer type");
  }

  if (tensor_buffer_type != kLrtTensorBufferTypeFastRpc) {
    return absl::InternalError("Unsupported tensor buffer type");
  }

  size_t tensor_buffer_size;
  if (auto status = LrtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size);
      status != kLrtStatusOk) {
    return absl::InternalError("Failed to get tensor buffer size");
  }

  size_t tensor_buffer_offset;
  if (auto status =
          LrtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLrtStatusOk) {
    return absl::InternalError("Failed to get tensor buffer offset");
  }

  LrtRankedTensorType tensor_type;
  if (auto status = LrtGetTensorBufferTensorType(tensor_buffer, &tensor_type);
      status != kLrtStatusOk) {
    return absl::InternalError("Failed to get tensor buffer's type");
  }

  auto tensor_data_type = ToQnnDataType(tensor_type.element_type);
  if (!tensor_data_type.ok()) {
    return tensor_data_type.status();
  }

  uint32_t tensor_rank = tensor_type.layout.rank;
  uint32_t* tensor_dimensions = reinterpret_cast<uint32_t*>(
      const_cast<int32_t*>(tensor_type.layout.dimensions));
  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    return absl::InternalError("Tensor strides are not supported by QNN");
  }

  void* fastrpc_buffer_addr;
  int fastrpc_buffer_fd;
#if LRT_HAS_FASTRPC_SUPPORT
  if (auto status = LrtGetTensorBufferFastRpcBuffer(
          tensor_buffer, &fastrpc_buffer_addr, &fastrpc_buffer_fd);
      status != kLrtStatusOk) {
    return absl::InternalError("Failed to get FastRPC buffer");
  }
#else
  (void)fastrpc_buffer_addr;
  (void)fastrpc_buffer_fd;
  return absl::InternalError("FastRPC support is missing on this platform");
#endif  // LRT_HAS_FASTRPC_SUPPORT

  QnnMemHtp_Descriptor_t mem_htp_descriptor = {};
  mem_htp_descriptor.type = QNN_HTP_MEM_SHARED_BUFFER;
  mem_htp_descriptor.size = tensor_buffer_size;
  mem_htp_descriptor.sharedBufferConfig = {fastrpc_buffer_fd,
                                           tensor_buffer_offset};

  Qnn_MemDescriptor_t mem_descriptor = {};
  mem_descriptor.memShape = {tensor_rank, tensor_dimensions, nullptr};
  mem_descriptor.dataType = *tensor_data_type;
  mem_descriptor.memType = QNN_MEM_TYPE_CUSTOM;
  mem_descriptor.customInfo = &mem_htp_descriptor;

  Qnn_MemHandle_t mem_handle = nullptr;
  if (auto status = qnn_.qnn_interface().memRegister(
          context_handle, &mem_descriptor, 1UL, &mem_handle);
      status != QNN_SUCCESS) {
    return absl::InternalError("Failed to register tensor buffer");
  }

  if (!mem_handle) {
    return absl::InternalError("Failed to register buffer: null mem_handle");
  }

  return mem_handle;
}
