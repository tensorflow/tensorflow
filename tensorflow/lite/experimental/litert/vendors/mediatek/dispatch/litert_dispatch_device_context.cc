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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"

#include <sys/mman.h>

#include <cstddef>
#include <memory>

#include "neuron/api/NeuronAdapter.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

using litert::Error;

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() = default;

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api) {
  return std::unique_ptr<LiteRtDispatchDeviceContextT>(
      new LiteRtDispatchDeviceContextT(neuron_adapter_api));
}

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    LiteRtTensorBuffer tensor_buffer) {
  LiteRtTensorBufferType tensor_buffer_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferType(tensor_buffer, &tensor_buffer_type));

  if (tensor_buffer_type != kLiteRtTensorBufferTypeAhwb &&
      tensor_buffer_type != kLiteRtTensorBufferTypeDmaBuf) {
    return Error(kLiteRtStatusErrorUnsupported, "Unsupported buffer type");
  }

  size_t tensor_buffer_size;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferSize(tensor_buffer, &tensor_buffer_size));

  size_t tensor_buffer_offset;
  if (auto status =
          LiteRtGetTensorBufferOffset(tensor_buffer, &tensor_buffer_offset);
      status != kLiteRtStatusOk) {
    if (status == kLiteRtStatusErrorNotFound) {
      tensor_buffer_offset = 0;
    } else {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to get buffer offset");
    }
  }

  LiteRtRankedTensorType tensor_type;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetTensorBufferTensorType(tensor_buffer, &tensor_type));

  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Tensor strides are not supported");
  }

  switch (tensor_buffer_type) {
    case kLiteRtTensorBufferTypeAhwb:
#if LITERT_HAS_AHWB_SUPPORT
      AHardwareBuffer* ahwb;
      if (auto status = LiteRtGetTensorBufferAhwb(tensor_buffer, &ahwb);
          status != kLiteRtStatusOk) {
        return Error(status, "Failed to get AHWB");
      }
#else
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "AHardwareBuffer is not supported on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      NeuronMemory* neuron_memory;
#if LITERT_HAS_AHWB_SUPPORT
      if (neuron_adapter_api_.api().memory_create_from_ahwb(
              ahwb, &neuron_memory) != NEURON_NO_ERROR) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to create NeuronMemory from AHWB");
      }
      return neuron_memory_registry_.Register(neuron_memory, tensor_buffer_size,
                                              tensor_buffer_offset);
#else
      (void)neuron_adapter_api_;
      return litert::Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          "AHardwareBuffer is not supported on this platform");
#endif  // LITERT_HAS_AHWB_SUPPORT
      break;

    case kLiteRtTensorBufferTypeDmaBuf:

      int fd;
#if LITERT_HAS_DMABUF_SUPPORT
      void* addr;
      if (auto status =
              LiteRtGetTensorBufferDmaBufBuffer(tensor_buffer, &addr, &fd);
          status != kLiteRtStatusOk) {
        return Error(status, "Failed to get DMA-BUF");
      }
#else
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "DMA-BUF is not supported on this platform");
#endif  // LITERT_HAS_DMABUF_SUPPORT
      if (neuron_adapter_api_.api().memory_create_from_fd(
              tensor_buffer_size, /*protect*/ PROT_READ | PROT_WRITE, fd,
              tensor_buffer_offset, &neuron_memory) != NEURON_NO_ERROR) {
        return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                  "Failed to create NeuronMemory from DMA-BUF");
      }
      return neuron_memory_registry_.Register(neuron_memory, tensor_buffer_size,
                                              tensor_buffer_offset);
      break;

    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported buffer type: %d",
                 tensor_buffer_type);
      return litert::Unexpected(kLiteRtStatusErrorUnsupported);
  }
}

LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::~NeuronMemoryRegistry() {
  for (auto i = 0; i < records_.size(); ++i) {
    auto& record = records_[i];
    if (record.neuron_memory != nullptr) {
      neuron_adapter_api_.api().memory_free(record.neuron_memory);
    }
  }
}

LiteRtTensorBufferHandle
LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::Register(
    NeuronMemory* neuron_memory, size_t size, size_t offset) {
  int dest_index = -1;
  for (auto i = 0; i < records_.size(); ++i) {
    if (!records_[i].neuron_memory) {
      dest_index = i;
      break;
    }
  }
  if (dest_index < 0) {
    dest_index = records_.size();
    records_.push_back({});
  }
  auto& dest = records_[dest_index];
  dest = {neuron_memory, size, offset};
  return dest_index;
}

litert::Expected<void>
LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::Unregister(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto record = Find(tensor_buffer_handle);
  if (!record) {
    return record.Error();
  } else {
    auto& mem = (*record)->neuron_memory;
    neuron_adapter_api_.api().memory_free(mem);
    mem = nullptr;
    return {};
  }
}

litert::Expected<LiteRtDispatchDeviceContextT::NeuronMemoryInfo*>
LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::Find(
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (tensor_buffer_handle < 0 || tensor_buffer_handle >= records_.size()) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Invalid tensor buffer handle");
  }
  return &records_[tensor_buffer_handle];
}
