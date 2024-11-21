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

#include <cstddef>
#include <memory>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

// NOLINTNEXTLINE
using litert::mediatek::NEURON_NO_ERROR;
using litert::mediatek::NeuronMemory;

LiteRtDispatchDeviceContextT::~LiteRtDispatchDeviceContextT() = default;

litert::Expected<LiteRtDispatchDeviceContextT::Ptr>
LiteRtDispatchDeviceContextT::Create(
    const litert::mediatek::NeuronAdapter& neuron_adapter) {
  return std::unique_ptr<LiteRtDispatchDeviceContextT>(
      new LiteRtDispatchDeviceContextT(neuron_adapter));
}

litert::Expected<LiteRtTensorBufferHandle>
LiteRtDispatchDeviceContextT::RegisterTensorBuffer(
    const litert::TensorBuffer& tensor_buffer) {
  auto tensor_buffer_type = tensor_buffer.BufferType();
  if (!tensor_buffer_type) {
    return tensor_buffer_type.Error();
  }

  if (*tensor_buffer_type != kLiteRtTensorBufferTypeAhwb) {
    LITERT_LOG(LITERT_ERROR, "Unsupported buffer type: %d",
               *tensor_buffer_type);
    return litert::Unexpected(kLiteRtStatusErrorUnsupported);
  }

  auto tensor_buffer_size = tensor_buffer.Size();
  if (!tensor_buffer_size) {
    return tensor_buffer_size.Error();
  }

  auto tensor_buffer_offset = tensor_buffer.Offset();
  if (!tensor_buffer_offset) {
    return tensor_buffer_offset.Error();
  }

  auto ahwb = tensor_buffer.GetAhwb();
  if (!ahwb) {
    return ahwb.Error();
  }

#ifdef __ANDROID__
  NeuronMemory* neuron_memory;
  if (neuron_adapter_.api().memory_create_from_ahwb(*ahwb, &neuron_memory) !=
      NEURON_NO_ERROR) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to create NeuronMemory from AHWB");
  }
  return neuron_memory_registry_.Register(neuron_memory, *tensor_buffer_size,
                                          *tensor_buffer_offset);
#else
  (void)neuron_adapter_;
  return litert::Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "AHardwareBuffer is not supported on this platform");
#endif
}

LiteRtDispatchDeviceContextT::NeuronMemoryRegistry::~NeuronMemoryRegistry() {
  for (auto i = 0; i < records_.size(); ++i) {
    auto& record = records_[i];
    if (record.neuron_memory != nullptr) {
      neuron_adapter_.api().memory_free(record.neuron_memory);
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
    neuron_adapter_.api().memory_free(mem);
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
