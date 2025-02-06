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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <memory>

#include "neuron/api/NeuronAdapter.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

class LiteRtDispatchDeviceContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchDeviceContextT>;
  struct NeuronMemoryInfo {
    NeuronMemory* neuron_memory;
    size_t size;
    size_t offset;
  };

  ~LiteRtDispatchDeviceContextT();

  static litert::Expected<Ptr> Create(
      const litert::mediatek::NeuronAdapterApi& neuron_adapter_api);

  litert::Expected<LiteRtTensorBufferHandle> RegisterTensorBuffer(
      const litert::TensorBuffer& tensor_buffer);

  litert::Expected<void> UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    return neuron_memory_registry_.Unregister(tensor_buffer_handle);
  }

  litert::Expected<NeuronMemoryInfo> GetNeuronMemoryInfo(
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    auto record = neuron_memory_registry_.Find(tensor_buffer_handle);
    if (!record) {
      return record.Error();
    } else {
      return NeuronMemoryInfo(**record);
    }
  }

 private:
  class NeuronMemoryRegistry {
   public:
    explicit NeuronMemoryRegistry(
        const litert::mediatek::NeuronAdapterApi& neuron_adapter_api)
        : neuron_adapter_api_(neuron_adapter_api) {}
    ~NeuronMemoryRegistry();
    LiteRtTensorBufferHandle Register(NeuronMemory* neuron_memory, size_t size,
                                      size_t offset);
    litert::Expected<void> Unregister(
        LiteRtTensorBufferHandle tensor_buffer_handle);
    litert::Expected<NeuronMemoryInfo*> Find(
        LiteRtTensorBufferHandle tensor_buffer_handle);

   private:
    const litert::mediatek::NeuronAdapterApi& neuron_adapter_api_;
    std::vector<NeuronMemoryInfo> records_;
  };

  explicit LiteRtDispatchDeviceContextT(
      const litert::mediatek::NeuronAdapterApi& neuron_adapter_api)
      : neuron_adapter_api_(neuron_adapter_api),
        neuron_memory_registry_(neuron_adapter_api) {}

  const litert::mediatek::NeuronAdapterApi& neuron_adapter_api_;
  NeuronMemoryRegistry neuron_memory_registry_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
