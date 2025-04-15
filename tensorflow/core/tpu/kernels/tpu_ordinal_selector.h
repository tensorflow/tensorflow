/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_ORDINAL_SELECTOR_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_ORDINAL_SELECTOR_H_

#include <optional>

#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_ordinal_selector_interface.h"

namespace tensorflow {
namespace tpu {

// A reserved ID for deferred core selection. Intentionally set at a number
// that is more than the number of cores available in a future system.
constexpr int32_t kDeferredCoreSelectionReserved = -8193;

class TPUOrdinalSelector : TPUOrdinalSelectorInterface {
 public:
  explicit TPUOrdinalSelector(int num_cores_per_replica = 1) {
    stream_executor::tpu::OpsApiFn()->TfTpuOrdinalSelector_CreateFn(
        &ordinal_selector_, num_cores_per_replica);
  }
  ~TPUOrdinalSelector() override {
    stream_executor::tpu::OpsApiFn()->TfTpuOrdinalSelector_DestroyFn(
        ordinal_selector_);
  }
  int64_t GetOrdinal(std::optional<uint64> key, int64_t* req_id) override {
    int64_t ordinal;
    stream_executor::tpu::OpsApiFn()->TfTpuOrdinalSelector_GetOrdinalFn(
        ordinal_selector_, key, req_id, &ordinal);
    return ordinal;
  }
  void DequeueFromCoreSelector(int32_t device_ordinal,
                               int64_t req_id) override {
    stream_executor::tpu::OpsApiFn()
        ->TfTpuOrdinalSelector_DequeueFromCoreSelectorFn(
            ordinal_selector_, device_ordinal, req_id);
  }

 private:
  TfTpuOrdinalSelector* ordinal_selector_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_ORDINAL_SELECTOR_H_
