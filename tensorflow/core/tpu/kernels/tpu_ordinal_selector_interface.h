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

#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_ORDINAL_SELECTOR_INTERFACE_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_ORDINAL_SELECTOR_INTERFACE_H_

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace tpu {

class TPUOrdinalSelectorInterface {
 public:
  virtual ~TPUOrdinalSelectorInterface() = default;
  virtual int64 GetOrdinal(absl::optional<uint64> key, int64_t* req_id) = 0;
  virtual void DequeueFromCoreSelector(int32_t device_ordinal,
                                       int64_t req_id) = 0;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_ORDINAL_SELECTOR_INTERFACE_H_
