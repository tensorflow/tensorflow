/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_UNDONATABLE_BUFFER_CONVERTER_H_
#define TENSORFLOW_CORE_TFRT_IFRT_UNDONATABLE_BUFFER_CONVERTER_H_

#include "absl/status/status.h"
#include "xla/python/ifrt/array.h"

namespace tensorflow {
namespace ifrt_serving {

// Converts, in place, each PjRt buffer backing `array` into an
// xla::UndonatableCommonPjRtBuffer aliasing the same device memory, by
// donating the underlying tracked buffer. `array` must be non-null and the
// exclusive owner of its buffers (i.e. freshly created and not yet shared).
// No-op for arrays whose buffers are not backed by CommonPjRtBuffer.
//
// Matches LoadedVariableArrayFn so it can be injected into
// AsyncLoadRestoredTensorAsIfrtLoadedVariable without adding PjRt
// implementation deps to ifrt_loaded_variable_utils.
absl::Status MakeArrayBuffersUndonatable(xla::ifrt::Array* array);

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_UNDONATABLE_BUFFER_CONVERTER_H_
