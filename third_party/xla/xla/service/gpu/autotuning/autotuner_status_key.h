/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_STATUS_KEY_H_
#define XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_STATUS_KEY_H_

#include "absl/strings/string_view.h"

namespace xla::gpu {

// Status payload key to put errors at when autotune cache hits are required.
// See absl::Status docs for full details, but methods like
// {Get,Set,Clear}Payload allow manipulating it. The value of the payload is not
// specified and individual sources of this error may provide different values.
extern const absl::string_view kAutotuneCacheRequiredErrorPayloadKey;

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_STATUS_KEY_H_
