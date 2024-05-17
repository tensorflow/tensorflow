/* Copyright 2017 The OpenXLA Authors.

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

// Utilities common between the client and server for working with
// StreamExecutor devices.

#ifndef XLA_DEVICE_UTIL_H_
#define XLA_DEVICE_UTIL_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "xla/stream_executor/stream_executor_interface.h"
#include "xla/types.h"

namespace xla {

// Returns a string that represents the device in terms of platform and ordinal;
// e.g. the first CUDA device will be "cuda:0"
std::string DeviceIdentifier(se::StreamExecutorInterface* stream_exec) {
  return absl::StrCat(stream_exec->GetPlatform()->Name(), ":",
                      stream_exec->device_ordinal());
}

}  // namespace xla

#endif  // XLA_DEVICE_UTIL_H_
