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

#include "xla/service/cpu/runtime/call_thunk.h"

#include <utility>

#include "absl/status/status.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

CallThunk::CallThunk(Info info, ThunkSequence called_sequence)
    : Thunk(Kind::kCall, std::move(info)),
      called_sequence_(std::move(called_sequence)) {}

absl::Status CallThunk::Execute(const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });
  return called_sequence_.Execute(params);
}

}  // namespace xla::cpu
