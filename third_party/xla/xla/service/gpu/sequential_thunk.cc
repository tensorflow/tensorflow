/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/sequential_thunk.h"

#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

using ::tsl::profiler::ScopedAnnotation;

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {}

std::string SequentialThunk::ToStringExtra(int indent) const {
  std::string result = "\n";
  absl::StrAppend(&result, thunks().ToString(indent + 1, nullptr));
  return result;
}

Status SequentialThunk::Initialize(const GpuExecutable& executable,
                                   se::StreamExecutor* executor) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(executable, executor));
  }
  return OkStatus();
}

Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  for (const auto& thunk : thunks_) {
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
