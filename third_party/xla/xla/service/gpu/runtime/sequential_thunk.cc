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

#include "xla/service/gpu/runtime/sequential_thunk.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/runtime/annotation.h"
#include "xla/service/gpu/runtime/thunk.h"
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

absl::Status SequentialThunk::Prepare(const PrepareParams& params,
                                      ResourceRequests& resource_requests) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  const ModuleAnnotations* annotations = GetCurrentModuleAnnotations();
  for (const auto& thunk : thunks_) {
    auto annotation =
        GetKernelAnnotation(annotations, thunk->profile_annotation());
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
