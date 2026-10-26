
/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {

tsl::thread::ThreadPool& IfrtModelContext::GetThreadPool() const {
  return thread_pool_;
}

std::optional<int64_t> IfrtModelContext::LookupProgramId(
    uint64_t fingerprint, absl::Span<const int> variable_arg_indices) const {
  absl::MutexLock lock(mutex_);
  auto it = compiled_programs_by_module_fingerprint_.find(fingerprint);
  if (it == compiled_programs_by_module_fingerprint_.end()) {
    return std::nullopt;
  }
  for (const CompiledProgram& program : it->second) {
    if (absl::MakeConstSpan(program.variable_arg_indices) ==
        variable_arg_indices) {
      return program.program_id;
    }
  }
  return std::nullopt;
}

bool IfrtModelContext::HasProgramWithFingerprint(uint64_t fingerprint) const {
  absl::MutexLock lock(mutex_);
  return compiled_programs_by_module_fingerprint_.contains(fingerprint);
}

void IfrtModelContext::RegisterProgramId(
    uint64_t fingerprint, absl::Span<const int> variable_arg_indices,
    int64_t program_id) {
  absl::MutexLock lock(mutex_);
  compiled_programs_by_module_fingerprint_[fingerprint].push_back(
      {.variable_arg_indices = std::vector<int>(variable_arg_indices.begin(),
                                                variable_arg_indices.end()),
       .program_id = program_id});
}

absl::Status IfrtModelContext::Freeze() {
  restore_tensor_registry_.Freeze();
  for (auto& program_handle : handles_) {
    TF_RETURN_IF_ERROR(program_handle.Freeze());
  }
  frozen_ = true;
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
