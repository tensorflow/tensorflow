
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
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_program.h"

namespace tensorflow {
namespace ifrt_serving {

tsl::thread::ThreadPool& IfrtModelContext::GetThreadPool() const {
  return thread_pool_;
}

absl::StatusOr<std::shared_ptr<IfrtProgram>>
IfrtModelContext::GetOrCreateProgram(
    uint64_t fingerprint, absl::Span<const int> variable_arg_indices,
    absl::FunctionRef<absl::StatusOr<std::unique_ptr<IfrtProgram>>()> create) {
  absl::MutexLock lock(mutex_);
  auto it = programs_by_module_fingerprint_.find(fingerprint);
  if (it != programs_by_module_fingerprint_.end()) {
    for (const ProgramEntry& entry : it->second) {
      if (absl::MakeConstSpan(entry.variable_arg_indices) ==
          variable_arg_indices) {
        return entry.program;
      }
    }
  }
  if (frozen_) {
    return nullptr;
  }
  // Create under the lock so that concurrent compilations of the same program
  // do not both create one. Creation does not compile, so this is cheap.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<IfrtProgram> program, create());
  std::shared_ptr<IfrtProgram> shared_program = std::move(program);
  programs_by_module_fingerprint_[fingerprint].push_back(
      {.variable_arg_indices = std::vector<int>(variable_arg_indices.begin(),
                                                variable_arg_indices.end()),
       .program = shared_program});
  return shared_program;
}

bool IfrtModelContext::HasProgramWithFingerprint(uint64_t fingerprint) const {
  absl::MutexLock lock(mutex_);
  return programs_by_module_fingerprint_.contains(fingerprint);
}

absl::Status IfrtModelContext::Freeze() {
  restore_tensor_registry_.Freeze();
  absl::MutexLock lock(handles_mutex_);
  for (auto& program_handle : handles_) {
    TF_RETURN_IF_ERROR(program_handle.Freeze());
  }
  {
    absl::MutexLock program_lock(mutex_);
    frozen_ = true;
  }
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
