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

#ifndef XLA_SERVICE_GPU_RUNTIME_SEQUENTIAL_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_SEQUENTIAL_THUNK_H_

#include <string>

#include "absl/status/status.h"
#include "xla/service/gpu/runtime/thunk.h"

namespace xla {
namespace gpu {

// A thunk that wraps a list of sub-thunks. Executing this thunk executes all
// the sub-thunks sequentially. This is useful to implement instructions that
// require multiple kernel launches or library calls.
class SequentialThunk : public Thunk {
 public:
  SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks);
  SequentialThunk(const SequentialThunk&) = delete;
  SequentialThunk& operator=(const SequentialThunk&) = delete;

  ThunkSequence& thunks() { return thunks_; }
  const ThunkSequence& thunks() const { return thunks_; }
  std::string ToStringExtra(int indent) const override;

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequests& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  // The list of sub-thunks.
  ThunkSequence thunks_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_SEQUENTIAL_THUNK_H_
