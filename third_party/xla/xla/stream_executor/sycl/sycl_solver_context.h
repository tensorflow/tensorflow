/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_SOLVER_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_SOLVER_CONTEXT_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {

class SyclSolverContext : public GpuSolverContext {
 public:
  ~SyclSolverContext() override;
  static absl::StatusOr<std::unique_ptr<GpuSolverContext>> Create();

  absl::Status SetStream(Stream* stream) override;

 private:
  explicit SyclSolverContext();
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_SOLVER_CONTEXT_H_
