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

#include "xla/pjrt/profiling/profiling_context_no_op.h"

#include <memory>

#include "xla/pjrt/profiling/profiling_context.h"

namespace xla {

std::unique_ptr<ProfilingContext> CreateProfilingContext() {
  return std::make_unique<ProfilingContextNoOp>();
}

std::unique_ptr<WithProfilingContext> CreateWithProfilingContext(
    ProfilingContext* switch_to) {
  return std::make_unique<WithProfilingContextNoOp>();
}

char ProfilingContext::ID = 0;
char WithProfilingContext::ID = 0;
char ProfilingContextNoOp::ID = 0;
char WithProfilingContextNoOp::ID = 0;
}  // namespace xla
