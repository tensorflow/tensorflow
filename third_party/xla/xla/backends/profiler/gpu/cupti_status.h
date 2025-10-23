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

<<<<<<< HEAD
<<<<<<< HEAD:third_party/xla/xla/backends/profiler/gpu/cupti_status.h
=======
>>>>>>> upstream/master
#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_

#include "absl/status/status.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"

namespace xla {
namespace profiler {

absl::Status ToStatus(CUptiResult result);

<<<<<<< HEAD
=======
#include <gtest/gtest.h>
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class IntelGpuCompilerTest : public HloTestBase {};

TEST_F(IntelGpuCompilerTest, CheckCompiler) {
  auto compiler = backend().compiler();
  EXPECT_EQ(compiler->PlatformId(), stream_executor::sycl::kSyclPlatformId);
>>>>>>> upstream/master:third_party/xla/xla/service/gpu/intel_gpu_compiler_test.cc
}
}  // namespace xla

<<<<<<< HEAD:third_party/xla/xla/backends/profiler/gpu/cupti_status.h
#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_
=======
}  // namespace
}  // namespace gpu
}  // namespace xla
>>>>>>> upstream/master:third_party/xla/xla/service/gpu/intel_gpu_compiler_test.cc
=======
}
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_
>>>>>>> upstream/master
