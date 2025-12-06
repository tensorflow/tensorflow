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

#include "xla/service/gpu/ptx_compile_options_from_debug_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/cuda/compilation_options.h"
#include "xla/xla.pb.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {
using ::stream_executor::cuda::CompilationOptions;
using ::testing::Field;

TEST(PtxCompileOptionsFromDebugOptionsTest,
     DefaultDebugOptionsResultsInDefaultCompilationOptions) {
  DebugOptions debug_options;
  EXPECT_EQ(PtxCompileOptionsFromDebugOptions(debug_options),
            CompilationOptions{});
}

TEST(PtxCompileOptionsFromDebugOptionsTest, OptimizationsCanBeDisabled) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_disable_gpuasm_optimizations(true);
  EXPECT_THAT(PtxCompileOptionsFromDebugOptions(debug_options),
              Field(&CompilationOptions::disable_optimizations, true));
}

TEST(PtxCompileOptionsFromDebugOptionsTest, LineInfoCanBeEnabled) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_generate_line_info(true);
  EXPECT_THAT(PtxCompileOptionsFromDebugOptions(debug_options),
              Field(&CompilationOptions::generate_line_info, true));
}

TEST(PtxCompileOptionsFromDebugOptionsTest, DebugInfoCanBeEnabled) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_generate_debug_info(true);
  EXPECT_THAT(PtxCompileOptionsFromDebugOptions(debug_options),
              Field(&CompilationOptions::generate_debug_info, true));
}

TEST(PtxCompileOptionsFromDebugOptionsTest,
     RegSpillAsErrorCanBeEnabledForAllKernels) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_fail_ptx_compilation_on_register_spilling(true);
  EXPECT_THAT(PtxCompileOptionsFromDebugOptions(debug_options),
              Field(&CompilationOptions::cancel_if_reg_spill, true));
}

}  // namespace
}  // namespace xla::gpu
