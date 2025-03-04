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

#include "xla/stream_executor/cuda/compilation_provider_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/strings/str_cat.h"
#include "xla/xla.pb.h"

namespace stream_executor::cuda {
namespace {

TEST(CompilationProviderOptionsTest, Default) {
  CompilationProviderOptions options;
  EXPECT_EQ(options.nvjitlink_mode(),
            CompilationProviderOptions::NvJitLinkMode::kDisabled);
  EXPECT_FALSE(options.enable_libnvptxcompiler());
  EXPECT_FALSE(options.enable_llvm_module_compilation_parallelism());
  EXPECT_FALSE(options.enable_driver_compilation());
  EXPECT_EQ(options.cuda_data_dir(), "");
}

TEST(CompilationProviderOptionsTest, FromDebugOptions) {
  xla::DebugOptions debug_options;
  debug_options.set_xla_gpu_libnvjitlink_mode(
      xla::DebugOptions::LIB_NV_JIT_LINK_MODE_ENABLED);
  debug_options.set_xla_gpu_enable_libnvptxcompiler(true);
  debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(true);
  debug_options.set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(true);
  debug_options.set_xla_gpu_cuda_data_dir("/usr/local/cuda");
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kEnabled, true, true,
                true, "/usr/local/cuda"));

  debug_options.set_xla_gpu_libnvjitlink_mode(
      xla::DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED);
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kDisabled, true,
                true, true, "/usr/local/cuda"));

  debug_options.set_xla_gpu_libnvjitlink_mode(
      xla::DebugOptions::LIB_NV_JIT_LINK_MODE_AUTO);
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kAuto, true, true,
                true, "/usr/local/cuda"));

  debug_options.set_xla_gpu_enable_libnvptxcompiler(false);
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kAuto, false, true,
                true, "/usr/local/cuda"));

  debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(false);
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kAuto, false, false,
                true, "/usr/local/cuda"));

  debug_options.set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kAuto, false, false,
                false, "/usr/local/cuda"));

  debug_options.set_xla_gpu_cuda_data_dir("");
  EXPECT_EQ(CompilationProviderOptions::FromDebugOptions(debug_options),
            CompilationProviderOptions(
                CompilationProviderOptions::NvJitLinkMode::kAuto, false, false,
                false, ""));
}

TEST(CompilationProviderOptionsTest, ToString) {
  CompilationProviderOptions options;
  EXPECT_EQ(options.ToString(),
            "CompilationProviderOptions{nvjitlink_mode: 0, "
            "enable_libnvptxcompiler: false, "
            "enable_llvm_module_compilation_parallelism: false, "
            "enable_driver_compilation: false, cuda_data_dir: }");
}

TEST(CompilationProviderOptionsTest, AbslStringify) {
  // This test doesn't want to guarantee any particular output format.
  // It's here just to ensure that the AbslStringify template can in fact be
  // instantiated.
  CompilationProviderOptions options{};
  EXPECT_THAT(absl::StrCat(options), testing::Not(testing::IsEmpty()));
}

TEST(CompilationProviderOptionsTest, AbslHashValue) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {CompilationProviderOptions{},
       CompilationProviderOptions{
           CompilationProviderOptions::NvJitLinkMode::kDisabled, false, false,
           true, ""},
       CompilationProviderOptions{
           CompilationProviderOptions::NvJitLinkMode::kEnabled, false, false,
           false, "/usr/local/cuda"},
       CompilationProviderOptions{
           CompilationProviderOptions::NvJitLinkMode::kAuto, false, true, false,
           ""},
       CompilationProviderOptions{
           CompilationProviderOptions::NvJitLinkMode::kDisabled, true, false,
           false, ""},
       CompilationProviderOptions{
           CompilationProviderOptions::NvJitLinkMode::kDisabled, false, false,
           false, "/usr/local/cuda"}}));
}

}  // namespace
}  // namespace stream_executor::cuda
