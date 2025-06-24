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

#include "xla/stream_executor/kernel_spec.h"

#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/string_view.h"

namespace stream_executor {
namespace {

using ::testing::Field;
using ::testing::Optional;

TEST(KernelLoaderSpec, InProcessSymbol) {
  void* symbol = absl::bit_cast<void*>(0xDEADBEEFul);
  auto spec = stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
      symbol, "kernel24", 2);
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  EXPECT_FALSE(spec.has_cuda_ptx_in_memory());
  EXPECT_TRUE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.in_process_symbol(),
              Optional(Field(&InProcessSymbol::symbol, symbol)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, CudaCubin) {
  static constexpr std::array<uint8_t, 4> kCubinData = {0xDE, 0xAD, 0xBE, 0xEF};
  auto spec = stream_executor::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kCubinData, "kernel24", 2);
  EXPECT_TRUE(spec.has_cuda_cubin_in_memory());
  EXPECT_FALSE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_cubin_in_memory(),
              Optional(Field(&CudaCubinInMemory::cubin_bytes, kCubinData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, CudaPtx) {
  static constexpr absl::string_view kPtxData = "PTX DEADBEEF";
  auto spec = stream_executor::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
      kPtxData, "kernel24", 2);
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  EXPECT_TRUE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_ptx_in_memory(),
              Optional(Field(&CudaPtxInMemory::ptx, kPtxData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

}  // namespace
}  // namespace stream_executor
