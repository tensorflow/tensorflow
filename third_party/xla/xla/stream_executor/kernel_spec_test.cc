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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace stream_executor {
namespace {

using ::testing::Field;
using ::testing::Optional;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

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

TEST(KernelLoaderSpec, OwningCudaCubin) {
  static constexpr std::array<uint8_t, 4> kCubinData = {0xDE, 0xAD, 0xBE, 0xEF};
  auto spec =
      stream_executor::KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
          std::vector<uint8_t>{kCubinData.begin(), kCubinData.end()},
          "kernel24", 2);
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

TEST(KernelLoaderSpec, OwningCudaPtx) {
  static constexpr absl::string_view kPtxData = "PTX DEADBEEF";
  auto spec =
      stream_executor::KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
          std::string{kPtxData}, "kernel24", 2);
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  EXPECT_TRUE(spec.has_cuda_ptx_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  EXPECT_THAT(spec.cuda_ptx_in_memory(),
              Optional(Field(&CudaPtxInMemory::ptx, kPtxData)));
  EXPECT_THAT(spec.kernel_name(), "kernel24");
}

TEST(KernelLoaderSpec, PtxKernelFromProto) {
  KernelLoaderSpecProto proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        ptx { data: "PTX!" }
        kernel_name: "kernel_name"
        arity: 42
      )pb",
      &proto));

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                          KernelLoaderSpec::FromProto(proto));
  EXPECT_THAT(spec.kernel_name(), "kernel_name");
  EXPECT_THAT(spec.arity(), 42);
  EXPECT_THAT(spec.cuda_ptx_in_memory(),
              Optional(Field(&CudaPtxInMemory::ptx, "PTX!")));
}

TEST(KernelLoaderSpec, PtxKernelToProto) {
  auto spec = stream_executor::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
      "PTX!", "kernel_name", 42);

  EXPECT_THAT(spec.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                ptx { data: "PTX!" }
                kernel_name: "kernel_name"
                arity: 42
              )pb")));
}

TEST(KernelLoaderSpec, CubinKernelFromProto) {
  KernelLoaderSpecProto proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        cubin { data: "CUBIN" }
        kernel_name: "kernel_name"
        arity: 42
      )pb",
      &proto));

  TF_ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                          KernelLoaderSpec::FromProto(proto));
  EXPECT_THAT(spec.kernel_name(), "kernel_name");
  std::array<uint8_t, 5> kCubin = {'C', 'U', 'B', 'I', 'N'};
  EXPECT_THAT(spec.arity(), 42);
  EXPECT_THAT(spec.cuda_cubin_in_memory(),
              Optional(Field(&CudaCubinInMemory::cubin_bytes, kCubin)));
}

TEST(KernelLoaderSpec, CubinKernelToProto) {
  std::array<uint8_t, 5> kCubin = {'C', 'U', 'B', 'I', 'N'};
  auto spec = stream_executor::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kCubin, "kernel_name", 42);

  EXPECT_THAT(spec.ToProto(), IsOkAndHolds(EqualsProto(R"pb(
                cubin { data: "CUBIN" }
                kernel_name: "kernel_name"
                arity: 42
              )pb")));
}

TEST(KernelLoaderSpec, InProcessSymbolToProto) {
  auto spec = stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(
      nullptr, "kernel_name", 42);

  EXPECT_THAT(spec.ToProto(), StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor
