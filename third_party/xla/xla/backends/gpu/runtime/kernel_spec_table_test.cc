/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/backends/gpu/runtime/kernel_spec_table.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/kernel_spec_table.pb.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

namespace se = ::stream_executor;

se::KernelLoaderSpec GenerateCubinSpec(absl::string_view kernel_name,
                                       absl::string_view cubin) {
  std::vector<uint8_t> bytes(cubin.begin(), cubin.end());
  return se::KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
      std::move(bytes), std::string(kernel_name), /*arity=*/2);
}

se::KernelLoaderSpec GenerateSymbolSpec(absl::string_view kernel_name) {
  return se::KernelLoaderSpec::CreateSerializableInProcessSymbolSpec(
      std::string(kernel_name), /*symbol=*/nullptr, std::string(kernel_name),
      /*arity=*/1);
}

TEST(KernelSpecTableTest, InternDeduplicatesEqualSpecs) {
  KernelSpecTable table;
  EXPECT_TRUE(table.empty());

  ASSERT_OK_AND_ASSIGN(const int32_t first,
                       table.Intern(GenerateCubinSpec("kernel", "cubin")));
  ASSERT_OK_AND_ASSIGN(const int32_t second,
                       table.Intern(GenerateCubinSpec("kernel", "cubin")));
  ASSERT_OK_AND_ASSIGN(const int32_t third,
                       table.Intern(GenerateCubinSpec("other", "cubin")));

  EXPECT_EQ(first, 0);
  EXPECT_EQ(second, 0);
  EXPECT_EQ(third, 1);
  EXPECT_EQ(table.size(), 2);
}

TEST(KernelSpecTableTest, InternKeepsDistinctSpecsApart) {
  KernelSpecTable table;
  constexpr int kNumSpecs = 128;
  for (int i = 0; i < kNumSpecs; ++i) {
    EXPECT_THAT(table.Intern(GenerateCubinSpec(absl::StrCat("kernel_", i),
                                               absl::StrCat("cubin_", i))),
                IsOkAndHolds(i));
  }
  EXPECT_EQ(table.size(), kNumSpecs);
  for (int i = 0; i < kNumSpecs; ++i) {
    ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec, table.Get(i));
    EXPECT_EQ(spec.kernel_name(), absl::StrCat("kernel_", i));
  }
}

TEST(KernelSpecTableTest, InternInternsInProcessSymbolSpecs) {
  KernelSpecTable table;
  EXPECT_THAT(table.Intern(GenerateSymbolSpec("symbol")), IsOkAndHolds(0));
  EXPECT_THAT(table.Intern(GenerateSymbolSpec("symbol")), IsOkAndHolds(0));
  EXPECT_EQ(table.size(), 1);
}

TEST(KernelSpecTableTest, InternRejectsUnserializableSpecs) {
  KernelSpecTable table;

  se::KernelLoaderSpec symbol_without_persistent_name =
      se::KernelLoaderSpec::CreateInProcessSymbolSpec(
          /*symbol=*/nullptr, "kernel", /*arity=*/1);
  EXPECT_THAT(table.Intern(std::move(symbol_without_persistent_name)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  se::KernelLoaderSpec spec_with_packing_func =
      se::KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
          /*cubin_bytes=*/{1, 2, 3}, "kernel", /*arity=*/1,
          [](const se::Kernel&, const se::KernelArgs&) {
            return std::unique_ptr<se::KernelArgsPackedArrayBase>();
          });
  EXPECT_THAT(table.Intern(std::move(spec_with_packing_func)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(table.empty());
}

TEST(KernelSpecTableTest, GetRejectsOutOfRangeIndices) {
  KernelSpecTable table;
  ASSERT_OK(table.Intern(GenerateCubinSpec("kernel", "cubin")));
  EXPECT_THAT(table.Get(1), StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(table.Get(-1), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(KernelSpecTableTest, ToProtoAndFromProtoRoundTrip) {
  KernelSpecTable table;
  EXPECT_THAT(table.Intern(GenerateCubinSpec("kernel_a", "cubin_a")),
              IsOkAndHolds(0));
  EXPECT_THAT(table.Intern(GenerateCubinSpec("kernel_b", "cubin_b")),
              IsOkAndHolds(1));

  ASSERT_OK_AND_ASSIGN(KernelSpecTableProto proto, table.ToProto());
  EXPECT_EQ(proto.kernel_specs_size(), 2);

  ASSERT_OK_AND_ASSIGN(KernelSpecTable reconstructed,
                       KernelSpecTable::FromProto(proto));
  EXPECT_EQ(reconstructed.size(), 2);
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec0, reconstructed.Get(0));
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec spec1, reconstructed.Get(1));
  EXPECT_EQ(spec0.kernel_name(), "kernel_a");
  EXPECT_EQ(spec1.kernel_name(), "kernel_b");
}

TEST(KernelSpecTableTest, FromProtoPreservesDuplicates) {
  KernelSpecTableProto proto;
  ASSERT_OK_AND_ASSIGN(*proto.add_kernel_specs(),
                       GenerateCubinSpec("kernel", "cubin").ToProto());
  ASSERT_OK_AND_ASSIGN(*proto.add_kernel_specs(),
                       GenerateCubinSpec("kernel", "cubin").ToProto());

  ASSERT_OK_AND_ASSIGN(KernelSpecTable table,
                       KernelSpecTable::FromProto(proto));
  EXPECT_EQ(table.size(), 2);
  // A subsequent intern still finds the first of the two.
  EXPECT_THAT(table.Intern(GenerateCubinSpec("kernel", "cubin")),
              IsOkAndHolds(0));
}

TEST(KernelSpecTableTest, InternSharesPayloadBufferAcrossCopies) {
  KernelSpecTable table;
  ASSERT_OK(table.Intern(GenerateCubinSpec("kernel", "shared_cubin_payload")));
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec copy1, table.Get(0));
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpec copy2, table.Get(0));

  ASSERT_TRUE(copy1.has_cuda_cubin_in_memory());
  ASSERT_TRUE(copy2.has_cuda_cubin_in_memory());
  EXPECT_EQ(copy1.cuda_cubin_in_memory()->cubin_bytes.data(),
            copy2.cuda_cubin_in_memory()->cubin_bytes.data());
}

}  // namespace
}  // namespace xla::gpu
