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
#include "xla/backends/gpu/runtime/internable_kernel_loader_spec.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/internable_kernel_loader_spec.pb.h"
#include "xla/backends/gpu/runtime/kernel_spec_table.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::tsl::proto_testing::EqualsProto;

namespace se = ::stream_executor;

se::KernelLoaderSpec GenerateCubinSpec(absl::string_view kernel_name,
                                       absl::string_view cubin) {
  std::vector<uint8_t> bytes(cubin.begin(), cubin.end());
  return se::KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
      std::move(bytes), std::string(kernel_name), /*arity=*/2);
}

TEST(InternableKernelLoaderSpecTest, ToProtoWithoutTableEmitsInlineSpec) {
  InternableKernelLoaderSpec internable(GenerateCubinSpec("kernel", "cubin"));
  EXPECT_EQ(internable.kernel_spec_table(), nullptr);

  ASSERT_OK_AND_ASSIGN(InternableKernelLoaderSpecProto proto,
                       internable.ToProto());
  EXPECT_TRUE(proto.has_kernel_spec());
  EXPECT_FALSE(proto.has_kernel_spec_index());
  ASSERT_OK_AND_ASSIGN(se::KernelLoaderSpecProto expected,
                       GenerateCubinSpec("kernel", "cubin").ToProto());
  EXPECT_THAT(proto.kernel_spec(), EqualsProto(expected));
}

TEST(InternableKernelLoaderSpecTest, ToProtoWithTableInternsAndEmitsIndex) {
  KernelSpecTable table;
  InternableKernelLoaderSpec first(GenerateCubinSpec("a", "cubin_a"), &table);
  InternableKernelLoaderSpec second(GenerateCubinSpec("b", "cubin_b"), &table);
  InternableKernelLoaderSpec third(GenerateCubinSpec("a", "cubin_a"), &table);

  ASSERT_OK_AND_ASSIGN(InternableKernelLoaderSpecProto proto1, first.ToProto());
  ASSERT_OK_AND_ASSIGN(InternableKernelLoaderSpecProto proto2,
                       second.ToProto());
  ASSERT_OK_AND_ASSIGN(InternableKernelLoaderSpecProto proto3, third.ToProto());

  EXPECT_EQ(table.size(), 2);
  EXPECT_TRUE(proto1.has_kernel_spec_index());
  EXPECT_EQ(proto1.kernel_spec_index(), 0);
  EXPECT_TRUE(proto2.has_kernel_spec_index());
  EXPECT_EQ(proto2.kernel_spec_index(), 1);
  EXPECT_TRUE(proto3.has_kernel_spec_index());
  EXPECT_EQ(proto3.kernel_spec_index(), 0);
}

TEST(InternableKernelLoaderSpecTest, FromProtoRoundTripsWithTable) {
  KernelSpecTable table;
  InternableKernelLoaderSpec original(
      GenerateCubinSpec("kernel", "cubin_bytes"), &table);
  ASSERT_OK_AND_ASSIGN(InternableKernelLoaderSpecProto proto,
                       original.ToProto());

  ASSERT_OK_AND_ASSIGN(
      InternableKernelLoaderSpec reconstructed,
      InternableKernelLoaderSpec::FromProto(proto, std::nullopt, &table));
  EXPECT_EQ(reconstructed.kernel_spec().kernel_name(), "kernel");
  EXPECT_EQ(reconstructed.kernel_spec().arity(), 2);
  ASSERT_TRUE(reconstructed.kernel_spec().has_cuda_cubin_in_memory());
}

TEST(InternableKernelLoaderSpecTest, FromProtoRejectsIndexWithoutTable) {
  InternableKernelLoaderSpecProto proto;
  proto.set_kernel_spec_index(0);
  EXPECT_THAT(InternableKernelLoaderSpec::FromProto(proto, std::nullopt,
                                                    /*table=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(InternableKernelLoaderSpecTest, FromProtoRejectsDanglingIndex) {
  KernelSpecTable empty_table;
  InternableKernelLoaderSpecProto proto;
  proto.set_kernel_spec_index(0);
  EXPECT_THAT(
      InternableKernelLoaderSpec::FromProto(proto, std::nullopt, &empty_table),
      StatusIs(absl::StatusCode::kOutOfRange));
}

}  // namespace
}  // namespace xla::gpu
