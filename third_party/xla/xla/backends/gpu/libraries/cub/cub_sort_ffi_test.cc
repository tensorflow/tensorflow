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

#include "xla/backends/gpu/libraries/cub/cub_sort_ffi.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/ffi/ffi.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;

TEST(CubSortFfiTest, GetCubSortOffsetsSize) {
  EXPECT_EQ(GetCubSortOffsetsSize(1), 8);
  EXPECT_EQ(GetCubSortOffsetsSize(4), 20);
}

TEST(CubSortFfiTest, MakeCubSortSegmentOffsets) {
  EXPECT_THAT(MakeCubSortSegmentOffsets(/*batch_size=*/1, /*segment_size=*/7),
              ElementsAre(0, 7));
  EXPECT_THAT(MakeCubSortSegmentOffsets(/*batch_size=*/3, /*segment_size=*/5),
              ElementsAre(0, 5, 10, 15));
}

// A fake platform stream type; the bindings only require a pointer type.
struct FakeStream;
using FakeStreamT = FakeStream*;

absl::StatusOr<std::unique_ptr<int64_t>> KeysInstantiate(
    ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::BufferR1<xla::U8>>, bool, int64_t) {
  return std::make_unique<int64_t>(0);
}

absl::Status KeysExecute(ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer>,
                         ffi::Result<ffi::BufferR1<xla::U8>>, bool, int64_t,
                         FakeStreamT) {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<int64_t>> PairsInstantiate(
    ffi::AnyBuffer, ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer>,
    ffi::Result<ffi::AnyBuffer>, ffi::Result<ffi::BufferR1<xla::U8>>, bool,
    int64_t) {
  return std::make_unique<int64_t>(0);
}

absl::Status PairsExecute(ffi::AnyBuffer, ffi::AnyBuffer,
                          ffi::Result<ffi::AnyBuffer>,
                          ffi::Result<ffi::AnyBuffer>,
                          ffi::Result<ffi::BufferR1<xla::U8>>, bool, int64_t,
                          FakeStreamT) {
  return absl::OkStatus();
}

// Verifies that the shared bindings match the CUB sort handler signatures.
TEST(CubSortFfiTest, BindingsMatchHandlerSignatures) {
  EXPECT_NE(BindCubSortKeysInstantiate().To(KeysInstantiate), nullptr);
  EXPECT_NE(BindCubSortKeysExecute<FakeStreamT>().To(KeysExecute), nullptr);
  EXPECT_NE(BindCubSortPairsInstantiate().To(PairsInstantiate), nullptr);
  EXPECT_NE(BindCubSortPairsExecute<FakeStreamT>().To(PairsExecute), nullptr);
}

}  // namespace
}  // namespace xla::gpu
