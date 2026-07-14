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

#include "xla/backends/gpu/runtime/thunk_id.h"

#include <gtest/gtest.h>

namespace xla::gpu {
namespace {

TEST(ThunkIdTest, GeneratesUniqueIds) {
  ThunkIdGenerator generator;

  ThunkId id1 = generator.GetNextThunkId();
  ThunkId id2 = generator.GetNextThunkId();
  ThunkId id3 = generator.GetNextThunkId();

  // The only property that we guarantee is uniqueness, no ordering, no
  // consecutiveness, etc.
  EXPECT_NE(id1, id2);
  EXPECT_NE(id1, id3);
  EXPECT_NE(id2, id3);
}

}  // namespace
}  // namespace xla::gpu
