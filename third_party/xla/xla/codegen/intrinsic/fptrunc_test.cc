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

#include "xla/codegen/intrinsic/fptrunc.h"

#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::intrinsics {

TEST(FpTruncTest, SclarIninsic) {
  EXPECT_EQ(FpTrunc::Name(Type::S(F32), Type::S(BF16)),
            "xla.fptrunc.f32.to.bf16");
}

TEST(FpTruncTest, VectorIninsic) {
  EXPECT_EQ(FpTrunc::Name(Type::V(F32, 4), Type::V(BF16, 4)),
            "xla.fptrunc.v4f32.to.v4bf16");
}
}  // namespace xla::codegen::intrinsics
