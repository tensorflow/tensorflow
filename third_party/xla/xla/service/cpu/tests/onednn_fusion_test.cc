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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

using OneDnnFusionTest = HloTestBase;

inline constexpr bool IsOneDnnGraphEnabled() {
#if defined(XLA_ONEDNN_USE_GRAPH_API)
  return true;
#endif  // XLA_ONEDNN_USE_GRAPH_API
  return false;
}

TEST_F(OneDnnFusionTest, Exponential) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule exp

    onednn_fusion {
      %p0 = f32[4] parameter(0)
      ROOT %exp = f32[4] exponential(%p0)
    }

    ENTRY entry {
      %p0 = f32[4] parameter(0)
      ROOT %fusion = f32[4] fusion(%p0), kind=kCustom, calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

// TODO(penporn): Make a parameterized BinaryEltwiseOp test instead.
TEST_F(OneDnnFusionTest, Add) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add

    onednn_fusion {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %add = f32[10] add(%p0, %p1)
    }

    ENTRY entry {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %fusion = f32[10] fusion(%p0, %p1), kind=kCustom, calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

TEST_F(OneDnnFusionTest, Mul) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule mul

    onednn_fusion {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %mul = f32[10] multiply(%p0, %p1)
    }

    ENTRY entry {
      %p0 = f32[10] parameter(0)
      %p1 = f32[10] parameter(1)
      ROOT %fusion = f32[10] fusion(%p0, %p1), kind=kCustom, calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

TEST_F(OneDnnFusionTest, MatMul) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule mul

    onednn_fusion {
      %p0 = f32[10,20] parameter(0)
      %p1 = f32[20,30] parameter(1)
      ROOT %mul = f32[10,30] dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY entry {
      %p0 = f32[10,20] parameter(0)
      %p1 = f32[20,30] parameter(1)
      ROOT %fusion = f32[10,30] fusion(%p0, %p1), kind=kCustom,
        calls=onednn_fusion,
        backend_config={"fusion_config": {kind: "__onednn_fusion"}}
    })";

  if (!IsOneDnnGraphEnabled()) {
    GTEST_SKIP() << "oneDNN fusion is not supported";
  }

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-5}));
}

}  // namespace
}  // namespace xla::cpu
