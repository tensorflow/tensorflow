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

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

class GemvTest : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {};

TEST_F(GemvTest, BatchedGemv) {
  absl::string_view hlo_string = R"(
    HloModule batched_gemv

    ENTRY e {
      p0 = f32[8,257] parameter(0)
      p1 = f32[8,257,517] parameter(1)
      ROOT dot = f32[8,517] dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={1},
                                         lhs_batch_dims={0}, rhs_batch_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{1e-5, 1e-3}));
}

TEST_F(GemvTest, ColumnMajorGemv) {
  absl::string_view hlo_string = R"(
    HloModule gemv_column_major

    ENTRY e {
      p0 = f32[1025] parameter(0)
      p1 = f32[1025,513] parameter(1)
      ROOT dot = f32[513] dot(p0, p1), lhs_contracting_dims={0},
                                       rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{1e-5, 1e-3}));
}

TEST_F(GemvTest, RowMajorGemv) {
  absl::string_view hlo_string = R"(
    HloModule gemv_row_major

    ENTRY e {
      p0 = f32[513,1025] parameter(1)
      p1 = f32[1025] parameter(0)
      ROOT dot = f32[513] dot(p0, p1), lhs_contracting_dims={1},
                                       rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{1e-5, 1e-3}));
}

}  // namespace
}  // namespace xla::cpu
