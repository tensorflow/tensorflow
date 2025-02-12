/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/gather_simplifier.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

class GatherSimplifierTest : public HloHardwareIndependentTestBase {};

TEST_F(GatherSimplifierTest, TransformsStartIndices) {
  // Verifies that GatherSimplifier
  // - Makes the index_vector_dim dimensions explicit
  // - Flattens start_indices into a 2d tensor.
  // - Undoes the flattening in the output.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34] parameter(0)
      indices = s32[42,43] parameter(1)
      ROOT gather = f32[42,43,7,8] gather(operand, indices),
          offset_dims={2,3},
          collapsed_slice_dims={},
          start_index_map={0},
          index_vector_dim=2,
          slice_sizes={7,8}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
         CHECK: %[[VECTOR_DIM:.*]] = s32[42,43,1]{2,1,0} reshape(%indices)
         CHECK: %[[INDICES_2D:.*]] = s32[1806,1]{1,0} reshape(%[[VECTOR_DIM]])
         CHECK: %[[GATHER:.*]] = f32[1806,7,8]{{.*}} gather(
    CHECK-SAME:     %operand, %[[INDICES_2D]]),
    CHECK-SAME:     offset_dims={1,2},
    CHECK-SAME:     collapsed_slice_dims={},
    CHECK-SAME:     start_index_map={0},
    CHECK-SAME:     index_vector_dim=1,
    CHECK-SAME:     slice_sizes={7,8}
         CHECK: ROOT %{{.*}} = f32[42,43,7,8]{3,2,1,0} reshape(%[[GATHER]])
  )");
}

TEST_F(GatherSimplifierTest, RemovesCollapsedSliceDims) {
  // Verifies that GatherSimplifier sets the collapsed_slice_dims parameter to
  // the empty list.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34] parameter(0)
      indices = s32[42,1] parameter(1)
      ROOT gather = f32[42] gather(operand, indices),
          offset_dims={},
          collapsed_slice_dims={0,1},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,1}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
           CHECK: %[[GATHER:.*]] = f32[42,1,1]{2,1,0} gather(%operand, %indices)
      CHECK-SAME:     offset_dims={1,2},
      CHECK-SAME:     collapsed_slice_dims={},
           CHECK: ROOT %{{.*}} = f32[42]{0} reshape(%[[GATHER]])
  )");
}

TEST_F(GatherSimplifierTest, MakesStartIndexMapIdentity) {
  // Verifies that GatherSimplifier ensures start_index_map is {0, 1, ...}.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34,35] parameter(0)
      indices = s32[42,3] parameter(1)
      ROOT gather = f32[42,1,2,3] gather(operand, indices),
          offset_dims={1,2,3},
          collapsed_slice_dims={},
          start_index_map={2,0,1},
          index_vector_dim=1,
          slice_sizes={1,2,3}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
  %operand = f32[33,34,35]{2,1,0} parameter(0)
           CHECK: %[[OPERAND:.*]] = f32[35,33,34]{2,1,0} transpose(%operand)
           CHECK: %[[GATHER:.*]] = f32[42,3,1,2]{{.*}} gather(%[[OPERAND]],
      CHECK-SAME:    start_index_map={0,1,2},
           CHECK: ROOT {{.*}} = f32[42,1,2,3]{{.*}} transpose(%[[GATHER]])
  )");
}

TEST_F(GatherSimplifierTest, CollapsesSomeDims) {
  // Verifies that GatherSimplifier can collapse only some dimensions.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[33,34,35] parameter(0)
      indices = s32[42,1] parameter(1)
      ROOT gather = f32[7,42] gather(operand, indices),
          offset_dims={0},
          collapsed_slice_dims={0,2},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,7,1}
    })";

  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
           CHECK: %[[GATHER:.*]] = f32[42,1,7,1]{3,2,1,0} gather(
           CHECK: %[[COLLAPSED:.*]] = f32[42,7]{1,0} reshape(%[[GATHER]])
           CHECK: ROOT {{.*}} = f32[7,42]{1,0} transpose(%[[COLLAPSED]]),
      CHECK-SAME: dimensions={1,0}
  )");
}

TEST_F(GatherSimplifierTest, ZeroDimStartIndices) {
  // Verifies that a zero-dimensional start indices tensor doesn't contribute
  // any dimensions to the output.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[8,16] parameter(0)
      indices = s32[2] parameter(1)

      ROOT gather = f32[8,16] gather(f32[8,16] operand, s32[2] indices),
          offset_dims={0,1},
          collapsed_slice_dims={},
          start_index_map={0,1},
          index_vector_dim=0,
          slice_sizes={8,16}
    })";

  // The shape check is sufficient.
  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
      CHECK: gather(
  )");
}

TEST_F(GatherSimplifierTest, ZeroSizeSlice) {
  // Verifies that slices of size zero result in a constant result.
  constexpr absl::string_view kModuleStr = R"(
    HloModule gather_simplifier

    ENTRY kernel_entry {
      operand = f32[0,2] parameter(0)
      indices = s32[3] parameter(1)

      ROOT gather = f32[3,2] gather(f32[0,2] operand, s32[3]{0} indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={0,2}
    })";

  // The shape check is sufficient.
  RunAndFilecheckHloRewrite(kModuleStr, GatherSimplifier(), R"(
      CHECK: %[[ZERO:.*]] = f32[] constant(0) 
      CHECK: ROOT {{.*}} = f32[3,2]{1,0} broadcast(%[[ZERO]]), dimensions={} 
  )");
}

}  // namespace
}  // namespace xla
