/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gather_scatter_normalizer.h"

#include <optional>

#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class GatherScatterNormalizerTest : public HloTestBase {};

TEST_F(GatherScatterNormalizerTest, NormalizeBatchGather) {
  // Verifies that GatherSimplifier
  // - Makes the index_vector_dim dimensions explicit
  // - Flattens start_indices into a 2d tensor.
  // - Undoes the flattening in the output.
  constexpr absl::string_view kModuleStr = R"(
HloModule StringifyGather, entry_computation_layout={(f32[50,49,48,47,46,512]{5,4,3,2,1,0}, s64[10,9,8,7,5,512]{5,4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0}}

ENTRY %Gather (input_tensor: f32[50,49,48,47,46,512], start_indices: s64[10,9,8,7,5,512]) -> f32[10,9,8,7,30,29,28,27,26,512] {
  %input_tensor = f32[50,49,48,47,46,512]{5,4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5,512]{5,4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0}
    gather(f32[50,49,48,47,46,512]{5,4,3,2,1,0} %input_tensor, s64[10,9,8,7,5,512]{5,4,3,2,1,0} %start_indices),
    offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, operand_batching_dims={5},
    start_indices_batching_dims={5}, index_vector_dim=4, slice_sizes={30,29,28,27,26,1}
})";

  RunAndFilecheckHloRewrite(kModuleStr, GatherScatterNormalizer(), R"(
         CHECK: %[[IOTA:.*]] = s64[10,9,8,7,1,512]{{.*}} iota()
         CHECK: %[[INDICES_CONCAT:.*]] = s64[10,9,8,7,6,512]{{.*}} concatenate(%start_indices, %[[IOTA]])
         CHECK: ROOT %[[GATHER:.*]] = f32[10,9,8,7,30,29,28,27,26,512]{{.*}} gather(
    CHECK-SAME:     %input_tensor, %[[INDICES_CONCAT]]),
    CHECK-SAME:     offset_dims={4,5,6,7,8},
    CHECK-SAME:     collapsed_slice_dims={5},
    CHECK-SAME:     start_index_map={0,1,2,3,4,5},
    CHECK-SAME:     index_vector_dim=4,
    CHECK-SAME:     slice_sizes={30,29,28,27,26,1}
  )");
}

TEST_F(GatherScatterNormalizerTest, NormalizeBatchScatter) {
  // Verifies that GatherSimplifier
  // - Makes the index_vector_dim dimensions explicit
  // - Flattens start_indices into a 2d tensor.
  // - Undoes the flattening in the output.
  constexpr absl::string_view kModuleStr = R"(

HloModule StringifyScatter, entry_computation_layout={(f32[50,49,48,47,46,512]{5,4,3,2,1,0}, s64[10,9,8,7,5,512]{5,4,3,2,1,0}, f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46,512]{5,4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46,512], scatter_indices: s64[10,9,8,7,5,512], updates: f32[10,9,8,7,30,29,28,27,26,512]) -> f32[50,49,48,47,46,512] {
  %input_tensor = f32[50,49,48,47,46,512]{5,4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5,512]{5,4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46,512]{5,4,3,2,1,0} scatter(
    f32[50,49,48,47,46,512]{5,4,3,2,1,0} %input_tensor,
    s64[10,9,8,7,5,512]{5,4,3,2,1,0} %scatter_indices,
    f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0} %updates),
    update_window_dims={4,5,6,7,8}, inserted_window_dims={},
    scatter_dims_to_operand_dims={0,1,2,3,4}, input_batching_dims={5},
    scatter_indices_batching_dims={5}, index_vector_dim=4, to_apply=%add_F32.v3
})";

  RunAndFilecheckHloRewrite(kModuleStr, GatherScatterNormalizer(), R"(
         CHECK: %[[IOTA:.*]] = s64[10,9,8,7,1,512]{{.*}} iota()
         CHECK: %[[INDICES_CONCAT:.*]] = s64[10,9,8,7,6,512]{{.*}} concatenate(%start_indices, %[[IOTA]])
         CHECK: ROOT %[[SCATTER:.*]] = f32[50,49,48,47,46,512]{{.*}} scatter(
    CHECK-SAME:     %input_tensor, %[[INDICES_CONCAT]]),
    CHECK-SAME:     update_window_dims={4,5,6,7,8},
    CHECK-SAME:     inserted_window_dims={5},
    CHECK-SAME:     scatter_dims_to_operand_dims={0,1,2,3,4,5},
    CHECK-SAME:     index_vector_dim=4,
    CHECK-SAME:     to_apply=%add_F32.v3
  )");
}

}  // namespace
}  // namespace xla
