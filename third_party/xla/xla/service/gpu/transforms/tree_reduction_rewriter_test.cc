/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/test.h"

namespace xla {

namespace {

class TreeReductionRewriterTest : public HloHardwareIndependentTestBase {
 public:
  void CheckTreeRewriter(absl::string_view hlo,
                         std::optional<absl::string_view> expected) {
    stream_executor::DeviceDescription device_description{
        stream_executor::GpuDeviceInfoProto{}};
    device_description.set_threads_per_warp(32);
    RunAndFilecheckHloRewrite(
        hlo, gpu::TreeReductionRewriter{device_description}, expected);
  }
};

TEST_F(TreeReductionRewriterTest, RowReductionSingleDimensionNoBatched) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[50021] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK: [[pad_0:%[^ ]+]] = f32[50022]{0} pad([[input_1:%[^ ]+]], [[zero_2:%[^ ]+]]), padding=0_1
// CHECK: [[bitcast_3:%[^ ]+]] = f32[126,397]{1,0} bitcast([[pad_0]])
// CHECK: [[reduce_4:%[^ ]+]] = f32[126]{0} reduce([[bitcast_3]], [[zero_2]]), dimensions={1}, to_apply=[[add_5:%[^ ]+]]
// CHECK: ROOT [[out_1_6:%[^ ]+]] = f32[] reduce([[reduce_4]], [[zero_2]]), dimensions={0}, to_apply=[[add_5]]
      )");
}

TEST_F(TreeReductionRewriterTest, RowReductionWeirdOutputLayout) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[2,4,17000]{2,1,0} parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[2,4]{0,1} reduce(input, zero), dimensions={2}, to_apply=add
}
)";

  // Check that we preserve the layout.
  CheckTreeRewriter(hlo,
                    R"(
// CHECK: f32[2,4]{0,1} reduce(
      )");
}

TEST_F(TreeReductionRewriterTest,
       RowReductionSingleDimensionNoBatchedDivisible) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[50048] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK: [[input_0:%[^ ]+]] = f32[50048]{0} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[128,391]{1,0} bitcast([[input_0]])
// CHECK: [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK: [[reduce_3:%[^ ]+]] = f32[128]{0} reduce([[bitcast_1]], [[zero_2]]), dimensions={1}, to_apply=[[add_4:%[^ ]+]]
// CHECK: ROOT [[out_1_5:%[^ ]+]] = f32[] reduce([[reduce_3]], [[zero_2]]), dimensions={0}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, RowReductionNoBatched) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[100,10,65536] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100,10] reduce(input, zero), dimensions={2}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK: [[bitcast_0:%[^ ]+]] = f32[100,10,256,256]{3,2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK: [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK: [[reduce_3:%[^ ]+]] = f32[100,10,256]{2,1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={3}, to_apply=[[add_4:%[^ ]+]]
// CHECK: ROOT [[out_1_5:%[^ ]+]] = f32[100,10]{1,0} reduce([[reduce_3]], [[zero_2]]), dimensions={2}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest,
       RowReductionSingleDimensionNoBatchedLargeInput) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1048576] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[1048576]{0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[1024,1024]{1,0} bitcast([[input_0]])
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[1024]{0} reduce([[bitcast_1]], [[zero_2]]), dimensions={1}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[] reduce([[reduce_3]], [[zero_2]]), dimensions={0}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, RowReductionBatchedDimensionFits) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[8,100,65536] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0,2}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK:  [[bitcast_0:%[^ ]+]] = f32[8,100,256,256]{3,2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[100,256]{1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={0,3}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[100]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={1}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, RowReductionBatchedDimensionDoesNotFit) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[32,100,90000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0,2}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK: [[reduce_0:%[^ ]+]] = f32[32,100]{1,0} reduce([[input_1:%[^ ]+]], [[zero_2:%[^ ]+]]), dimensions={2}, to_apply=[[add_3:%[^ ]+]]
// CHECK:  ROOT [[out_1_4:%[^ ]+]] = f32[100]{0} reduce([[reduce_0]], [[zero_2]]), dimensions={0}, to_apply=[[add_3]]
      )");
}

TEST_F(TreeReductionRewriterTest, ColumnReductionSimple) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[16384,100] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(

// CHECK:  [[input_0:%[^ ]+]] = f32[16384,100]{1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[128,128,100]{2,1,0} bitcast([[input_0]])
// CHECK:  [[reduce_2:%[^ ]+]] = f32[128,100]{1,0} reduce([[bitcast_1]], [[zero_3:%[^ ]+]]), dimensions={1}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[100]{0} reduce([[reduce_2]], [[zero_3]]), dimensions={0}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, ColumnReductionSimpleNoDivisible) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[10303,100] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[100] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[10303,100]{1,0} parameter(0)
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[pad_0:%[^ ]+]] = f32[10304,100]{1,0} pad([[input_1:%[^ ]+]], [[zero_2:%[^ ]+]]), padding=0_1x0_0
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[64,161,100]{2,1,0} bitcast([[pad_0]])
// CHECK:  [[reduce_3:%[^ ]+]] = f32[64,100]{1,0} reduce([[bitcast_1]], [[zero_2]]), dimensions={1}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[100]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={0}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, ColumnReductionOtherIndex) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[16384,2,2,2] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[2,2,2] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[16384,2,2,2]{3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[128,128,2,2,2]{4,3,2,1,0} bitcast([[input_0]])
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[128,2,2,2]{3,2,1,0} reduce([[bitcast_1]], [[zero_2]]), dimensions={1}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[2,2,2]{2,1,0} reduce([[reduce_3]], [[zero_2]]), dimensions={0}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, ColumnReductionVeryLargeInput) {
  const char* hlo = R"(
HloModule ReduceWithPadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1048576,5] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[5] reduce(input, zero), dimensions={0}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(

// CHECK:  [[bitcast_0:%[^ ]+]] = f32[1024,1024,5]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[1024,5]{1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={1}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[5]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={0}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, VariadicReductionLargeRow) {
  const char* hlo = R"(
HloModule Reduce_R1x2_to_R0x2_argmax

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[2,100003] parameter(0)
  idxs = u32[2,100003] iota(), iota_dimension=0
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[2], u32[2]) reduce(
    input, idxs, zero, zero_idx),
    dimensions={1},
    to_apply=%argmax
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK:  [[pad_0:%[^ ]+]] = f32[2,100005]{1,0} pad([[input_1:%[^ ]+]], [[zero_2:%[^ ]+]]), padding=0_0x0_2
// CHECK:  [[bitcast_3:%[^ ]+]] = f32[2,295,339]{2,1,0} bitcast([[pad_0]])
// CHECK:  [[zero_idx_4:%[^ ]+]] = u32[] constant(0)
// CHECK:  [[pad_1_5:%[^ ]+]] = u32[2,100005]{1,0} pad([[idxs_6:%[^ ]+]], [[zero_idx_4]]), padding=0_0x0_2
// CHECK:  [[bitcast_1_7:%[^ ]+]] = u32[2,295,339]{2,1,0} bitcast([[pad_1_5]])
// CHECK:  [[reduce_8:%[^ ]+]] = (f32[2,295]{1,0}, u32[2,295]{1,0}) reduce([[bitcast_3]], [[bitcast_1_7]], [[zero_2]], [[zero_idx_4]]), dimensions={2}, to_apply=[[argmax_9:%[^ ]+]]
// CHECK:  [[get_tuple_element_10:%[^ ]+]] = f32[2,295]{1,0} get-tuple-element([[reduce_8]]), index=0
// CHECK:  [[get_tuple_element_1_11:%[^ ]+]] = u32[2,295]{1,0} get-tuple-element([[reduce_8]]), index=1
// CHECK:  ROOT [[out_1_12:%[^ ]+]] = (f32[2]{0}, u32[2]{0}) reduce([[get_tuple_element_10]], [[get_tuple_element_1_11]], [[zero_2]], [[zero_idx_4]]), dimensions={1}, to_apply=[[argmax_9]]
      )");
}

TEST_F(TreeReductionRewriterTest, VariadicReductionLargeBatchSize) {
  const char* hlo = R"(
HloModule Reduce_R1x2_to_R0x2_argmax

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  input = f32[20,2,100] parameter(0)
  idxs = u32[20,2,100] iota(), iota_dimension=0
  zero = f32[] constant(0)
  zero_idx = u32[] constant(0)

  ROOT out = (f32[2], u32[2]) reduce(
    input, idxs, zero, zero_idx),
    dimensions={0,2},
    to_apply=%argmax
}
)";

  CheckTreeRewriter(hlo,
                    R"(
// CHECK:  [[reduce_0:%[^ ]+]] = (f32[20,2]{1,0}, u32[20,2]{1,0}) reduce([[input_1:%[^ ]+]], [[idxs_2:%[^ ]+]], [[zero_3:%[^ ]+]], [[zero_idx_4:%[^ ]+]]), dimensions={2}, to_apply=[[argmax_5:%[^ ]+]]
// CHECK:  [[get_tuple_element_6:%[^ ]+]] = f32[20,2]{1,0} get-tuple-element([[reduce_0]]), index=0
// CHECK:  [[get_tuple_element_1_7:%[^ ]+]] = u32[20,2]{1,0} get-tuple-element([[reduce_0]]), index=1
// CHECK:  ROOT [[out_1_8:%[^ ]+]] = (f32[2]{0}, u32[2]{0}) reduce([[get_tuple_element_6]], [[get_tuple_element_1_7]], [[zero_3]], [[zero_idx_4]]), dimensions={0}, to_apply=[[argmax_5]]
      )");
}

TEST_F(TreeReductionRewriterTest, KeepInnerReductionVectorized) {
  const char* hlo = R"(
HloModule KeepInnerRowReductionVectorized

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1024,73984] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[1024] reduce(input, zero), dimensions={1}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(

// CHECK:  [[bitcast_0:%[^ ]+]] = f32[1024,289,256]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[1024,289]{1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={2}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[1024]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={1}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, PreferLargeVectorizedDimension) {
  const char* hlo = R"(
HloModule PreferLargeVectorizedDimension

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1024,98304] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[1024] reduce(input, zero), dimensions={1}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(

// CHECK:  [[bitcast_0:%[^ ]+]] = f32[1024,256,384]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[1024,256]{1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={2}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[1024]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={1}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, SwapIfNonAlignedBeforePadding) {
  const char* hlo = R"(
HloModule SwapIfNonAlignedBeforePadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1024,19739] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[1024] reduce(input, zero), dimensions={1}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(

// CHECK-DAG:  [[bitcast_0:%[^ ]+]] = f32[1024,140,141]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK-DAG:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[1024,140]{1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={2}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[1024]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={1}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, DontSwapIfNonAlignedBeforePadding) {
  const char* hlo = R"(
HloModule DontSwapIfNonAlignedBeforePadding

add {
  accum = f32[] parameter(0)
  op = f32[] parameter(1)
  ROOT out = f32[] add(accum, op)
}

ENTRY main {
  input = f32[1024,19459] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[1024] reduce(input, zero), dimensions={1}, to_apply=add
}
)";

  CheckTreeRewriter(hlo,
                    R"(

// CHECK-DAG:  [[bitcast_0:%[^ ]+]] = f32[1024,140,139]{2,1,0} bitcast([[input_1:%[^ ]+]])
// CHECK-DAG:  [[zero_2:%[^ ]+]] = f32[] constant(0)
// CHECK:  [[reduce_3:%[^ ]+]] = f32[1024,140]{1,0} reduce([[bitcast_0]], [[zero_2]]), dimensions={2}, to_apply=[[add_4:%[^ ]+]]
// CHECK:  ROOT [[out_1_5:%[^ ]+]] = f32[1024]{0} reduce([[reduce_3]], [[zero_2]]), dimensions={1}, to_apply=[[add_4]]
      )");
}

TEST_F(TreeReductionRewriterTest, NonCosequtiveReductionDims) {
  const char* hlo = R"(
    HloModule NonCosequtiveReductionDims

    add {
      accum = f32[] parameter(0)
      op = f32[] parameter(1)
      ROOT out = f32[] add(accum, op)
    }

    ENTRY main {
      input = f32[5,3,4,5] parameter(0)
      zero = f32[] constant(0)
      ROOT out = f32[5,4] reduce(input, zero), dimensions={1,3}, to_apply=add
    }
  )";

  CheckTreeRewriter(hlo, std::nullopt);
}

}  // namespace
}  // namespace xla
