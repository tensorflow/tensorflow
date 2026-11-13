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

#include "xla/pjrt/dynamic_pad_to_static.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/infer_dispatch_info.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/device_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

std::vector<Shape> MakeCanonicalDynamicParameterShapes(
    const HloModule& module) {
  std::vector<Shape> shapes =
      GetParameterShapes(module.entry_computation_layout());
  for (Shape& shape : shapes) {
    ShapeUtil::ForEachMutableSubshape(
        &shape, [](Shape* subshape, const ShapeIndex&) {
          if (subshape->IsArray() && subshape->is_dynamic() &&
              subshape->has_layout()) {
            subshape->mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(
                1024);
          }
        });
  }
  return shapes;
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsPadRealToStaticAndPreservesPadToStatic) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule pad_real_vs_pad_std

ENTRY main {
  p0 = bf16[<=16,32]{1,0} parameter(0)
  p1 = f32[<=16,32]{1,0} parameter(1)
  p2 = s32[<=16,32]{1,0} parameter(2)
  d0 = s32[] parameter(3)
  d1 = s32[] parameter(4)
  bitcast_p2 = s32[<=16,32]{1,0} bitcast(p2)
  pad_real_0 = bf16[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_std_1 = (f32[16,32]{1,0}, s32[], s32[]) custom-call(p1),
      custom_call_target="PadToStatic"
  pad_real_2 = s32[16,32]{1,0} custom-call(bitcast_p2, d0, d1),
      custom_call_target="PadRealToStatic"
  gte_1 = f32[16,32]{1,0} get-tuple-element(pad_std_1), index=0
  ROOT out = (bf16[16,32]{1,0}, f32[16,32]{1,0}, s32[16,32]{1,0})
      tuple(pad_real_0, gte_1, pad_real_2)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsWideParametersReadByPlaneCustomCalls) {
  // `pad_real_to_static_plane_index` picks one 32-bit plane of a wide input.
  constexpr absl::string_view kHlo = R"hlo(
HloModule wide_pad_real_to_static

ENTRY main {
  p0_s64 = s64[<=16,32]{1,0} parameter(0)
  p1_f64 = f64[<=16,32]{1,0} parameter(1)
  p2_c128 = c128[<=16,32]{1,0} parameter(2)
  p3_s64_std = s64[<=16,32]{1,0} parameter(3)
  d0 = s32[] parameter(4)
  d1 = s32[] parameter(5)

  pad_s64_lo = u32[16,32]{1,0} custom-call(p0_s64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":0}
  pad_s64_hi = u32[16,32]{1,0} custom-call(p0_s64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
  pad_f64_lo = f32[16,32]{1,0} custom-call(p1_f64, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
  pad_c128 = f32[16,32]{1,0} custom-call(p2_c128, d0, d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":3}

  std_lo = u32[<=16,32]{1,0} custom-call(p3_s64_std),
      custom_call_target="X64SplitLow"
  pad_std = (u32[16,32]{1,0}, s32[], s32[]) custom-call(std_lo),
      custom_call_target="PadToStatic"
  pad_std_data = u32[16,32]{1,0} get-tuple-element(pad_std), index=0

  ROOT out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0},
              f32[16,32]{1,0}, u32[16,32]{1,0})
      tuple(pad_s64_lo, pad_s64_hi, pad_f64_lo, pad_c128, pad_std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[3].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsThroughEmbeddedOffloadDispatcherChain) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule offloaded_pad_real_to_static

%called_computation.2 (param_0.6: f32[<=256], param_1.8: s32[]) -> f32[256] {
  %param_0.6 = f32[<=256]{0} parameter(0)
  %param_1.8 = s32[] parameter(1)
  ROOT %pad_real_to_static.6 = f32[256]{0} custom-call(%param_0.6, %param_1.8),
      custom_call_target="PadRealToStatic"
}, execution_thread="sparsecore"

%async_computation.2 (param_0.7: f32[<=256], param_1.9: s32[]) -> f32[256] {
  %param_0.7 = f32[<=256]{0} parameter(0)
  %param_1.9 = s32[] parameter(1)
  ROOT %call = f32[256]{0} call(%param_0.7, %param_1.9),
      to_apply=%called_computation.2
}, execution_thread="sparsecore"

%called_computation.3 (param_0.8: f32[<=256], param_1.10: s32[]) -> f32[256] {
  %param_0.8 = f32[<=256]{0} parameter(0)
  %param_1.10 = s32[] parameter(1)
  %tuple.2 = (f32[<=256]{0}, s32[]) tuple(%param_0.8, %param_1.10)
  %call-start = ((f32[<=256]{0}, s32[]), f32[256]{0}, u32[])
      async-start(%tuple.2#0, %tuple.2#1),
      async_execution_thread="sparsecore", calls=%async_computation.2
  ROOT %call-done = f32[256]{0} async-done(%call-start)
}, execution_thread="sparsecore"

%async_computation (param_0.5: f32[<=256], param_1.7: s32[]) -> f32[256] {
  %param_0.5 = f32[<=256]{0} parameter(0)
  %param_1.7 = s32[] parameter(1)
  ROOT %custom-call = f32[256]{0} custom-call(%param_0.5, %param_1.7),
      custom_call_target="SparseCoreCaller",
      called_computations={%called_computation.3}
}, execution_thread="sparsecore"

ENTRY %main (param: f32[<=256]) -> f32[256] {
  %param = f32[<=256]{0} parameter(0)
  %size0 = s32[] constant(1)
  %offload_start = ((), (), (s32[12]{0})) async-start(),
      async_execution_thread="sparsecore", calls=%async_computation
  %offload_update = ((f32[<=256]{0}, s32[]), f32[256]{0}, (s32[12]{0}))
      async-update(%offload_start, %param, %size0),
      output_to_operand_aliasing={{2,0}: (0, {2,0})}
  ROOT %offload_done = f32[256]{0} async-done(%offload_update)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsThroughAsyncStartAndAsyncUpdateChain) {
  // Dataflow forwards the entry parameter across `async-update` and `call`.
  constexpr absl::string_view kHlo = R"hlo(
HloModule async_offload_pad_real_to_static

%leaf_lo (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":0}
}, execution_thread="sparsecore"

%leaf_hi (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic",
      backend_config={"pad_real_to_static_plane_index":1}
}, execution_thread="sparsecore"

%outer_async_lo (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %call = u32[16,32]{1,0} call(%out_p0, %out_d0, %out_d1),
      to_apply=%leaf_lo
}, execution_thread="sparsecore"

%outer_async_hi (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %call = u32[16,32]{1,0} call(%out_p0, %out_d0, %out_d1),
      to_apply=%leaf_hi
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], u32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %d0 = s32[] parameter(2)
  %d1 = s32[] parameter(3)
  %start_lo = ((), (), u32[]) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_lo
  %update_lo = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0}, u32[])
      async-update(%start_lo, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_lo = u32[16,32]{1,0} async-done(%update_lo),
      async_execution_thread="sparsecore"

  %start_hi = ((), (), u32[]) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_hi
  %update_hi = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0}, u32[])
      async-update(%start_hi, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_hi = u32[16,32]{1,0} async-done(%update_hi),
      async_execution_thread="sparsecore"

  %std_lo = u32[<=16,32]{1,0} custom-call(%p1),
      custom_call_target="X64SplitLow"
  %pad_std = (u32[16,32]{1,0}, s32[], s32[]) custom-call(%std_lo),
      custom_call_target="PadToStatic"
  %std_data = u32[16,32]{1,0} get-tuple-element(%pad_std), index=0

  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, u32[16,32]{1,0})
      tuple(%done_lo, %done_hi, %std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsTupledEntryParameterElement) {
  // `GetParameterShapes` flattens a single tuple parameter into its elements.
  constexpr absl::string_view kHlo = R"hlo(
HloModule tupled_entry_pad_real_to_static

ENTRY main {
  arg_tuple = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s64[<=16,32]{1,0},
               s32[], s32[]) parameter(0)
  arg_copy = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s64[<=16,32]{1,0},
              s32[], s32[]) copy(arg_tuple)
  p0 = s64[<=16,32]{1,0} get-tuple-element(arg_copy), index=0
  p1 = f32[<=16,32]{1,0} get-tuple-element(arg_copy), index=1
  p2 = s64[<=16,32]{1,0} get-tuple-element(arg_copy), index=2
  d0 = s32[] get-tuple-element(arg_copy), index=3
  d1 = s32[] get-tuple-element(arg_copy), index=4
  nested = (s64[<=16,32]{1,0}, s32[]) tuple(p2, d0)
  nested2 = ((s64[<=16,32]{1,0}, s32[]), s32[]) tuple(nested, d1)
  nested_inner = (s64[<=16,32]{1,0}, s32[]) get-tuple-element(nested2), index=0
  nested_p2 = s64[<=16,32]{1,0} get-tuple-element(nested_inner), index=0
  pad_real = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_nested = u32[16,32]{1,0} custom-call(nested_p2, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(p1),
      custom_call_target="PadToStatic"
  gte = f32[16,32]{1,0} get-tuple-element(pad_std), index=0
  ROOT out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(pad_real, pad_nested, gte)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_EQ(param_shapes.size(), 5);
  ASSERT_FALSE(param_shapes[0].IsTuple());
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsParameterWithOtherUsersWithoutPartialMutation) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule shared_pad_real_to_static_parameter

ENTRY main {
  p_valid = bf16[<=16,32]{1,0} parameter(0)
  p_shared = s64[<=16,32]{1,0} parameter(1)
  d0 = s32[] parameter(2)
  d1 = s32[] parameter(3)
  pad_valid = bf16[16,32]{1,0} custom-call(p_valid, d0, d1),
      custom_call_target="PadRealToStatic"
  pad_shared = u32[16,32]{1,0} custom-call(p_shared, d0, d1),
      custom_call_target="PadRealToStatic"
  dyn_sum = s64[<=16,32]{1,0} add(p_shared, p_shared)
  ROOT out = (bf16[16,32]{1,0}, u32[16,32]{1,0}, s64[<=16,32]{1,0})
      tuple(pad_valid, pad_shared, dyn_sum)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 1"),
                     HasSubstr("PadRealToStatic"), HasSubstr("dyn_sum"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsTupleParameterElementWithOtherUsers) {
  // The same tuple element read through a second get-tuple-element.
  constexpr absl::string_view kHlo = R"hlo(
HloModule shared_tuple_pad_real_to_static_parameter

ENTRY main {
  arg = (f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  p = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  p_again = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  d0 = s32[] get-tuple-element(arg), index=1
  d1 = s32[] get-tuple-element(arg), index=2
  pad = f32[16,32]{1,0} custom-call(p, d0, d1),
      custom_call_target="PadRealToStatic"
  neg = f32[<=16,32]{1,0} negate(p_again)
  ROOT out = (f32[16,32]{1,0}, f32[<=16,32]{1,0}) tuple(pad, neg)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0 {0}"), HasSubstr("neg"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsReturnedParameter) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule returned_pad_real_to_static_parameter

ENTRY main {
  p0 = f32[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  pad = f32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  ROOT out = (f32[16,32]{1,0}, f32[<=16,32]{1,0}) tuple(pad, p0)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("returned as a result of computation main"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsAliasedPadRealToStaticParameter) {
  constexpr absl::string_view kDirectAliasHlo = R"hlo(
HloModule aliased_pad_real_to_static,
    input_output_alias={ {}: (0, {}, may-alias) }

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ROOT pad = s64[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> direct_module,
                       ParseAndReturnUnverifiedModule(kDirectAliasHlo));
  std::vector<Shape> direct_shapes =
      MakeCanonicalDynamicParameterShapes(*direct_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *direct_module, absl::MakeSpan(direct_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("PadRealToStatic"), HasSubstr("aliased"))));
  EXPECT_EQ(direct_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kTupleLeafAliasHlo = R"hlo(
HloModule aliased_tuple_pad_real_to_static,
    input_output_alias={ {0}: (0, {0}, must-alias) }

ENTRY main {
  arg = (f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  p = f32[<=16,32]{1,0} get-tuple-element(arg), index=0
  d0 = s32[] get-tuple-element(arg), index=1
  d1 = s32[] get-tuple-element(arg), index=2
  pad = f32[16,32]{1,0} custom-call(p, d0, d1),
      custom_call_target="PadRealToStatic"
  ROOT out = (f32[16,32]{1,0}, s32[]) tuple(pad, d0)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> tuple_module,
                       ParseAndReturnUnverifiedModule(kTupleLeafAliasHlo));
  std::vector<Shape> tuple_shapes =
      MakeCanonicalDynamicParameterShapes(*tuple_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *tuple_module, absl::MakeSpan(tuple_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0 {0}"),
                     HasSubstr("PadRealToStatic"), HasSubstr("aliased"))));
  EXPECT_EQ(tuple_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     AllowsStaticAliasedPadRealToStaticParameter) {
  constexpr absl::string_view kStaticAliasHlo = R"hlo(
HloModule static_aliased_pad_real_to_static,
    input_output_alias={ {}: (0, {}, may-alias) }

ENTRY main {
  p0 = s32[16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ROOT pad = s32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kStaticAliasHlo));
  std::vector<Shape> shapes = MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(shapes)));
  EXPECT_EQ(shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsDynamicArrayCopyBeforePadRealToStatic) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule copy_before_pad_real_to_static

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  copy_p0 = s64[<=16,32]{1,0} copy(p0)
  ROOT pad = u32[16,32]{1,0} custom-call(copy_p0, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("PadRealToStatic"),
                     HasSubstr("entry computation parameter"),
                     HasSubstr("copy_p0"), HasSubstr("in computation main"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsNonParameterDynamicOperand) {
  constexpr absl::string_view kSliceToDynamicHlo = R"hlo(
HloModule slice_to_dynamic_into_pad_real_to_static

ENTRY main {
  p0 = s32[16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  slice_dyn = s32[<=16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="SliceToDynamic"
  ROOT pad = s32[16,32]{1,0} custom-call(slice_dyn, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> slice_module,
                       ParseAndReturnUnverifiedModule(kSliceToDynamicHlo));
  std::vector<Shape> slice_shapes =
      MakeCanonicalDynamicParameterShapes(*slice_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *slice_module, absl::MakeSpan(slice_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("PadRealToStatic"),
                             HasSubstr("entry computation parameter"),
                             HasSubstr("slice_dyn"))));

  constexpr absl::string_view kAllGatherHlo = R"hlo(
HloModule all_gather_into_pad_real_to_static

ENTRY main {
  p0 = f32[<=8,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ag = f32[<=16,32]{1,0} all-gather(p0), dimensions={0}, replica_groups={{0,1}}
  ROOT pad = f32[16,32]{1,0} custom-call(ag, d0, d1),
      custom_call_target="PadRealToStatic"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ag_module,
                       ParseAndReturnUnverifiedModule(kAllGatherHlo));
  std::vector<Shape> ag_shapes =
      MakeCanonicalDynamicParameterShapes(*ag_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *ag_module, absl::MakeSpan(ag_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("PadRealToStatic"),
                             HasSubstr("entry computation parameter"),
                             HasSubstr("all-gather"))));
  EXPECT_EQ(ag_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsOtherCustomCallReadingParameter) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule other_custom_call_reading_parameter

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  pad = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  other_cc = s32[] custom-call(p0), custom_call_target="OtherTarget"
  ROOT out = (u32[16,32]{1,0}, s32[]) tuple(pad, other_cc)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("other_cc"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsCustomCallWithCalledComputationSharingParameter) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule embedded_custom_call_sharing_parameter

%unused_param_comp (sub_p0: s64[<=16,32]) -> s32[] {
  %sub_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %c = s32[] constant(0)
}

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  pad = u32[16,32]{1,0} custom-call(p0, d0, d1),
      custom_call_target="PadRealToStatic"
  other_cc = s32[] custom-call(p0),
      custom_call_target="OtherTarget",
      called_computations={%unused_param_comp}
  ROOT out = (u32[16,32]{1,0}, s32[]) tuple(pad, other_cc)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("other_cc"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsOffloadedSubcomputationReturningParameter) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule offloaded_subcomputation_escaping_parameter

%leaf_comp (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
  ROOT %escaped = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) tuple(%pad, %leaf_p0)
}

ENTRY main {
  p0 = s64[<=16,32]{1,0} parameter(0)
  d0 = s32[] parameter(1)
  d1 = s32[] parameter(2)
  ROOT res = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) call(p0, d0, d1),
      to_apply=%leaf_comp
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *module, absl::MakeSpan(param_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("returned as a result of computation"))));
  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsAliasedAsyncStartOperand) {
  constexpr absl::string_view kAliasedAsyncStartHlo = R"hlo(
HloModule aliased_async_start_pad_real_to_static

%async_pad (ap0: s32[<=16,32], ad0: s32[], ad1: s32[]) -> s32[16,32] {
  %ap0 = s32[<=16,32]{1,0} parameter(0)
  %ad0 = s32[] parameter(1)
  %ad1 = s32[] parameter(2)
  ROOT %pad = s32[16,32]{1,0} custom-call(%ap0, %ad0, %ad1),
      custom_call_target="PadRealToStatic"
}, execution_thread="sparsecore"

ENTRY %main (p0: s32[<=16,32], d0: s32[], d1: s32[]) -> s32[16,32] {
  %p0 = s32[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start = ((s32[<=16,32]{1,0}, s32[], s32[]), s32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore",
      output_to_operand_aliasing={{1}: (0, {})},
      calls=%async_pad
  ROOT %done = s32[16,32]{1,0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> aliased_async_module,
                       ParseAndReturnUnverifiedModule(kAliasedAsyncStartHlo));
  std::vector<Shape> aliased_async_shapes =
      MakeCanonicalDynamicParameterShapes(*aliased_async_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *aliased_async_module, absl::MakeSpan(aliased_async_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("%start"),
                     HasSubstr("output_to_operand_aliasing"))));
  EXPECT_EQ(
      aliased_async_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
      1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsEmbeddedAsyncWrappedSubcomputationCall) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule embedded_sparsecore_async_pad_real_to_static

%sc_plane_lo (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%sc_plane_hi_tupled (leaf_tup: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %leaf_tup = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %leaf_p0 = s64[<=16,32]{1,0} get-tuple-element(%leaf_tup), index=0
  %leaf_d0 = s32[] get-tuple-element(%leaf_tup), index=1
  %leaf_d1 = s32[] get-tuple-element(%leaf_tup), index=2
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap_p0_lo (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_plane_lo
}, execution_thread="sparsecore"

%async_wrap_p0_hi (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  %packed = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%out_p0, %out_d0, %out_d1)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%packed),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_plane_hi_tupled
}, execution_thread="sparsecore"

%async_wrap_p1_lo (out_p1: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p1 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p1, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_plane_lo
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], p2: f32[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], u32[16,32], f32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %p2 = f32[<=16,32]{1,0} parameter(2)
  %d0 = s32[] parameter(3)
  %d1 = s32[] parameter(4)

  %start_p0_lo = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_p0_lo
  %update_p0_lo = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-update(%start_p0_lo, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_p0_lo = u32[16,32]{1,0} async-done(%update_p0_lo),
      async_execution_thread="sparsecore"

  %start_p0_hi = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_p0_hi
  %update_p0_hi = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-update(%start_p0_hi, %p0, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_p0_hi = u32[16,32]{1,0} async-done(%update_p0_hi),
      async_execution_thread="sparsecore"

  %start_p1_lo = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_p1_lo
  %update_p1_lo = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-update(%start_p1_lo, %p1, %d0, %d1),
      async_execution_thread="sparsecore"
  %done_p1_lo = u32[16,32]{1,0} async-done(%update_p1_lo),
      async_execution_thread="sparsecore"

  %pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(%p2),
      custom_call_target="PadToStatic"
  %std_data = f32[16,32]{1,0} get-tuple-element(%pad_std), index=0

  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(%done_p0_lo, %done_p0_hi, %done_p1_lo, %std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsEmbeddedAsyncWrappedSubcomputationMisuse) {
  constexpr absl::string_view kEscapedInEmbeddedHlo = R"hlo(
HloModule embedded_sparsecore_escaping_parameter

%sc_escape (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
  ROOT %escaped = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) tuple(%pad, %leaf_p0)
}

%async_wrap (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = (u32[16,32]{1,0}, s64[<=16,32]{1,0})
      custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_escape
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start = ((s64[<=16,32]{1,0}, s32[], s32[]), (u32[16,32]{1,0}, s64[<=16,32]{1,0}))
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap
  ROOT %done = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> escaped_module,
                       ParseAndReturnUnverifiedModule(kEscapedInEmbeddedHlo));
  std::vector<Shape> escaped_shapes =
      MakeCanonicalDynamicParameterShapes(*escaped_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *escaped_module, absl::MakeSpan(escaped_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"),
                     HasSubstr("returned as a result of computation"))));
  EXPECT_EQ(escaped_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kSharedNonPadAsyncHlo = R"hlo(
HloModule embedded_sparsecore_shared_with_non_pad_offload

%sc_pad (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%sc_other (other_p0: s64[<=16,32]) -> s32[] {
  %other_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %zero = s32[] constant(0)
}

%async_wrap_pad (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_pad
}, execution_thread="sparsecore"

%async_wrap_other (out_p0: s64[<=16,32]) -> s32[] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %sc_other_call = s32[] custom-call(%out_p0),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_other
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s32[]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start_pad = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_pad
  %done_pad = u32[16,32]{1,0} async-done(%start_pad),
      async_execution_thread="sparsecore"
  %start_other = ((s64[<=16,32]{1,0}), s32[]) async-start(%p0),
      async_execution_thread="sparsecore", calls=%async_wrap_other
  %done_other = s32[] async-done(%start_other),
      async_execution_thread="sparsecore"
  ROOT %out = (u32[16,32]{1,0}, s32[]) tuple(%done_pad, %done_other)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> shared_module,
                       ParseAndReturnUnverifiedModule(kSharedNonPadAsyncHlo));
  std::vector<Shape> shared_shapes =
      MakeCanonicalDynamicParameterShapes(*shared_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *shared_module, absl::MakeSpan(shared_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("sc_other_call"),
                     HasSubstr("in computation async_wrap_other"))));
  EXPECT_EQ(shared_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kAliasedEmbeddedCallHlo = R"hlo(
HloModule embedded_sparsecore_aliased_operand

%sc_pad (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap_alias (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_aliased = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall",
      output_to_operand_aliasing={{}: (0, {})},
      to_apply=%sc_pad
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> u32[16,32] {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_alias
  ROOT %done = u32[16,32]{1,0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> aliased_module,
                       ParseAndReturnUnverifiedModule(kAliasedEmbeddedCallHlo));
  std::vector<Shape> aliased_shapes =
      MakeCanonicalDynamicParameterShapes(*aliased_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *aliased_module, absl::MakeSpan(aliased_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 0"), HasSubstr("sc_aliased"))));
  EXPECT_EQ(aliased_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kCopyIntoEmbeddedHlo = R"hlo(
HloModule embedded_sparsecore_copy_before_offload

%sc_pad (leaf_p0: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p0 = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p0, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap (out_p0: s64[<=16,32], out_d0: s32[], out_d1: s32[]) -> u32[16,32] {
  %out_p0 = s64[<=16,32]{1,0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %out_d1 = s32[] parameter(2)
  ROOT %sc_call = u32[16,32]{1,0} custom-call(%out_p0, %out_d0, %out_d1),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_pad
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> u32[16,32] {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %copy_p0 = s64[<=16,32]{1,0} copy(%p0)
  %start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%copy_p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap
  ROOT %done = u32[16,32]{1,0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> copy_module,
                       ParseAndReturnUnverifiedModule(kCopyIntoEmbeddedHlo));
  std::vector<Shape> copy_shapes =
      MakeCanonicalDynamicParameterShapes(*copy_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *copy_module, absl::MakeSpan(copy_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("entry computation parameter"),
                     HasSubstr("copy_p0"), HasSubstr("in computation main"))));
  EXPECT_EQ(copy_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsPrefixAcrossPrefetchedDeduplicatedAndNestedAsyncTileOffloads) {
  constexpr absl::string_view kFiveLayerDeduplicatedHlo = R"hlo(
HloModule prefetched_deduplicated_nested_async_tile_offload

%inner_tile_callee (leaf_p: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%inner_tile_async_wrap (tile_p: s64[<=16,32], tile_d0: s32[], tile_d1: s32[]) -> u32[16,32] {
  %tile_p = s64[<=16,32]{1,0} parameter(0)
  %tile_d0 = s32[] parameter(1)
  %tile_d1 = s32[] parameter(2)
  ROOT %inner_call = u32[16,32]{1,0} call(%tile_p, %tile_d0, %tile_d1),
      to_apply=%inner_tile_callee
}, execution_thread="sparsecore"

%reconstructed_scs_comp (scs_tuple_param: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %scs_tuple_param = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %scs_p = s64[<=16,32]{1,0} get-tuple-element(%scs_tuple_param), index=0
  %scs_d0 = s32[] get-tuple-element(%scs_tuple_param), index=1
  %scs_d1 = s32[] get-tuple-element(%scs_tuple_param), index=2
  %tile_start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%scs_p, %scs_d0, %scs_d1),
      async_execution_thread="sparsecore", calls=%inner_tile_async_wrap
  ROOT %tile_done = u32[16,32]{1,0} async-done(%tile_start),
      async_execution_thread="sparsecore"
}

%outer_async_wrap_0 (outer_tuple_0: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %outer_tuple_0 = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call_0 = u32[16,32]{1,0} custom-call(%outer_tuple_0),
      custom_call_target="AsyncOffloadCall", to_apply=%reconstructed_scs_comp
}, execution_thread="sparsecore"

%outer_async_wrap_1 (outer_tuple_1: (s64[<=16,32], s32[], s32[])) -> u32[16,32] {
  %outer_tuple_1 = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call_1 = u32[16,32]{1,0} custom-call(%outer_tuple_1),
      custom_call_target="AsyncOffloadCall", to_apply=%reconstructed_scs_comp
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], p2: f32[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], f32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %p2 = f32[<=16,32]{1,0} parameter(2)
  %d0 = s32[] parameter(3)
  %d1 = s32[] parameter(4)
  %tuple0 = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%p0, %d0, %d1)
  %start0 = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_wrap_0
  %update0 = (((s64[<=16,32]{1,0}, s32[], s32[])), u32[16,32]{1,0})
      async-update(%start0, %tuple0), async_execution_thread="sparsecore"
  %done0 = u32[16,32]{1,0} async-done(%update0),
      async_execution_thread="sparsecore"
  %tuple1 = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%p1, %d0, %d1)
  %start1 = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_wrap_1
  %update1 = (((s64[<=16,32]{1,0}, s32[], s32[])), u32[16,32]{1,0})
      async-update(%start1, %tuple1), async_execution_thread="sparsecore"
  %done1 = u32[16,32]{1,0} async-done(%update1),
      async_execution_thread="sparsecore"
  %pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(%p2),
      custom_call_target="PadToStatic"
  %pad_std_data = f32[16,32]{1,0} get-tuple-element(%pad_std), index=0
  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, f32[16,32]{1,0})
      tuple(%done0, %done1, %pad_std_data)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnUnverifiedModule(kFiveLayerDeduplicatedHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[2].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsNestedAsyncTileSharedReader) {
  constexpr absl::string_view kInnerTileSharedReaderHlo = R"hlo(
HloModule nested_async_tile_shared_reader

%inner_tile_callee (leaf_p: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%inner_tile_async_wrap (tile_p: s64[<=16,32], tile_d0: s32[], tile_d1: s32[]) -> u32[16,32] {
  %tile_p = s64[<=16,32]{1,0} parameter(0)
  %tile_d0 = s32[] parameter(1)
  %tile_d1 = s32[] parameter(2)
  ROOT %inner_call = u32[16,32]{1,0} call(%tile_p, %tile_d0, %tile_d1),
      to_apply=%inner_tile_callee
}, execution_thread="sparsecore"

%other_tile_wrap (other_p: s64[<=16,32]) -> s64[<=16,32] {
  %other_p = s64[<=16,32]{1,0} parameter(0)
  ROOT %sum = s64[<=16,32]{1,0} add(%other_p, %other_p)
}, execution_thread="sparsecore"

%reconstructed_scs_comp (scs_tuple_param: (s64[<=16,32], s32[], s32[])) -> (u32[16,32], s64[<=16,32]) {
  %scs_tuple_param = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %scs_p = s64[<=16,32]{1,0} get-tuple-element(%scs_tuple_param), index=0
  %scs_d0 = s32[] get-tuple-element(%scs_tuple_param), index=1
  %scs_d1 = s32[] get-tuple-element(%scs_tuple_param), index=2
  %tile_start = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%scs_p, %scs_d0, %scs_d1),
      async_execution_thread="sparsecore", calls=%inner_tile_async_wrap
  %tile_done = u32[16,32]{1,0} async-done(%tile_start),
      async_execution_thread="sparsecore"
  %other_tile_start = ((s64[<=16,32]{1,0}), s64[<=16,32]{1,0})
      async-start(%scs_p),
      async_execution_thread="sparsecore", calls=%other_tile_wrap
  %other_tile_done = s64[<=16,32]{1,0} async-done(%other_tile_start),
      async_execution_thread="sparsecore"
  ROOT %res = (u32[16,32]{1,0}, s64[<=16,32]{1,0})
      tuple(%tile_done, %other_tile_done)
}

%outer_async_wrap (outer_tuple: (s64[<=16,32], s32[], s32[])) -> (u32[16,32], s64[<=16,32]) {
  %outer_tuple = (s64[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call = (u32[16,32]{1,0}, s64[<=16,32]{1,0})
      custom-call(%outer_tuple),
      custom_call_target="AsyncOffloadCall", to_apply=%reconstructed_scs_comp
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s64[<=16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %tuple0 = (s64[<=16,32]{1,0}, s32[], s32[]) tuple(%p0, %d0, %d1)
  %start0 = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%outer_async_wrap
  %update0 = (((s64[<=16,32]{1,0}, s32[], s32[])), (u32[16,32]{1,0}, s64[<=16,32]{1,0}))
      async-update(%start0, %tuple0), async_execution_thread="sparsecore"
  ROOT %done0 = (u32[16,32]{1,0}, s64[<=16,32]{1,0}) async-done(%update0),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> shared_tile_module,
      ParseAndReturnUnverifiedModule(kInnerTileSharedReaderHlo));
  std::vector<Shape> shared_tile_shapes =
      MakeCanonicalDynamicParameterShapes(*shared_tile_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *shared_tile_module, absl::MakeSpan(shared_tile_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry parameter 0"),
                             HasSubstr("other_tile_start"))));
  EXPECT_EQ(
      shared_tile_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
      1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsUnusedParameterCallAndAsyncStartSharingParameter) {
  constexpr absl::string_view kUnusedAsyncParamHlo = R"hlo(
HloModule unused_async_param_sharing_pad_parameter

%unused_async (unused_p0: s64[<=16,32]) -> s32[] {
  %unused_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %zero = s32[] constant(0)
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s32[]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %pad = u32[16,32]{1,0} custom-call(%p0, %d0, %d1),
      custom_call_target="PadRealToStatic"
  %start_unused = ((s64[<=16,32]{1,0}), s32[]) async-start(%p0),
      async_execution_thread="sparsecore", calls=%unused_async
  %done_unused = s32[] async-done(%start_unused),
      async_execution_thread="sparsecore"
  ROOT %out = (u32[16,32]{1,0}, s32[]) tuple(%pad, %done_unused)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> async_module,
                       ParseAndReturnUnverifiedModule(kUnusedAsyncParamHlo));
  std::vector<Shape> async_shapes =
      MakeCanonicalDynamicParameterShapes(*async_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *async_module, absl::MakeSpan(async_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry parameter 0"),
                             HasSubstr("start_unused"))));
  EXPECT_EQ(async_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kUnusedCallParamHlo = R"hlo(
HloModule unused_call_param_sharing_pad_parameter

%unused_callee (unused_p0: s64[<=16,32]) -> s32[] {
  %unused_p0 = s64[<=16,32]{1,0} parameter(0)
  ROOT %zero = s32[] constant(0)
}

ENTRY %main (p0: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], s32[]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %d0 = s32[] parameter(1)
  %d1 = s32[] parameter(2)
  %pad = u32[16,32]{1,0} custom-call(%p0, %d0, %d1),
      custom_call_target="PadRealToStatic"
  %call_unused = s32[] call(%p0), to_apply=%unused_callee
  ROOT %out = (u32[16,32]{1,0}, s32[]) tuple(%pad, %call_unused)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> call_module,
                       ParseAndReturnUnverifiedModule(kUnusedCallParamHlo));
  std::vector<Shape> call_shapes =
      MakeCanonicalDynamicParameterShapes(*call_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *call_module, absl::MakeSpan(call_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry parameter 0"),
                             HasSubstr("call_unused"))));
  EXPECT_EQ(call_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsNonCallSignatureOrNonAsyncWrappedCustomCall) {
  constexpr absl::string_view kMismatchedSignatureHlo = R"hlo(
HloModule mismatched_signature_async_custom_call

%mismatched_pad_callee (sub_p: f32[<=16], sub_d: s32[]) -> f32[16] {
  %sub_p = f32[<=16]{0} parameter(0)
  %sub_d = s32[] parameter(1)
  ROOT %pad = f32[16]{0} custom-call(%sub_p, %sub_d),
      custom_call_target="PadRealToStatic"
}

%async_wrap_mismatch (wrap_p: f32[<=256], wrap_d: s32[]) -> f32[16] {
  %wrap_p = f32[<=256]{0} parameter(0)
  %wrap_d = s32[] parameter(1)
  ROOT %bad_cc = f32[16]{0} custom-call(%wrap_p, %wrap_d),
      custom_call_target="OtherSliceCall", to_apply=%mismatched_pad_callee
}, execution_thread="sparsecore"

ENTRY %main (p0: f32[<=256], d0: s32[]) -> f32[16] {
  %p0 = f32[<=256]{0} parameter(0)
  %d0 = s32[] parameter(1)
  %start = ((f32[<=256]{0}, s32[]), f32[16]{0}) async-start(%p0, %d0),
      async_execution_thread="sparsecore", calls=%async_wrap_mismatch
  ROOT %done = f32[16]{0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> mismatch_module,
                       ParseAndReturnUnverifiedModule(kMismatchedSignatureHlo));
  std::vector<Shape> mismatch_shapes =
      MakeCanonicalDynamicParameterShapes(*mismatch_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *mismatch_module, absl::MakeSpan(mismatch_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("entry computation parameter")));
  EXPECT_EQ(mismatch_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kNonRootCustomCallHlo = R"hlo(
HloModule non_root_async_custom_call

%sc_pad (leaf_p0: f32[<=256], leaf_d0: s32[]) -> f32[256] {
  %leaf_p0 = f32[<=256]{0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  ROOT %pad = f32[256]{0} custom-call(%leaf_p0, %leaf_d0),
      custom_call_target="PadRealToStatic"
}

%async_wrap_non_root (out_p0: f32[<=256], out_d0: s32[]) -> f32[256] {
  %out_p0 = f32[<=256]{0} parameter(0)
  %out_d0 = s32[] parameter(1)
  %non_root_cc = f32[256]{0} custom-call(%out_p0, %out_d0),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_pad
  ROOT %neg = f32[256]{0} negate(%non_root_cc)
}, execution_thread="sparsecore"

ENTRY %main (p0: f32[<=256], d0: s32[]) -> f32[256] {
  %p0 = f32[<=256]{0} parameter(0)
  %d0 = s32[] parameter(1)
  %start = ((f32[<=256]{0}, s32[]), f32[256]{0}) async-start(%p0, %d0),
      async_execution_thread="sparsecore", calls=%async_wrap_non_root
  ROOT %done = f32[256]{0} async-done(%start),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> non_root_module,
                       ParseAndReturnUnverifiedModule(kNonRootCustomCallHlo));
  std::vector<Shape> non_root_shapes =
      MakeCanonicalDynamicParameterShapes(*non_root_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *non_root_module, absl::MakeSpan(non_root_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("entry computation parameter")));
  EXPECT_EQ(non_root_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kMainThreadCustomCallHlo = R"hlo(
HloModule main_thread_async_custom_call

%sc_pad (leaf_p0: f32[<=256], leaf_d0: s32[]) -> f32[256] {
  %leaf_p0 = f32[<=256]{0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  ROOT %pad = f32[256]{0} custom-call(%leaf_p0, %leaf_d0),
      custom_call_target="PadRealToStatic"
}

%async_wrap_main (out_p0: f32[<=256], out_d0: s32[]) -> f32[256] {
  %out_p0 = f32[<=256]{0} parameter(0)
  %out_d0 = s32[] parameter(1)
  ROOT %main_cc = f32[256]{0} custom-call(%out_p0, %out_d0),
      custom_call_target="AsyncOffloadCall", to_apply=%sc_pad
}

ENTRY %main (p0: f32[<=256], d0: s32[]) -> f32[256] {
  %p0 = f32[<=256]{0} parameter(0)
  %d0 = s32[] parameter(1)
  %start = ((f32[<=256]{0}, s32[]), f32[256]{0}) async-start(%p0, %d0),
      calls=%async_wrap_main
  ROOT %done = f32[256]{0} async-done(%start)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> main_thread_module,
      ParseAndReturnUnverifiedModule(kMainThreadCustomCallHlo));
  std::vector<Shape> main_thread_shapes =
      MakeCanonicalDynamicParameterShapes(*main_thread_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *main_thread_module, absl::MakeSpan(main_thread_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("entry computation parameter")));
  EXPECT_EQ(
      main_thread_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
      1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     RejectsDeduplicatedEmbeddedOffloadWhenSecondCallerInvalid) {
  constexpr absl::string_view kSecondCallerSharedHlo = R"hlo(
HloModule deduplicated_embedded_second_caller_shared

%shared_sc_pad (leaf_p: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap_0 (p_0: s64[<=16,32], d0_0: s32[], d1_0: s32[]) -> u32[16,32] {
  %p_0 = s64[<=16,32]{1,0} parameter(0)
  %d0_0 = s32[] parameter(1)
  %d1_0 = s32[] parameter(2)
  ROOT %sc_call_0 = u32[16,32]{1,0} custom-call(%p_0, %d0_0, %d1_0),
      custom_call_target="AsyncOffloadCall", to_apply=%shared_sc_pad
}, execution_thread="sparsecore"

%async_wrap_1 (p_1: s64[<=16,32], d0_1: s32[], d1_1: s32[]) -> u32[16,32] {
  %p_1 = s64[<=16,32]{1,0} parameter(0)
  %d0_1 = s32[] parameter(1)
  %d1_1 = s32[] parameter(2)
  ROOT %sc_call_1 = u32[16,32]{1,0} custom-call(%p_1, %d0_1, %d1_1),
      custom_call_target="AsyncOffloadCall", to_apply=%shared_sc_pad
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32], s64[<=16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %d0 = s32[] parameter(2)
  %d1 = s32[] parameter(3)
  %start0 = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_0
  %done0 = u32[16,32]{1,0} async-done(%start0),
      async_execution_thread="sparsecore"
  %start1 = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p1, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_1
  %done1 = u32[16,32]{1,0} async-done(%start1),
      async_execution_thread="sparsecore"
  %sum_p1 = s64[<=16,32]{1,0} add(%p1, %p1)
  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}, s64[<=16,32]{1,0})
      tuple(%done0, %done1, %sum_p1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> shared_module,
                       ParseAndReturnUnverifiedModule(kSecondCallerSharedHlo));
  std::vector<Shape> shared_shapes =
      MakeCanonicalDynamicParameterShapes(*shared_module);
  EXPECT_THAT(
      ClearPadRealToStaticParameterMetadataPrefixes(
          *shared_module, absl::MakeSpan(shared_shapes)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry parameter 1"), HasSubstr("sum_p1"))));
  EXPECT_EQ(shared_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(shared_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);

  constexpr absl::string_view kSecondCallerCopiedHlo = R"hlo(
HloModule deduplicated_embedded_second_caller_copied

%shared_sc_pad (leaf_p: s64[<=16,32], leaf_d0: s32[], leaf_d1: s32[]) -> u32[16,32] {
  %leaf_p = s64[<=16,32]{1,0} parameter(0)
  %leaf_d0 = s32[] parameter(1)
  %leaf_d1 = s32[] parameter(2)
  ROOT %pad = u32[16,32]{1,0} custom-call(%leaf_p, %leaf_d0, %leaf_d1),
      custom_call_target="PadRealToStatic"
}

%async_wrap_0 (p_0: s64[<=16,32], d0_0: s32[], d1_0: s32[]) -> u32[16,32] {
  %p_0 = s64[<=16,32]{1,0} parameter(0)
  %d0_0 = s32[] parameter(1)
  %d1_0 = s32[] parameter(2)
  ROOT %sc_call_0 = u32[16,32]{1,0} custom-call(%p_0, %d0_0, %d1_0),
      custom_call_target="AsyncOffloadCall", to_apply=%shared_sc_pad
}, execution_thread="sparsecore"

%async_wrap_1 (p_1: s64[<=16,32], d0_1: s32[], d1_1: s32[]) -> u32[16,32] {
  %p_1 = s64[<=16,32]{1,0} parameter(0)
  %d0_1 = s32[] parameter(1)
  %d1_1 = s32[] parameter(2)
  ROOT %sc_call_1 = u32[16,32]{1,0} custom-call(%p_1, %d0_1, %d1_1),
      custom_call_target="AsyncOffloadCall", to_apply=%shared_sc_pad
}, execution_thread="sparsecore"

ENTRY %main (p0: s64[<=16,32], p1: s64[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], u32[16,32]) {
  %p0 = s64[<=16,32]{1,0} parameter(0)
  %p1 = s64[<=16,32]{1,0} parameter(1)
  %d0 = s32[] parameter(2)
  %d1 = s32[] parameter(3)
  %copy_p1 = s64[<=16,32]{1,0} copy(%p1)
  %start0 = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%p0, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_0
  %done0 = u32[16,32]{1,0} async-done(%start0),
      async_execution_thread="sparsecore"
  %start1 = ((s64[<=16,32]{1,0}, s32[], s32[]), u32[16,32]{1,0})
      async-start(%copy_p1, %d0, %d1),
      async_execution_thread="sparsecore", calls=%async_wrap_1
  %done1 = u32[16,32]{1,0} async-done(%start1),
      async_execution_thread="sparsecore"
  ROOT %out = (u32[16,32]{1,0}, u32[16,32]{1,0}) tuple(%done0, %done1)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> copied_module,
                       ParseAndReturnUnverifiedModule(kSecondCallerCopiedHlo));
  std::vector<Shape> copied_shapes =
      MakeCanonicalDynamicParameterShapes(*copied_module);
  EXPECT_THAT(ClearPadRealToStaticParameterMetadataPrefixes(
                  *copied_module, absl::MakeSpan(copied_shapes)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("entry computation parameter"),
                             HasSubstr("copy_p1"))));
  EXPECT_EQ(copied_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
  EXPECT_EQ(copied_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST(ClearPadRealToStaticParameterMetadataPrefixesTest,
     ClearsMixedTupleForwardedThroughEmbeddedOffloadDispatcher) {
  constexpr absl::string_view kMixedTupleOffloadHlo = R"hlo(
HloModule mixed_tuple_embedded_offload

%sc_mixed (sc_tup: (s64[<=16,32], f32[<=16,32], s32[], s32[])) -> (u32[16,32], f32[16,32]) {
  %sc_tup = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  %leaf_real = s64[<=16,32]{1,0} get-tuple-element(%sc_tup), index=0
  %leaf_std = f32[<=16,32]{1,0} get-tuple-element(%sc_tup), index=1
  %d0 = s32[] get-tuple-element(%sc_tup), index=2
  %d1 = s32[] get-tuple-element(%sc_tup), index=3
  %pad_real = u32[16,32]{1,0} custom-call(%leaf_real, %d0, %d1),
      custom_call_target="PadRealToStatic"
  %pad_std = (f32[16,32]{1,0}, s32[], s32[]) custom-call(%leaf_std),
      custom_call_target="PadToStatic"
  %std_data = f32[16,32]{1,0} get-tuple-element(%pad_std), index=0
  ROOT %res = (u32[16,32]{1,0}, f32[16,32]{1,0}) tuple(%pad_real, %std_data)
}

%async_wrap_mixed (wrap_tup: (s64[<=16,32], f32[<=16,32], s32[], s32[])) -> (u32[16,32], f32[16,32]) {
  %wrap_tup = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s32[], s32[]) parameter(0)
  ROOT %sc_call = (u32[16,32]{1,0}, f32[16,32]{1,0}) custom-call(%wrap_tup),
      custom_call_target="SparseCoreCaller", to_apply=%sc_mixed
}, execution_thread="sparsecore"

ENTRY %main (p0_real: s64[<=16,32], p1_std: f32[<=16,32], d0: s32[], d1: s32[]) -> (u32[16,32], f32[16,32]) {
  %p0_real = s64[<=16,32]{1,0} parameter(0)
  %p1_std = f32[<=16,32]{1,0} parameter(1)
  %d0 = s32[] parameter(2)
  %d1 = s32[] parameter(3)
  %packed = (s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s32[], s32[])
      tuple(%p0_real, %p1_std, %d0, %d1)
  %start = ((), ()) async-start(),
      async_execution_thread="sparsecore", calls=%async_wrap_mixed
  %update = (((s64[<=16,32]{1,0}, f32[<=16,32]{1,0}, s32[], s32[])),
             (u32[16,32]{1,0}, f32[16,32]{1,0}))
      async-update(%start, %packed), async_execution_thread="sparsecore"
  ROOT %done = (u32[16,32]{1,0}, f32[16,32]{1,0}) async-done(%update),
      async_execution_thread="sparsecore"
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kMixedTupleOffloadHlo));
  std::vector<Shape> param_shapes =
      MakeCanonicalDynamicParameterShapes(*module);
  ASSERT_OK(ClearPadRealToStaticParameterMetadataPrefixes(
      *module, absl::MakeSpan(param_shapes)));

  EXPECT_EQ(param_shapes[0].layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(param_shapes[1].layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

class FakeDynamicPrefixTopology : public PjRtTopologyDescription {
 public:
  explicit FakeDynamicPrefixTopology(int default_memory_space_kind_id)
      : default_memory_space_kind_id_(default_memory_space_kind_id) {}

  PjRtPlatformId platform_id() const override { return TpuId(); }
  absl::string_view platform_name() const override { return "tpu"; }
  absl::string_view platform_version() const override { return "fake"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    return {};
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return LayoutUtil::GetDefaultLayoutForShape(
        ShapeUtil::MakeShape(element_type, dims));
  }
  absl::StatusOr<Shape> MakeCanonicalShapeForMemorySpace(
      int memory_space_kind_id, Shape shape,
      const Layout* layout) const override {
    if (layout != nullptr) {
      *shape.mutable_layout() = *layout;
    } else {
      LayoutUtil::SetToDefaultLayout(&shape);
    }
    if (shape.IsArray() && shape.is_dynamic()) {
      shape.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(1024);
    }
    return shape;
  }
  absl::StatusOr<int> GetMemorySpaceKindForShape(
      const Shape& shape) const override {
    return default_memory_space_kind_id_;
  }

 private:
  int default_memory_space_kind_id_;
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class InferDispatchInfoMlirPadRealToStaticTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(client_, GetXlaPjrtCpuClient(CpuClientOptions{}));
    ASSERT_FALSE(client_->addressable_devices().empty());
    device_ = client_->addressable_devices().front();
    ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * default_ms,
                         device_->default_memory_space());
    topology_ =
        std::make_unique<FakeDynamicPrefixTopology>(default_ms->kind_id());
    device_assignment_ = std::make_shared<DeviceAssignment>(1, 1);
    (*device_assignment_)(0, 0) = device_->global_device_id().value();
  }

  absl::StatusOr<PjRtLoadedExecutableDispatchInfo> RunMlirInferDispatchInfo(
      absl::string_view mlir_source) {
    mlir::MLIRContext context;
    ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                     ParseMlirModuleString(mlir_source, context));
    CompileOptions options;
    std::vector<PjRtLoadedExecutable::LogicalDeviceIds> logical_ids = {
        PjRtLoadedExecutable::LogicalDeviceIds{0, 0}};
    std::vector<PjRtDevice*> addressable_devices = {device_};
    return InferDispatchInfo(topology_.get(), *module, options,
                             device_assignment_, std::move(logical_ids),
                             std::move(addressable_devices),
                             /*tuple_inputs=*/false);
  }

  std::unique_ptr<PjRtClient> client_;
  PjRtDevice* device_ = nullptr;
  std::unique_ptr<FakeDynamicPrefixTopology> topology_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
};

TEST_F(InferDispatchInfoMlirPadRealToStaticTest,
       ClearsPrefixForPadRealToStaticAndPreservesPadToStatic) {
  constexpr absl::string_view kMlir = R"mlir(
module @pad_real_vs_pad_std_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<?x32xf32, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg3: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg2, %arg3)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    %1:3 = stablehlo.custom_call @PadToStatic(%arg1)
        : (tensor<?x32xf32, #stablehlo.bounds<16, ?>>)
        -> (tensor<16x32xf32>, tensor<i32>, tensor<i32>)
    return %0, %1#0 : tensor<16x32xi64>, tensor<16x32xf32>
  }
}
)mlir";

  ASSERT_OK_AND_ASSIGN(PjRtLoadedExecutableDispatchInfo info,
                       RunMlirInferDispatchInfo(kMlir));
  ASSERT_EQ(info.parameter_device_shapes.size(), 4);
  EXPECT_TRUE(info.parameter_device_shapes[0].is_bounded_dynamic_dimension(0));
  EXPECT_EQ(info.parameter_device_shapes[0].dimensions(0), 16);
  EXPECT_EQ(info.parameter_device_shapes[0]
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            0);
  EXPECT_TRUE(info.parameter_device_shapes[1].is_bounded_dynamic_dimension(0));
  EXPECT_EQ(info.parameter_device_shapes[1].dimensions(0), 16);
  EXPECT_EQ(info.parameter_device_shapes[1]
                .layout()
                .dynamic_shape_metadata_prefix_bytes(),
            1024);
  ASSERT_OK_AND_ASSIGN(
      const std::vector<std::shared_ptr<const PjRtLayout>>& layouts,
      info.extras->parameter_layouts);
  EXPECT_EQ(layouts[0]->xla_layout().dynamic_shape_metadata_prefix_bytes(), 0);
  EXPECT_EQ(layouts[1]->xla_layout().dynamic_shape_metadata_prefix_bytes(),
            1024);
}

TEST_F(InferDispatchInfoMlirPadRealToStaticTest,
       RejectsSharedDynamicArgumentInMlir) {
  constexpr absl::string_view kSharedOpMlir = R"mlir(
module @shared_op_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<?x32xi64, #stablehlo.bounds<16, ?>>
              {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg1, %arg2)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    %1 = stablehlo.add %arg0, %arg0
        : tensor<?x32xi64, #stablehlo.bounds<16, ?>>
    return %0, %1 : tensor<16x32xi64>,
                    tensor<?x32xi64, #stablehlo.bounds<16, ?>>
  }
}
)mlir";

  EXPECT_THAT(
      RunMlirInferDispatchInfo(kSharedOpMlir),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry argument 0"),
                     HasSubstr("PadRealToStatic"), HasSubstr("other users"))));

  constexpr absl::string_view kReturnedArgMlir = R"mlir(
module @returned_arg_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
          tensor<?x32xi64, #stablehlo.bounds<16, ?>>
              {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg1, %arg2)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    return %0, %arg0 : tensor<16x32xi64>,
                       tensor<?x32xi64, #stablehlo.bounds<16, ?>>
  }
}
)mlir";

  EXPECT_THAT(
      RunMlirInferDispatchInfo(kReturnedArgMlir),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Entry argument 0"),
                     HasSubstr("PadRealToStatic"), HasSubstr("other users"))));
}

TEST_F(InferDispatchInfoMlirPadRealToStaticTest,
       RejectsAliasedDynamicArgumentInMlir) {
  constexpr absl::string_view kAliasedMlir = R"mlir(
module @aliased_arg_mlir {
  sdy.mesh @mesh = <[]>
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>
          {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>,
           tf.aliasing_output = 0 : i32},
      %arg1: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>},
      %arg2: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>})
      -> (tensor<16x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg1, %arg2)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    return %0 : tensor<16x32xi64>
  }
}
)mlir";

  EXPECT_THAT(RunMlirInferDispatchInfo(kAliasedMlir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Entry argument 0"),
                             HasSubstr("tf.aliasing_output"))));
}

TEST(ShouldClearPadRealToStaticMetadataPrefixTest,
     ValidatesAndClassifiesMainArguments) {
  constexpr absl::string_view kMlir = R"mlir(
module @should_clear_prefix_direct {
  func.func @main(
      %arg0: tensor<?x32xi64, #stablehlo.bounds<16, ?>>,
      %arg1: tensor<?x32xf32, #stablehlo.bounds<16, ?>>,
      %arg2: tensor<i32>,
      %arg3: tensor<i32>) -> (tensor<16x32xi64>, tensor<16x32xf32>) {
    %0 = stablehlo.custom_call @PadRealToStatic(%arg0, %arg2, %arg3)
        : (tensor<?x32xi64, #stablehlo.bounds<16, ?>>, tensor<i32>, tensor<i32>)
        -> tensor<16x32xi64>
    %1:3 = stablehlo.custom_call @PadToStatic(%arg1)
        : (tensor<?x32xf32, #stablehlo.bounds<16, ?>>)
        -> (tensor<16x32xf32>, tensor<i32>, tensor<i32>)
    return %0, %1#0 : tensor<16x32xi64>, tensor<16x32xf32>
  }
}
)mlir";

  mlir::MLIRContext context;
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       ParseMlirModuleString(kMlir, context));
  mlir::func::FuncOp main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_NE(main, nullptr);
  EXPECT_THAT(ShouldClearPadRealToStaticMetadataPrefix(main, 0),
              IsOkAndHolds(true));
  EXPECT_THAT(ShouldClearPadRealToStaticMetadataPrefix(main, 1),
              IsOkAndHolds(false));
  EXPECT_THAT(ShouldClearPadRealToStaticMetadataPrefix(main, 2),
              IsOkAndHolds(false));
  EXPECT_THAT(
      ShouldClearPadRealToStaticMetadataPrefix(main, 4),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("out of range")));
}

}  // namespace
}  // namespace xla
