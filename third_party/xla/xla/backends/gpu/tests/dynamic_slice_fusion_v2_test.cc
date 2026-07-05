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

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/ffi/ffi.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr char kPlatform[] = "gpu";

static absl::Status MemsetScale1(se::Stream* stream, ffi::AnyBuffer input,
                                 ffi::Result<ffi::BufferR1<F32>> result,
                                 float scale0) {
  const WhileLoopState* state = IsInsideWhileLoop();
  if (state == nullptr) {
    return absl::InternalError("MemsetScale1: not inside a while loop");
  }
  float iter = static_cast<float>(state->loop_iteration);
  se::DeviceAddressBase dst = result->device_memory();
  return stream->Memset32(&dst, absl::bit_cast<uint32_t>(iter * scale0),
                          dst.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetScale1, MemsetScale1,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::BufferR1<F32>>()
                           .Attr<float>("scale0"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memset_scale1",
                         kPlatform, kMemsetScale1);

static absl::Status MemsetScale12(se::Stream* stream, ffi::AnyBuffer input,
                                  ffi::Result<ffi::BufferR1<F32>> out0,
                                  ffi::Result<ffi::BufferR1<F32>> out1,
                                  float scale0, float scale1) {
  const WhileLoopState* state = IsInsideWhileLoop();
  if (state == nullptr) {
    return absl::InternalError("MemsetScale12: not inside a while loop");
  }
  float iter = static_cast<float>(state->loop_iteration);
  se::DeviceAddressBase dst0 = out0->device_memory();
  auto status = stream->Memset32(&dst0, absl::bit_cast<uint32_t>(iter * scale0),
                                 dst0.size());
  if (!status.ok()) {
    return status;
  }
  se::DeviceAddressBase dst1 = out1->device_memory();
  return stream->Memset32(&dst1, absl::bit_cast<uint32_t>(iter * scale1),
                          dst1.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetScale12, MemsetScale12,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::BufferR1<F32>>()
                           .Ret<ffi::BufferR1<F32>>()
                           .Attr<float>("scale0")
                           .Attr<float>("scale1"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memset_scale12",
                         kPlatform, kMemsetScale12);

static absl::Status MemsetConst(se::Stream* stream, ffi::AnyBuffer input,
                                ffi::Result<ffi::BufferR1<F32>> result,
                                float value) {
  se::DeviceAddressBase dst = result->device_memory();
  return stream->Memset32(&dst, absl::bit_cast<uint32_t>(value), dst.size());
}

XLA_FFI_DEFINE_HANDLER(kMemsetConst, MemsetConst,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::BufferR1<F32>>()
                           .Attr<float>("value"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memset_const",
                         kPlatform, kMemsetConst);

class DynamicSliceFusionV2Test : public HloPjRtGpuTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_dynamic_slice_fusion_verify_offsets(
        true);
    return debug_options;
  }
};

TEST_F(DynamicSliceFusionV2Test, SingleOutputOneDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
      %fill_2d = f32[1,4] bitcast(%fill)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p0, %fill_2d, %p1, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    body {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      zero = s32[] constant(0)
      updated = f32[4,4] fusion(buf, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4]) tuple(next_i, updated)
    }

    cond {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4]) tuple(zero, init_buf)
      while = (s32[], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=1
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {2.0f, 2.0f, 2.0f, 2.0f},
                                                   {3.0f, 3.0f, 3.0f, 3.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, SingleOutputOneDUSWithOffsetExpression) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
      %fill_2d = f32[1,4] bitcast(%fill)
      %one = s32[] constant(1)
      %offset = s32[] add(%p1, %one)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p0, %fill_2d, %offset, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":16,"byte_stride":16}}
    }

    body {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      zero = s32[] constant(0)
      updated = f32[4,4] fusion(buf, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4]) tuple(next_i, updated)
    }

    cond {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(3)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(-1)), dimensions={}
      init = (s32[], f32[4,4]) tuple(zero, init_buf)
      while = (s32[], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=1
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{-1.0f, -1.0f, -1.0f, -1.0f},
                                                   {0.0f, 0.0f, 0.0f, 0.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {2.0f, 2.0f, 2.0f, 2.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, CublasLtMatmulWithBitcastSlicedOperand) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[1,4] parameter(1)
      %p2 = s32[] parameter(2)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p0, %p2, %zero),
        dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %lhs = f32[4,1] bitcast(%ds)
      ROOT %gemm = f32[4,4] custom-call(%lhs, %p1),
        custom_call_target="__cublas$lt$matmul",
        backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,
          "beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT",
          "DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","grad_x":false,
          "grad_y":false,"damax_output":false}}
    }

    body {
      param = (s32[], f32[4,4], f32[1,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      input = f32[4,4] get-tuple-element(param), index=1
      rhs = f32[1,4] get-tuple-element(param), index=2
      output = f32[4,4] get-tuple-element(param), index=3
      updated = f32[4,4] fusion(input, rhs, i),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[1,4], f32[4,4])
        tuple(next_i, input, rhs, updated)
    }

    cond {
      param = (s32[], f32[4,4], f32[1,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      input = f32[4,4] broadcast(f32[] constant(1)), dimensions={}
      rhs = f32[1,4] broadcast(f32[] constant(1)), dimensions={}
      init_output = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[1,4], f32[4,4])
        tuple(zero, input, rhs, init_output)
      while = (s32[], f32[4,4], f32[1,4], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=3
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{1.0f, 1.0f, 1.0f, 1.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, TupleOutputTwoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      %call = (f32[4], f32[4]) custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32, scale1 = 2.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %gte1 = f32[4] get-tuple-element(%call), index=1
      %bc0 = f32[1,4] bitcast(%gte0)
      %bc1 = f32[1,4] bitcast(%gte1)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p0, %bc0, %p2, %p3),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,4]
        dynamic-update-slice(%p1, %bc1, %p2, %p3),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %tuple = (f32[4,4], f32[4,4]) tuple(%dus0, %dus1)
    }

    body {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf0 = f32[4,4] get-tuple-element(param), index=1
      buf1 = f32[4,4] get-tuple-element(param), index=2
      zero = s32[] constant(0)
      fused = (f32[4,4], f32[4,4]) fusion(buf0, buf1, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      updated0 = f32[4,4] get-tuple-element(fused), index=0
      updated1 = f32[4,4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4])
        tuple(next_i, updated0, updated1)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init0 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init1 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4]) tuple(zero, init0, init1)
      while = (s32[], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=1
      r1 = f32[4,4] get-tuple-element(while), index=2
      ROOT result = (f32[4,4], f32[4,4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {1.0f, 1.0f, 1.0f, 1.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f}});
  Literal expected1 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {4.0f, 4.0f, 4.0f, 4.0f},
                                                    {6.0f, 6.0f, 6.0f, 6.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, SingleOutputNoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      ROOT %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
    }

    body {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      fill = f32[4] fusion(buf),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      fill_2d = f32[1,4] bitcast(fill)
      zero = s32[] constant(0)
      updated = f32[4,4] dynamic-update-slice(buf, fill_2d, i, zero)
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4]) tuple(next_i, updated)
    }

    cond {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4]) tuple(zero, init_buf)
      while = (s32[], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=1
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {2.0f, 2.0f, 2.0f, 2.0f},
                                                   {3.0f, 3.0f, 3.0f, 3.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, AsyncSingleOutputOneDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
      %fill_2d = f32[1,4] bitcast(%fill)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p0, %fill_2d, %p1, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    %async_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      ROOT %fusion = f32[4,4] fusion(%p0, %p1, %p2),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
    }

    body {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      zero = s32[] constant(0)
      start = ((f32[4,4], s32[], s32[]), f32[4,4], u32[])
        async-start(buf, i, zero), calls=%async_computation
      updated = f32[4,4] async-done(start)
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4]) tuple(next_i, updated)
    }

    cond {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4]) tuple(zero, init_buf)
      while = (s32[], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=1
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {2.0f, 2.0f, 2.0f, 2.0f},
                                                   {3.0f, 3.0f, 3.0f, 3.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, AsyncTupleOutputTwoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      %call = (f32[4], f32[4]) custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32, scale1 = 2.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %gte1 = f32[4] get-tuple-element(%call), index=1
      %bc0 = f32[1,4] bitcast(%gte0)
      %bc1 = f32[1,4] bitcast(%gte1)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p0, %bc0, %p2, %p3),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,4]
        dynamic-update-slice(%p1, %bc1, %p2, %p3),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %tuple = (f32[4,4], f32[4,4]) tuple(%dus0, %dus1)
    }

    %async_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      ROOT %fusion = (f32[4,4], f32[4,4]) fusion(%p0, %p1, %p2, %p3),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
    }

    body {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf0 = f32[4,4] get-tuple-element(param), index=1
      buf1 = f32[4,4] get-tuple-element(param), index=2
      zero = s32[] constant(0)
      start = ((f32[4,4], f32[4,4], s32[], s32[]), (f32[4,4], f32[4,4]), u32[])
        async-start(buf0, buf1, i, zero), calls=%async_computation
      fused = (f32[4,4], f32[4,4]) async-done(start)
      updated0 = f32[4,4] get-tuple-element(fused), index=0
      updated1 = f32[4,4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4])
        tuple(next_i, updated0, updated1)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init0 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init1 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4]) tuple(zero, init0, init1)
      while = (s32[], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=1
      r1 = f32[4,4] get-tuple-element(while), index=2
      ROOT result = (f32[4,4], f32[4,4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {1.0f, 1.0f, 1.0f, 1.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f}});
  Literal expected1 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {4.0f, 4.0f, 4.0f, 4.0f},
                                                    {6.0f, 6.0f, 6.0f, 6.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, TupleOutputOneDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %call = (f32[4], f32[4]) custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32, scale1 = 2.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %bc0 = f32[1,4] bitcast(%gte0)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p0, %bc0, %p1, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %gte1 = f32[4] get-tuple-element(%call), index=1
      ROOT %tuple = (f32[4,4], f32[4]) tuple(%dus0, %gte1)
    }

    body {
      param = (s32[], f32[4,4], f32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      prev = f32[4] get-tuple-element(param), index=2
      zero = s32[] constant(0)
      fused = (f32[4,4], f32[4]) fusion(buf, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      updated = f32[4,4] get-tuple-element(fused), index=0
      cur = f32[4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4]) tuple(next_i, updated, cur)
    }

    cond {
      param = (s32[], f32[4,4], f32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init_prev = f32[4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4]) tuple(zero, init_buf, init_prev)
      while = (s32[], f32[4,4], f32[4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=1
      r1 = f32[4] get-tuple-element(while), index=2
      ROOT result = (f32[4,4], f32[4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {1.0f, 1.0f, 1.0f, 1.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f}});
  Literal expected1 = LiteralUtil::CreateR1<float>({6.0f, 6.0f, 6.0f, 6.0f});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, TupleOutputNoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4] parameter(0)
      %call = (f32[4], f32[4]) custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32, scale1 = 2.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %gte1 = f32[4] get-tuple-element(%call), index=1
      ROOT %tuple = (f32[4], f32[4]) tuple(%gte0, %gte1)
    }

    body {
      param = (s32[], f32[4], f32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      prev0 = f32[4] get-tuple-element(param), index=1
      prev1 = f32[4] get-tuple-element(param), index=2
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      fused = (f32[4], f32[4]) fusion(prev0),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      cur0 = f32[4] get-tuple-element(fused), index=0
      cur1 = f32[4] get-tuple-element(fused), index=1
      ROOT tuple = (s32[], f32[4], f32[4]) tuple(next_i, cur0, cur1)
    }

    cond {
      param = (s32[], f32[4], f32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init0 = f32[4] broadcast(f32[] constant(0)), dimensions={}
      init1 = f32[4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4], f32[4]) tuple(zero, init0, init1)
      while = (s32[], f32[4], f32[4])
        while(init), condition=cond, body=body
      r0 = f32[4] get-tuple-element(while), index=1
      r1 = f32[4] get-tuple-element(while), index=2
      ROOT result = (f32[4], f32[4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR1<float>({3.0f, 3.0f, 3.0f, 3.0f});
  Literal expected1 = LiteralUtil::CreateR1<float>({6.0f, 6.0f, 6.0f, 6.0f});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, AsyncTupleOutputOneDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %call = (f32[4], f32[4]) custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32, scale1 = 2.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %bc0 = f32[1,4] bitcast(%gte0)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p0, %bc0, %p1, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %gte1 = f32[4] get-tuple-element(%call), index=1
      ROOT %tuple = (f32[4,4], f32[4]) tuple(%dus0, %gte1)
    }

    %async_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      ROOT %fusion = (f32[4,4], f32[4]) fusion(%p0, %p1, %p2),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
    }

    body {
      param = (s32[], f32[4,4], f32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      prev = f32[4] get-tuple-element(param), index=2
      zero = s32[] constant(0)
      start = ((f32[4,4], s32[], s32[]), (f32[4,4], f32[4]), u32[])
        async-start(buf, i, zero), calls=%async_computation
      fused = (f32[4,4], f32[4]) async-done(start)
      updated = f32[4,4] get-tuple-element(fused), index=0
      cur = f32[4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4]) tuple(next_i, updated, cur)
    }

    cond {
      param = (s32[], f32[4,4], f32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init_prev = f32[4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4]) tuple(zero, init_buf, init_prev)
      while = (s32[], f32[4,4], f32[4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=1
      r1 = f32[4] get-tuple-element(while), index=2
      ROOT result = (f32[4,4], f32[4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {1.0f, 1.0f, 1.0f, 1.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f}});
  Literal expected1 = LiteralUtil::CreateR1<float>({6.0f, 6.0f, 6.0f, 6.0f});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, OffsetCheckWrongStride) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
      %fill_2d = f32[1,4] bitcast(%fill)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p0, %fill_2d, %p1, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":32}}
    }

    body {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      zero = s32[] constant(0)
      updated = f32[4,4] fusion(buf, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4]) tuple(next_i, updated)
    }

    cond {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4]) tuple(zero, init_buf)
      while = (s32[], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=1
    }
  )";

  auto result = Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                        /*run_hlo_passes=*/false);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("offset mismatch"));
}

TEST_F(DynamicSliceFusionV2Test, NestedTupleOutputTwoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = s32[] parameter(2)
      %p3 = s32[] parameter(3)
      %call = (f32[4], f32[4]) custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32, scale1 = 2.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %gte1 = f32[4] get-tuple-element(%call), index=1
      %bc0 = f32[1,4] bitcast(%gte0)
      %bc1 = f32[1,4] bitcast(%gte1)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p0, %bc0, %p2, %p3),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,4]
        dynamic-update-slice(%p1, %bc1, %p2, %p3),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %outer = ((f32[4,4]), f32[4,4]) tuple(
        (f32[4,4]) tuple(%dus0), %dus1)
    }

    body {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf0 = f32[4,4] get-tuple-element(param), index=1
      buf1 = f32[4,4] get-tuple-element(param), index=2
      zero = s32[] constant(0)
      fused = ((f32[4,4]), f32[4,4]) fusion(buf0, buf1, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      inner = (f32[4,4]) get-tuple-element(fused), index=0
      updated0 = f32[4,4] get-tuple-element(inner), index=0
      updated1 = f32[4,4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4])
        tuple(next_i, updated0, updated1)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init0 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init1 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4]) tuple(zero, init0, init1)
      while = (s32[], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=1
      r1 = f32[4,4] get-tuple-element(while), index=2
      ROOT result = (f32[4,4], f32[4,4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {1.0f, 1.0f, 1.0f, 1.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f}});
  Literal expected1 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                                    {4.0f, 4.0f, 4.0f, 4.0f},
                                                    {6.0f, 6.0f, 6.0f, 6.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, SingleOutputOneDSOneDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = f32[4,4] parameter(1)
      %p2 = s32[] parameter(2)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p0, %p2, %zero),
        dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %ds_flat = f32[4] bitcast(%ds)
      %hero = f32[4] custom-call(%ds_flat),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 3.0 : f32}"
      %hero_2d = f32[1,4] bitcast(%hero)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p1, %hero_2d, %p2, %zero),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    body {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      input = f32[4,4] get-tuple-element(param), index=1
      output = f32[4,4] get-tuple-element(param), index=2
      updated = f32[4,4] fusion(input, output, i),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4]) tuple(next_i, input, updated)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      input = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      output = f32[4,4] broadcast(f32[] constant(-1)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4]) tuple(zero, input, output)
      while = (s32[], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=2
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                   {3.0f, 3.0f, 3.0f, 3.0f},
                                                   {6.0f, 6.0f, 6.0f, 6.0f},
                                                   {9.0f, 9.0f, 9.0f, 9.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, CombinedDSInputTupleOutputTwoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p_input = f32[4,4] parameter(0)
      %p_buf0 = f32[4,4] parameter(1)
      %p_buf1 = f32[4,4] parameter(2)
      %p_i = s32[] parameter(3)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p_input, %p_i, %zero),
        dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %ds_flat = f32[4] bitcast(%ds)
      %call = (f32[4], f32[4]) custom-call(%ds_flat),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 3.0 : f32, scale1 = 5.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %gte1 = f32[4] get-tuple-element(%call), index=1
      %bc0 = f32[1,4] bitcast(%gte0)
      %bc1 = f32[1,4] bitcast(%gte1)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p_buf0, %bc0, %p_i, %zero),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,4]
        dynamic-update-slice(%p_buf1, %bc1, %p_i, %zero),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %tuple = (f32[4,4], f32[4,4]) tuple(%dus0, %dus1)
    }

    body {
      param = (s32[], f32[4,4], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      input = f32[4,4] get-tuple-element(param), index=1
      buf0 = f32[4,4] get-tuple-element(param), index=2
      buf1 = f32[4,4] get-tuple-element(param), index=3
      fused = (f32[4,4], f32[4,4]) fusion(input, buf0, buf1, i),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      updated0 = f32[4,4] get-tuple-element(fused), index=0
      updated1 = f32[4,4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4], f32[4,4])
        tuple(next_i, input, updated0, updated1)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      input = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      buf0 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      buf1 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4], f32[4,4])
        tuple(zero, input, buf0, buf1)
      while = (s32[], f32[4,4], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=2
      r1 = f32[4,4] get-tuple-element(while), index=3
      ROOT result = (f32[4,4], f32[4,4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f},
                                                    {6.0f, 6.0f, 6.0f, 6.0f},
                                                    {9.0f, 9.0f, 9.0f, 9.0f}});
  Literal expected1 =
      LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                    {5.0f, 5.0f, 5.0f, 5.0f},
                                    {10.0f, 10.0f, 10.0f, 10.0f},
                                    {15.0f, 15.0f, 15.0f, 15.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, SingleOutputOneDSNoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p0, %p1, %zero),
        dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %ds_flat = f32[4] bitcast(%ds)
      ROOT %hero = f32[4] custom-call(%ds_flat),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
    }

    body {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      input = f32[4,4] get-tuple-element(param), index=1
      output = f32[4,4] get-tuple-element(param), index=2
      zero = s32[] constant(0)
      hero_out = f32[4] fusion(input, i),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      hero_2d = f32[1,4] bitcast(hero_out)
      updated = f32[4,4] dynamic-update-slice(output, hero_2d, i, zero)
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4]) tuple(next_i, input, updated)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      input = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      output = f32[4,4] broadcast(f32[] constant(-1)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4]) tuple(zero, input, output)
      while = (s32[], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=2
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                   {1.0f, 1.0f, 1.0f, 1.0f},
                                                   {2.0f, 2.0f, 2.0f, 2.0f},
                                                   {3.0f, 3.0f, 3.0f, 3.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, NestedTupleDSInputTwoDUS) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p_input = f32[4,4] parameter(0)
      %p_buf0 = f32[4,4] parameter(1)
      %p_buf1 = f32[4,4] parameter(2)
      %p_i = s32[] parameter(3)
      %zero = s32[] constant(0)
      %ds = f32[1,4] dynamic-slice(%p_input, %p_i, %zero),
        dynamic_slice_sizes={1,4},
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %ds_flat = f32[4] bitcast(%ds)
      %call = (f32[4], f32[4]) custom-call(%ds_flat),
        custom_call_target="__xla_test$$memset_scale12",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 3.0 : f32, scale1 = 5.0 : f32}"
      %gte0 = f32[4] get-tuple-element(%call), index=0
      %gte1 = f32[4] get-tuple-element(%call), index=1
      %bc0 = f32[1,4] bitcast(%gte0)
      %bc1 = f32[1,4] bitcast(%gte1)
      %dus0 = f32[4,4]
        dynamic-update-slice(%p_buf0, %bc0, %p_i, %zero),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      %dus1 = f32[4,4]
        dynamic-update-slice(%p_buf1, %bc1, %p_i, %zero),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
      ROOT %outer = ((f32[4,4]), f32[4,4]) tuple(
        (f32[4,4]) tuple(%dus0), %dus1)
    }

    body {
      param = (s32[], f32[4,4], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      input = f32[4,4] get-tuple-element(param), index=1
      buf0 = f32[4,4] get-tuple-element(param), index=2
      buf1 = f32[4,4] get-tuple-element(param), index=3
      fused = ((f32[4,4]), f32[4,4]) fusion(input, buf0, buf1, i),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      inner = (f32[4,4]) get-tuple-element(fused), index=0
      updated0 = f32[4,4] get-tuple-element(inner), index=0
      updated1 = f32[4,4] get-tuple-element(fused), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4], f32[4,4], f32[4,4])
        tuple(next_i, input, updated0, updated1)
    }

    cond {
      param = (s32[], f32[4,4], f32[4,4], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      input = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      buf0 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      buf1 = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4], f32[4,4], f32[4,4])
        tuple(zero, input, buf0, buf1)
      while = (s32[], f32[4,4], f32[4,4], f32[4,4])
        while(init), condition=cond, body=body
      r0 = f32[4,4] get-tuple-element(while), index=2
      r1 = f32[4,4] get-tuple-element(while), index=3
      ROOT result = (f32[4,4], f32[4,4]) tuple(r0, r1)
    }
  )";

  Literal expected0 = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                    {3.0f, 3.0f, 3.0f, 3.0f},
                                                    {6.0f, 6.0f, 6.0f, 6.0f},
                                                    {9.0f, 9.0f, 9.0f, 9.0f}});
  Literal expected1 =
      LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                    {5.0f, 5.0f, 5.0f, 5.0f},
                                    {10.0f, 10.0f, 10.0f, 10.0f},
                                    {15.0f, 15.0f, 15.0f, 15.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));

  EXPECT_TRUE(LiteralTestUtil::Equal(expected0, LiteralSlice(result, {0})));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected1, LiteralSlice(result, {1})));
}

TEST_F(DynamicSliceFusionV2Test, ConstantOffsetDUSNoLoop) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_const",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{value = 42.0 : f32}"
      %fill_2d = f32[1,4] bitcast(%fill)
      %two = s32[] constant(2)
      %zero = s32[] constant(0)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p0, %fill_2d, %two, %zero),
        backend_config={"dynamic_slice_config":
          {"byte_offset":32,"byte_stride":0}}
    }

    ENTRY main {
      buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      ROOT result = f32[4,4] fusion(buf),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
    }
  )";

  Literal expected = LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                                   {0.0f, 0.0f, 0.0f, 0.0f},
                                                   {42.0f, 42.0f, 42.0f, 42.0f},
                                                   {0.0f, 0.0f, 0.0f, 0.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(DynamicSliceFusionV2Test, OobClampingAccumulatesInLastSlice) {
  // 16 iterations into a buffer with leading dimension 4. Iterations 4-15
  // clamp to the last row, so the final value there is iter=15.
  const char* hlo = R"(
    HloModule test, is_scheduled=true

    %dsf_computation {
      %p0 = f32[4,4] parameter(0)
      %p1 = s32[] parameter(1)
      %p2 = s32[] parameter(2)
      %fill = f32[4] custom-call(%p0),
        custom_call_target="__xla_test$$memset_scale1",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{scale0 = 1.0 : f32}"
      %fill_2d = f32[1,4] bitcast(%fill)
      ROOT %dus = f32[4,4]
        dynamic-update-slice(%p0, %fill_2d, %p1, %p2),
        backend_config={"dynamic_slice_config":
          {"loop_index":0,"byte_offset":0,"byte_stride":16}}
    }

    body {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      buf = f32[4,4] get-tuple-element(param), index=1
      zero = s32[] constant(0)
      updated = f32[4,4] fusion(buf, i, zero),
        kind=kCustom, calls=%dsf_computation,
        backend_config={"fusion_backend_config":{
          "kind":"__custom_fusion",
          "custom_fusion_config":
            {"name":"dynamic_slice_fusion"}}}
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      ROOT tuple = (s32[], f32[4,4]) tuple(next_i, updated)
    }

    cond {
      param = (s32[], f32[4,4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(16)
      ROOT cmp = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      init_buf = f32[4,4] broadcast(f32[] constant(0)), dimensions={}
      init = (s32[], f32[4,4]) tuple(zero, init_buf)
      while = (s32[], f32[4,4])
        while(init), condition=cond, body=body
      ROOT result = f32[4,4] get-tuple-element(while), index=1
    }
  )";

  // Rows 0-2 written once at iterations 0,1,2. Row 3 last written at iter 15.
  Literal expected =
      LiteralUtil::CreateR2<float>({{0.0f, 0.0f, 0.0f, 0.0f},
                                    {1.0f, 1.0f, 1.0f, 1.0f},
                                    {2.0f, 2.0f, 2.0f, 2.0f},
                                    {15.0f, 15.0f, 15.0f, 15.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal result, Execute(std::move(*ParseAndReturnVerifiedModule(hlo)), {},
                              /*run_hlo_passes=*/false));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace xla::gpu
