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

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "xla/error_spec.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

#if GOOGLE_CUDA
static constexpr char kPlatform[] = "CUDA";
#elif TENSORFLOW_USE_ROCM
static constexpr char kPlatform[] = "ROCM";
#endif

class DynamicSliceFusionTest : public HloTestBase {};

TEST_F(DynamicSliceFusionTest, GemmSlice) {
  const char* hlo_reference = R"(
    HloModule reference

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      p2 = f16[4,8,8]{2,1,0} parameter(2)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast.41, bitcast.42),
        custom_call_target="__cublas$gemm",
        backend_config={"gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"64",
          "rhs_stride":"64",
          "grad_x":false,
          "grad_y":false
        }}

      get-tuple-element.0 = f16[8,8]{1,0} get-tuple-element(custom-call.1), index=0
      bitcast.43 = f16[1,8,8]{2,1,0} bitcast(get-tuple-element.0)
      ROOT dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, c1_s32, c0_s32, c0_s32)
    }
)";

  const char* hlo_dynamic_slice_fusion = R"(
    HloModule dynamic_slice_fusion

    dynamic-slice-fusion {
      p4 = f16[4,8,8]{2,1,0} parameter(4)
      p0.1 = f16[2,8,8]{2,1,0} parameter(0)
      p1.1 = s32[] parameter(1)
      p2.1 = s32[] parameter(2)
      slice.0 = f16[1,8,8]{2,1,0} dynamic-slice(p0.1, p1.1, p2.1, p2.1), dynamic_slice_sizes={1,8,8}
      bitcast.0 = f16[8,8]{1,0} bitcast(slice.0)
      p3 = f16[2,8,8]{2,1,0} parameter(3)
      slice.1 = f16[1,8,8]{2,1,0} dynamic-slice(p3, p1.1, p2.1, p2.1), dynamic_slice_sizes={1,8,8}
      bitcast.1 = f16[8,8]{1,0} bitcast(slice.1)
      custom-call.0 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast.0, bitcast.1),
        custom_call_target="__cublas$gemm",
        backend_config={"gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"64",
          "rhs_stride":"64",
          "grad_x":false,
          "grad_y":false
        }}
      get-tuple-element.2 = f16[8,8]{1,0} get-tuple-element(custom-call.0), index=0
      bitcast.2 = f16[1,8,8]{2,1,0} bitcast(get-tuple-element.2)
      dus.1 = f16[4,8,8]{2,1,0} dynamic-update-slice(p4, bitcast.2, p1.1, p2.1, p2.1)
      get-tuple-element.3 = s8[256]{0} get-tuple-element(custom-call.0), index=1
      ROOT tuple.1 = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(dus.1, get-tuple-element.3)
    }

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      p2 = f16[4,8,8]{2,1,0} parameter(2)
      address_computation = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion(p0, c1_s32, c0_s32, p1, p2),
        kind=kCustom, calls=dynamic-slice-fusion,
        backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],
                        "fusion_backend_config":{
                          "kind":"__custom_fusion",
                          "custom_fusion_config":{
                            "name":"dynamic_address_computation"
                           }},
                        "force_earliest_schedule":false}
      ROOT gte = f16[4,8,8]{2,1,0} get-tuple-element(address_computation), index=0
    }
)";

  auto reference = ParseAndReturnVerifiedModule(hlo_reference).value();
  auto fusion = ParseAndReturnVerifiedModule(hlo_dynamic_slice_fusion).value();

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(reference), std::move(fusion),
                                      ErrorSpec{1e-7, 1e-7},
                                      /*run_hlo_passes=*/false));
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  return stream->MemcpyD2D(
      &dst->data, src.data,
      absl::c_accumulate(src.dimensions, 1.0, std::multiplies<int64_t>()) *
          primitive_util::ByteWidth(src.dtype));
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Ret<ffi::AnyBuffer>()  // dst
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", kPlatform,
                         kMemcpy);

TEST_F(DynamicSliceFusionTest, CustomCallSlice) {
  const char* hlo_reference = R"(
    HloModule reference

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[4,8,8]{2,1,0} parameter(1)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)

      custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41),
        custom_call_target="__xla_test$$memcpy",
        api_version=API_VERSION_TYPED_FFI

      bitcast.43 = f16[1,8,8]{2,1,0} bitcast(custom-call.1)
      ROOT dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p1, bitcast.43, c1_s32, c0_s32, c0_s32)
    }
)";

  const char* hlo_dynamic_slice_fusion = R"(
    HloModule dynamic_slice_fusion

    dynamic-slice-fusion {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = f16[4,8,8]{2,1,0} parameter(3)
      slice.0 = f16[1,8,8]{2,1,0} dynamic-slice(p0, p1, p2, p2), dynamic_slice_sizes={1,8,8}
      bitcast.0 = f16[8,8]{1,0} bitcast(slice.0)

      custom-call.0 = f16[8,8]{1,0} custom-call(bitcast.0),
        custom_call_target="__xla_test$$memcpy",
        api_version=API_VERSION_TYPED_FFI

      bitcast.2 = f16[1,8,8]{2,1,0} bitcast(custom-call.0)
      ROOT dus.1 = f16[4,8,8]{2,1,0} dynamic-update-slice(p3, bitcast.2, p1, p2, p2)
    }

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[4,8,8]{2,1,0} parameter(1)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      ROOT address_computation = f16[4,8,8]{2,1,0} fusion(p0, c1_s32, c0_s32, p1),
        kind=kCustom, calls=dynamic-slice-fusion,
        backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],
                        "fusion_backend_config":{
                          "kind":"__custom_fusion",
                          "custom_fusion_config":{
                            "name":"dynamic_address_computation"
                           }},
                        "force_earliest_schedule":false}
    }
)";

  auto reference = ParseAndReturnVerifiedModule(hlo_reference).value();
  auto fusion = ParseAndReturnVerifiedModule(hlo_dynamic_slice_fusion).value();

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(reference), std::move(fusion),
                                      ErrorSpec{1e-7, 1e-7},
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
