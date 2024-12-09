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
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {
namespace {

static absl::Status Minimal(
    ffi::Result<ffi::BufferR0<PrimitiveType::F32>> unused) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kMinimal, Minimal,
    ffi::Ffi::Bind()
        .Ret<ffi::BufferR0<PrimitiveType::F32>>());  // Unused out buffer

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_bm$$minimal", "Host",
                         kMinimal);

static void BM_CustomCall_Minimal(benchmark::State& state) {
  const char* kModuleStr = R"(
    HloModule module

    ENTRY custom_call {
      ROOT custom-call = f32[] custom-call(),
        custom_call_target="__xla_bm$$minimal",
        api_version=API_VERSION_TYPED_FFI
    }
  )";
  CHECK_OK(RunHloBenchmark(state, kModuleStr, /*args=*/{},
                           /*replacements=*/{}));
}

static absl::Status ManyIntAttributes(
    ffi::Result<ffi::AnyBuffer> unused, int32_t attr0, int32_t attr1,
    int32_t attr2, int32_t attr3, int32_t attr4, int32_t attr5, int32_t attr6,
    int32_t attr7, int32_t attr8, int32_t attr9, int32_t attr10, int32_t attr11,
    int32_t attr12, int32_t attr13, int32_t attr14, int32_t attr15) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kManyIntAttributes, ManyIntAttributes,
                       ffi::Ffi::Bind()
                           .Ret<ffi::AnyBuffer>()  // Unused out buffer
                           .Attr<int32_t>("attr0")
                           .Attr<int32_t>("attr1")
                           .Attr<int32_t>("attr2")
                           .Attr<int32_t>("attr3")
                           .Attr<int32_t>("attr4")
                           .Attr<int32_t>("attr5")
                           .Attr<int32_t>("attr6")
                           .Attr<int32_t>("attr7")
                           .Attr<int32_t>("attr8")
                           .Attr<int32_t>("attr9")
                           .Attr<int32_t>("attr10")
                           .Attr<int32_t>("attr11")
                           .Attr<int32_t>("attr12")
                           .Attr<int32_t>("attr13")
                           .Attr<int32_t>("attr14")
                           .Attr<int32_t>("attr15"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_bm$$many_int_attributes",
                         "Host", kManyIntAttributes);

static void BM_CustomCall_16IntAttributes(benchmark::State& state) {
  absl::string_view hlo = R"(
    HloModule module

    ENTRY custom_call {
      ROOT custom-call = f32[] custom-call(),
        custom_call_target="__xla_bm$$many_int_attributes",
        api_version=API_VERSION_TYPED_FFI, backend_config="$config"
    }
  )";
  std::stringstream config;
  config << "{";
  for (int i = 0; i < 16; ++i) {
    config << "attr" << i << " = 5 : i32" << (i < 15 ? ", " : "");
  }
  config << "}";
  CHECK_OK(RunHloBenchmark(state, hlo, /*args=*/{},
                           /*replacements=*/{{"$config", config.str()}}));
}

static absl::Status ManyFloatBuffers(
    ffi::Buffer<PrimitiveType::F32> arg0, ffi::Buffer<PrimitiveType::F32> arg1,
    ffi::Buffer<PrimitiveType::F32> arg2, ffi::Buffer<PrimitiveType::F32> arg3,
    ffi::Buffer<PrimitiveType::F32> arg4, ffi::Buffer<PrimitiveType::F32> arg5,
    ffi::Buffer<PrimitiveType::F32> arg6, ffi::Buffer<PrimitiveType::F32> arg7,
    ffi::Buffer<PrimitiveType::F32> arg8, ffi::Buffer<PrimitiveType::F32> arg9,
    ffi::Result<ffi::Buffer<PrimitiveType::F32>> ret0,
    ffi::Result<ffi::Buffer<PrimitiveType::F32>> ret1,
    ffi::Result<ffi::Buffer<PrimitiveType::F32>> ret2,
    ffi::Result<ffi::Buffer<PrimitiveType::F32>> ret3,
    ffi::Result<ffi::Buffer<PrimitiveType::F32>> ret4,
    ffi::Result<ffi::Buffer<PrimitiveType::F32>> ret5) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kManyFloatBuffers, ManyFloatBuffers,
                       ffi::Ffi::Bind()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg0
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg1
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg2
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg3
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg4
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg5
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg6
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg7
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg8
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()    // arg9
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()    // ret0
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()    // ret1
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()    // ret2
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()    // ret3
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()    // ret4
                           .Ret<ffi::Buffer<PrimitiveType::F32>>());  // ret5

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_bm$$many_float_buffers",
                         "Host", kManyFloatBuffers);

static void BM_CustomCall_16FloatBuffers(benchmark::State& state) {
  int64_t d = 128;

  absl::string_view hlo = R"(
    HloModule module

    ENTRY custom_call {
      p0 = f32[$d,$d] parameter(0)
      p1 = f32[$d,$d] parameter(1)
      p2 = f32[$d,$d] parameter(2)
      p3 = f32[$d,$d] parameter(3)
      p4 = f32[$d,$d] parameter(4)
      p5 = f32[$d,$d] parameter(5)
      p6 = f32[$d,$d] parameter(6)
      p7 = f32[$d,$d] parameter(7)
      p8 = f32[$d,$d] parameter(8)
      p9 = f32[$d,$d] parameter(9)
      ROOT custom-call = (f32[$d,$d], f32[$d,$d], f32[$d,$d],
        f32[$d,$d], f32[$d,$d], f32[$d,$d])
        custom-call(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9),
        custom_call_target="__xla_bm$$many_float_buffers",
        api_version=API_VERSION_TYPED_FFI
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {d, d});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  std::vector<const Literal*> args(10, &p0);

  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d", absl::StrCat(d)}}));
}

BENCHMARK(BM_CustomCall_Minimal)->MeasureProcessCPUTime();
BENCHMARK(BM_CustomCall_16IntAttributes)->MeasureProcessCPUTime();
BENCHMARK(BM_CustomCall_16FloatBuffers)->MeasureProcessCPUTime();

}  // namespace
}  // namespace xla::cpu
