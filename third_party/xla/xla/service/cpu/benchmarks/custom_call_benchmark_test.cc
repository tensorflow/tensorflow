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

#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
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
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_CustomCall_Minimal)->MeasureProcessCPUTime();

}  // namespace
}  // namespace xla::cpu
