// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/service/cpu/runtime/rng_ffi.h"

#include "absl/status/status.h"
#include "xla/runtime/aot_ffi.h"
#include "xla/runtime/aot_ffi_execution_context.h"
#include "xla/runtime/ffi/ffi_api.h"
#include "xla/runtime/ffi/ffi_c_api.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/cpu/runtime/rng.h"
#include "xla/xla_data.pb.h"

namespace xla {
struct ExecutableRunOptions;
}  // namespace xla

namespace aot = ::xla::runtime::aot;
namespace ffi = ::xla::runtime::ffi;

namespace {

using ::xla::runtime::FlatMemrefView;

// Converts an ffi::FlatBufferArg to an xla::runtime::FlatMemrefView.
FlatMemrefView ToFlatMemrefView(const ffi::FlatBufferArg& view) {
  auto dtype = static_cast<xla::PrimitiveType>(view.dtype);
  return FlatMemrefView{dtype, view.data, view.size_in_bytes};
}

ffi::FfiStatus ThreeFryFfi(xla::ExecutableRunOptions* executable_run_options,
                           ffi::FlatBufferArg state_buffer,
                           ffi::FlatBufferArg state_out_buffer,
                           ffi::FlatBufferArg values_buffer) {
  xla::cpu::XlaThreeFry three_fry;
  absl::Status status = three_fry(
      executable_run_options, ToFlatMemrefView(state_buffer),
      ToFlatMemrefView(state_out_buffer), ToFlatMemrefView(values_buffer));
  return status.ok() ? ffi::FfiStatus::Ok() : ffi::FfiStatus::Internal("err");
}

XLA_FFI_DEFINE_FUNCTION(FFI_ThreeFry, ThreeFryFfi,
                        ffi::Ffi::Binding()
                            .ApiPriv<xla::ExecutableRunOptions*>()
                            .Arg<ffi::FlatBufferArg>()    // state_buffer
                            .Arg<ffi::FlatBufferArg>()    // state_out_buffer
                            .Arg<ffi::FlatBufferArg>());  // values_buffer

ffi::FfiStatus PhiloxFfi(xla::ExecutableRunOptions* executable_run_options,
                         ffi::FlatBufferArg state_buffer,
                         ffi::FlatBufferArg state_out_buffer,
                         ffi::FlatBufferArg values_buffer) {
  xla::cpu::XlaPhilox philox;
  absl::Status status = philox(
      executable_run_options, ToFlatMemrefView(state_buffer),
      ToFlatMemrefView(state_out_buffer), ToFlatMemrefView(values_buffer));
  return status.ok() ? ffi::FfiStatus::Ok() : ffi::FfiStatus::Internal("err");
}

XLA_FFI_DEFINE_FUNCTION(FFI_Philox, PhiloxFfi,
                        ffi::Ffi::Binding()
                            .ApiPriv<xla::ExecutableRunOptions*>()
                            .Arg<ffi::FlatBufferArg>()    // state_buffer
                            .Arg<ffi::FlatBufferArg>()    // state_out_buffer
                            .Arg<ffi::FlatBufferArg>());  // values_buffer

}  // namespace

bool xla_cpu_rng_three_fry(void* execution_context, void** args, void** attrs,
                           void** rets) {
  auto ctx = static_cast<aot::ExecutionContext*>(execution_context);
  void* executable_run_options = ctx->custom_call_data;

  XLA_FFI_Api api = aot::FfiApi();
  api.priv = executable_run_options;

  XLA_FFI_Function_Args ffi_args = aot::FfiArgs(&api, args, attrs, rets);

  XLA_FFI_Error* error = FFI_ThreeFry(&ffi_args);
  return aot::ProcessErrorIfAny(error);
}

bool xla_cpu_rng_philox(void* execution_context, void** args, void** attrs,
                        void** rets) {
  auto ctx = static_cast<aot::ExecutionContext*>(execution_context);
  void* executable_run_options = ctx->custom_call_data;

  XLA_FFI_Api api = aot::FfiApi();
  api.priv = executable_run_options;

  XLA_FFI_Function_Args ffi_args = aot::FfiArgs(&api, args, attrs, rets);

  XLA_FFI_Error* error = FFI_Philox(&ffi_args);
  return aot::ProcessErrorIfAny(error);
}
