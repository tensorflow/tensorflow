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

#include "xla/service/cpu/runtime/convolution_ffi.h"

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/runtime/aot_ffi.h"
#include "xla/runtime/aot_ffi_execution_context.h"
#include "xla/runtime/ffi/ffi_api.h"
#include "xla/runtime/ffi/ffi_c_api.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/cpu/runtime/convolution.h"
#include "xla/xla_data.pb.h"

namespace xla {
struct ExecutableRunOptions;
}  // namespace xla

namespace aot = ::xla::runtime::aot;
namespace ffi = ::xla::runtime::ffi;

namespace {

using ::xla::runtime::MemrefView;

ffi::FfiStatus ConvolutionFfi(
    xla::ExecutableRunOptions* executable_run_options, ffi::BufferArg input,
    ffi::BufferArg kernel, ffi::BufferArg output, int64_t inputBatchDimension,
    ffi::Span<const int64_t> inputSpatialDimensions,
    int64_t inputFeatureDimension,
    ffi::Span<const int64_t> kernelSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ffi::Span<const int64_t> outputSpatialDimensions,
    ffi::Span<const int64_t> window_strides, ffi::Span<const int64_t> padding,
    ffi::Span<const int64_t> lhs_dilation,
    ffi::Span<const int64_t> rhs_dilation, int64_t feature_group_count) {
  auto to_memref_view = [](const ffi::BufferArg& view) -> MemrefView {
    auto dtype = static_cast<xla::PrimitiveType>(view.dtype);
    return MemrefView{
        dtype, view.data,
        absl::MakeConstSpan(view.sizes.begin(), view.sizes.end())};
  };
  auto to_span =
      [](ffi::Span<const int64_t> span) -> absl::Span<const int64_t> {
    return absl::MakeConstSpan(span.begin(), span.end());
  };

  xla::cpu::XlaConvolution convolution;
  absl::Status status = convolution(
      executable_run_options, to_memref_view(input), to_memref_view(kernel),
      to_memref_view(output), inputBatchDimension,
      to_span(inputSpatialDimensions), inputFeatureDimension,
      to_span(kernelSpatialDimensions), kernelInputFeatureDimension,
      kernelOutputFeatureDimension, to_span(outputSpatialDimensions),
      to_span(window_strides), to_span(padding), to_span(lhs_dilation),
      to_span(rhs_dilation), feature_group_count);
  return status.ok() ? ffi::FfiStatus::Ok() : ffi::FfiStatus::Internal("err");
}

XLA_FFI_DEFINE_FUNCTION(
    FFI_Convolution, ConvolutionFfi,
    ffi::Ffi::Binding()
        .ApiPriv<xla::ExecutableRunOptions*>()
        .Arg<ffi::BufferArg>()  // input
        .Arg<ffi::BufferArg>()  // kernel
        .Arg<ffi::BufferArg>()  // output
        .Attr<int64_t>("inputBatchDimension")
        .Attr<ffi::Span<const int64_t>>("inputSpatialDimensions")
        .Attr<int64_t>("inputFeatureDimension")
        .Attr<ffi::Span<const int64_t>>("kernelSpatialDimensions")
        .Attr<int64_t>("kernelInputFeatureDimension")
        .Attr<int64_t>("kernelOutputFeatureDimension")
        .Attr<ffi::Span<const int64_t>>("outputSpatialDimensions")
        .Attr<ffi::Span<const int64_t>>("window_strides")
        .Attr<ffi::Span<const int64_t>>("padding")
        .Attr<ffi::Span<const int64_t>>("lhs_dilation")
        .Attr<ffi::Span<const int64_t>>("rhs_dilation")
        .Attr<int64_t>("feature_group_count"));

}  // namespace

bool xla_cpu_convolution(void* execution_context, void** args, void** attrs,
                         void** rets) {
  auto ctx = static_cast<aot::ExecutionContext*>(execution_context);
  void* executable_run_options = ctx->custom_call_data;

  XLA_FFI_Api api = aot::FfiApi();
  api.priv = executable_run_options;

  XLA_FFI_Function_Args ffi_args = aot::FfiArgs(&api, args, attrs, rets);

  XLA_FFI_Error* error = FFI_Convolution(&ffi_args);
  return aot::ProcessErrorIfAny(error);
}
