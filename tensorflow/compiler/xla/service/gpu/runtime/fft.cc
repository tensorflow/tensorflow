/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/fft.h"

#include <memory>

#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/state.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/stream_executor/fft.h"

namespace xla {

using xla::runtime::CustomCall;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

//===----------------------------------------------------------------------===//
// Register FFT attributes decoding with the Xla runtime.
//===----------------------------------------------------------------------===//

namespace runtime {

XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(se::fft::Type);

}  // namespace runtime

//===----------------------------------------------------------------------===//
// Encoding from MHLO attributes to Xla runtime aggregate attributes.
//===----------------------------------------------------------------------===//

namespace gpu {

namespace mhlo = ::mlir::mhlo;

static se::fft::Type ConvertFftType(mhlo::FftType type) {
  switch (type) {
    case mhlo::FftType::FFT:
      return se::fft::Type::kC2CForward;
    case mhlo::FftType::IFFT:
      return se::fft::Type::kC2CInverse;
    case mhlo::FftType::RFFT:
      return se::fft::Type::kR2C;
    case mhlo::FftType::IRFFT:
      return se::fft::Type::kC2R;
    default:
      return se::fft::Type::kInvalid;
  }
}

void PopulateFftAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding) {
  encoding.Add<runtime::EnumAttrEncoding<mhlo::FftTypeAttr, mhlo::FftType,
                                         se::fft::Type>>(ConvertFftType);
}

//===----------------------------------------------------------------------===//
// FFT custom call implementation.
//===----------------------------------------------------------------------===//

static absl::Status FftImpl(const ServiceExecutableRunOptions* run_options,
                            State<std::unique_ptr<FftPlanCache>> state,
                            StridedMemrefView input, StridedMemrefView output,
                            absl::Span<const int64_t> fft_length,
                            se::fft::Type fft_type) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  if (input.dtype == PrimitiveType::F64 || input.dtype == PrimitiveType::C128) {
    // Adjust FFT type to reflect double precision.
    switch (fft_type) {
      case se::fft::Type::kC2CForward:
        fft_type = se::fft::Type::kZ2ZForward;
        break;
      case se::fft::Type::kC2CInverse:
        fft_type = se::fft::Type::kZ2ZInverse;
        break;
      case se::fft::Type::kR2C:
        fft_type = se::fft::Type::kD2Z;
        break;
      case se::fft::Type::kC2R:
        fft_type = se::fft::Type::kZ2D;
        break;
      default:
        return absl::InvalidArgumentError("Unsupported FFT type");
    }
  }

  absl::StatusOr<std::unique_ptr<FftPlanCache>*> fft_plan_cache =
      state.GetOrCreate([]() -> absl::StatusOr<std::unique_ptr<FftPlanCache>> {
        return std::make_unique<FftPlanCache>();
      });
  if (!fft_plan_cache.ok()) return fft_plan_cache.status();

  auto st =
      RunFft(GetDeviceAddress(input), ToShape(input), GetDeviceAddress(output),
             ToShape(output), fft_type, fft_length, executor->device_ordinal(),
             (*fft_plan_cache)->get(), stream, run_options->allocator());
  if (!st.ok()) return st;

  return absl::OkStatus();
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Fft, FunctionWrapper<FftImpl>(), checks,
    CustomCall::Bind("xla.gpu.fft")
        .UserData<const ServiceExecutableRunOptions*>()
        .State<std::unique_ptr<FftPlanCache>>("uid")
        .Arg<StridedMemrefView>()  // input
        .Arg<StridedMemrefView>()  // output
        .Attr<absl::Span<const int64_t>>("fft_length")
        .Attr<se::fft::Type>("fft_type"));

//===----------------------------------------------------------------------===//

void RegisterFftCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.fft", Fft);
}

}  // namespace gpu
}  // namespace xla
