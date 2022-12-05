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

#include "tensorflow/compiler/xla/service/gpu/runtime/memset.h"

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::Executable;

namespace {

struct Memset {
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          runtime::StridedMemrefView dst,
                          CustomCall::VariantArg constant) const;
  static Memset Handler() { return Memset(); }
};

}  // namespace

// TODO(ezhulenev): Add `VariantArg` type dispatching for all scalar types
// supported by Xla (PrimitiveType).

// Checks all supported data types to see if the value is zero.
static bool IsZero(CustomCall::VariantArg constant) {
  if (auto i1 = constant.get<bool>(); succeeded(i1))
    return *i1 == false;
  else if (auto i8 = constant.get<int8_t>(); succeeded(i8))
    return *i8 == 0;
  else if (auto i16 = constant.get<int16_t>(); succeeded(i16))
    return *i16 == 0;
  else if (auto i32 = constant.get<int32_t>(); succeeded(i32))
    return *i32 == 0;
  else if (auto i64 = constant.get<int64_t>(); succeeded(i64))
    return *i64 == 0;
  else if (auto bf16 = constant.get<bfloat16>(); succeeded(bf16))
    return *bf16 == bfloat16(0.0);
  else if (auto f16 = constant.get<half>(); succeeded(f16))
    return *f16 == half(0.0);
  else if (auto f32 = constant.get<float>(); succeeded(f32))
    return *f32 == 0.0;
  else if (auto f64 = constant.get<double>(); succeeded(f64))
    return *f64 == 0.0;

  return false;
}

// Convert constant value to 32-bit pattern.
static absl::StatusOr<uint32_t> ToBitPattern(CustomCall::VariantArg constant) {
  // If the value is 8 or 16 bits wide, we can emit a 32-bit memset by
  // repeating the value 4 or 2 times, so long as the destination buffer is
  // an even multiple of 32 bits long.
  //
  // This code is identical to `ir_emitter_unnested`.
  //
  // We use `memcpy` operation to copy bytes between value and the uint32_t bit
  // pattern because in theory they might have incompatible alignment, and we
  // rely on LLVM to optimize it.
  auto extend = [](auto value) -> uint32_t {
    static constexpr size_t num_bytes = sizeof(value);
    static_assert(num_bytes < 4);

    uint16_t pattern16;
    if constexpr (num_bytes == 1) {
      uint8_t b = value;
      pattern16 = uint16_t{b} | (uint16_t{b} << 8);
    } else {
      memcpy(&pattern16, &value, sizeof(pattern16));
    }
    return uint32_t{pattern16} | (uint32_t{pattern16} << 16);
  };

  // Truncate value to 32-bit pattern.
  auto truncate = [](auto value) -> uint32_t {
    static_assert(sizeof(value) >= 4);

    uint32_t pattern;
    memcpy(&pattern, &value, sizeof(pattern));
    return pattern;
  };

  if (auto i1 = constant.get<bool>(); succeeded(i1))
    return extend(*i1);
  else if (auto i8 = constant.get<int8_t>(); succeeded(i8))
    return extend(*i8);
  else if (auto i16 = constant.get<int16_t>(); succeeded(i16))
    return extend(*i16);
  else if (auto i32 = constant.get<int32_t>(); succeeded(i32))
    return truncate(*i32);
  else if (auto i64 = constant.get<int64_t>(); succeeded(i64))
    return truncate(*i64);
  else if (auto bf16 = constant.get<bfloat16>(); succeeded(bf16))
    return extend(static_cast<uint16_t>(*bf16));
  else if (auto f16 = constant.get<half>(); succeeded(f16))
    return extend(static_cast<uint16_t>(*f16));
  else if (auto f32 = constant.get<float>(); succeeded(f32))
    return truncate(*f32);
  else if (auto f64 = constant.get<double>(); succeeded(f64))
    return truncate(*f64);

  return absl::InvalidArgumentError("Unsupported memset constant type");
}

absl::Status Memset::operator()(const ServiceExecutableRunOptions* run_options,
                                runtime::StridedMemrefView dst,
                                CustomCall::VariantArg constant) const {
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryBase dst_data = GetDeviceAddress(dst);

  // If the constant is zero we can use memzero directly.
  if (IsZero(constant)) {
    stream->ThenMemZero(&dst_data, dst_data.size());
    return absl::OkStatus();
  }

  // If the constant is not zero, use the given pattern to `memset`.
  absl::StatusOr<uint32_t> pattern = ToBitPattern(constant);
  if (!pattern.ok()) return pattern.status();

  if (dst_data.size() % 4 != 0)
    return absl::InvalidArgumentError("Memref size is not divisible by 4");

  stream->ThenMemset32(&dst_data, *pattern, dst_data.size());

  return absl::OkStatus();
}

static bool MemsetFn(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.memset")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // dst
                             .Arg<CustomCall::VariantArg>()      // constant
                             .To<checks>(Memset::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void RegisterMemsetCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.memset", &MemsetFn);
}

}  // namespace gpu
}  // namespace xla
