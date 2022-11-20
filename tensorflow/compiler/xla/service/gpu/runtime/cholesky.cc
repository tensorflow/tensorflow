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

#include "tensorflow/compiler/xla/service/gpu/runtime/cholesky.h"

#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/xla.pb.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::Executable;

namespace {
struct Cholesky {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(const ServiceExecutableRunOptions* run_options,
                          const DebugOptions* debug_options,
                          runtime::StridedMemrefView operand,
                          runtime::StridedMemrefView a,
                          runtime::MemrefView workspace,
                          runtime::MemrefView info, int64_t batch_size,
                          bool is_lower, int64_t n) const;
  static Cholesky Handler() { return Cholesky(); }
};
}  // namespace

absl::Status Cholesky::operator()(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, runtime::StridedMemrefView operand,
    runtime::StridedMemrefView a, runtime::MemrefView workspace,
    runtime::MemrefView info, int64_t batch_size, bool is_lower,
    int64_t n) const {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  se::DeviceMemoryBase operand_buffer = GetDeviceAddress(operand);
  se::DeviceMemoryBase a_buffer = GetDeviceAddress(a);
  se::DeviceMemoryBase workspace_buffer = GetDeviceAddress(workspace);
  se::DeviceMemoryBase info_buffer = GetDeviceAddress(info);

  VLOG(3) << "Running Cholesky";
  se::Stream* stream = run_options->stream();

  // Copy operand to the a buffer if they are different.
  if (a.data != operand.data)
    stream->ThenMemcpy(&a_buffer, operand_buffer, operand_buffer.size());

  using UpperLower = se::blas::UpperLower;
  UpperLower uplo = is_lower ? UpperLower::kLower : UpperLower::kUpper;

  CholeskyParams params{n,        batch_size,       uplo,
                        a_buffer, workspace_buffer, info_buffer};
  auto executed = RunCholesky(xla::gpu::PtxOptsFromDebugOptions(*debug_options),
                              operand.dtype, &params, stream);
  if (!executed.ok()) return ToAbslStatus(executed);

  return absl::OkStatus();
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return absl::InternalError("Not implemented without Gpu");
#endif
}

static bool Cholesky(runtime::ExecutionContext* ctx, void** args, void** attrs,
                     void** rets) {
  static auto* handler = CustomCall::Bind("xla.gpu.cholesky")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // operand
                             .Arg<runtime::StridedMemrefView>()  // a
                             .Arg<runtime::MemrefView>()         // workspace
                             .Arg<runtime::MemrefView>()         // info
                             .Attr<int64_t>("batch_size")
                             .Attr<bool>("is_lower")
                             .Attr<int64_t>("n")
                             .To<checks>(Cholesky::Handler())
                             .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void RegisterCholeskyCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.cholesky", &xla::gpu::Cholesky);
}

}  // namespace gpu
}  // namespace xla
