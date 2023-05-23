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

using ::xla::runtime::CustomCall;
using ::xla::runtime::MemrefView;
using ::xla::runtime::StridedMemrefView;

static absl::Status CholeskyImpl(const ServiceExecutableRunOptions* run_options,
                                 const DebugOptions* debug_options,
                                 StridedMemrefView operand, StridedMemrefView a,
                                 MemrefView workspace, MemrefView info,
                                 int64_t batch_size, bool is_lower, int64_t n) {
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
  if (!executed.ok()) return executed;

  return absl::OkStatus();
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return absl::InternalError("Cholesky is not supported without GPU");
#endif
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Cholesky, FunctionWrapper<CholeskyImpl>(), checks,
    CustomCall::Bind("xla.gpu.cholesky")
        .UserData<const ServiceExecutableRunOptions*>()
        .UserData<const DebugOptions*>()
        .Arg<StridedMemrefView>()  // operand
        .Arg<StridedMemrefView>()  // a
        .Arg<MemrefView>()         // workspace
        .Arg<MemrefView>()         // info
        .Attr<int64_t>("batch_size")
        .Attr<bool>("is_lower")
        .Attr<int64_t>("n"));

void RegisterCholeskyCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.cholesky", Cholesky);
}

}  // namespace gpu
}  // namespace xla
