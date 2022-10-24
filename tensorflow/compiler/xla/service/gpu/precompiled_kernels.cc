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

#include "tensorflow/compiler/xla/service/gpu/precompiled_kernels.h"

#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/util.h"

#if TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
namespace stream_executor {
namespace gpu {

extern void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                                   void** ptrs_out);

}
}  // namespace stream_executor
#endif

namespace xla {
namespace gpu {
namespace {

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//
// Generated from the following CUDA code.
//
// extern "C" {
// __global__ void __xla_MakeBatchPointers(char* base, int stride,
//                                         int n, void** ptrs_out) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= n) return;
//   ptrs_out[idx] = base + idx * stride;
// }
// }
constexpr const char* kMakeBatchPointersPtx = R"(
.version 4.2
.target sm_35
.address_size 64

.visible .entry __xla_MakeBatchPointers(
        .param .u64 __xla_MakeBatchPointers_param_0,
        .param .u32 __xla_MakeBatchPointers_param_1,
        .param .u32 __xla_MakeBatchPointers_param_2,
        .param .u64 __xla_MakeBatchPointers_param_3
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<8>;
        .reg .b64       %rd<8>;

        ld.param.u32    %r2, [__xla_MakeBatchPointers_param_2];
        mov.u32         %r3, %tid.x;
        mov.u32         %r4, %ctaid.x;
        mov.u32         %r5, %ntid.x;
        mad.lo.s32      %r6, %r4, %r5, %r3;
        setp.ge.s32     %p1, %r6, %r2;
        @%p1 bra        LBB0_2;
        ld.param.u64    %rd3, [__xla_MakeBatchPointers_param_0];
        ld.param.u64    %rd4, [__xla_MakeBatchPointers_param_3];
        cvta.to.global.u64      %rd5, %rd4;
        ld.param.u32    %r1, [__xla_MakeBatchPointers_param_1];
        mul.wide.s32    %rd6, %r6, 8;
        add.s64         %rd1, %rd5, %rd6;
        mul.lo.s32      %r7, %r6, %r1;
        cvt.s64.s32     %rd7, %r7;
        add.s64         %rd2, %rd3, %rd7;
        st.global.u64   [%rd1], %rd2;
LBB0_2:
        ret;
}
)";

// Lazily compiles ptx kernel, once per StreamExecutor.
//
// Thread-safe.
template <typename... KernelArgs>
class LazyKernel {
 public:
  LazyKernel(absl::string_view kernel_name, const char* ptx,
             const se::GpuAsmOpts& asm_opts)
      : kernel_name_(kernel_name), ptx_(ptx), asm_opts_(asm_opts) {}

  StatusOr<se::TypedKernel<KernelArgs...>*> Get(
      se::StreamExecutor* stream_exec) {
    absl::MutexLock lock(&mu_);

    auto result = kernels_.emplace(stream_exec, nullptr);
    if (result.second) {
      absl::Span<const uint8_t> compiled_ptx;
      StatusOr<absl::Span<const uint8_t>> compiled_ptx_or =
          se::CompileGpuAsmOrGetCached(stream_exec->device_ordinal(), ptx_,
                                       asm_opts_);
      if (compiled_ptx_or.ok()) {
        compiled_ptx = std::move(compiled_ptx_or).value();
      } else {
        static absl::once_flag logged_once;
        absl::call_once(logged_once, [&]() {
          LOG(WARNING)
              << compiled_ptx_or.status().ToString()
              << "\nRelying on driver to perform ptx compilation. "
              << "\nSetting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda "
              << " or modifying $PATH can be used to set the location of ptxas."
              << "\nThis message will only be logged once.";
        });
      }

      auto kernel = stream_exec->CreateTypedKernel<KernelArgs...>(
          kernel_name_, ptx_, compiled_ptx);
      if (kernel.ok()) {
        result.first->second = *std::move(kernel);
      } else {
        kernels_.erase(result.first);
        return kernel.status();
      }
    }
    return result.first->second.get();
  }

 private:
  std::string kernel_name_;
  const char* ptx_;
  se::GpuAsmOpts asm_opts_;

  absl::Mutex mu_;

  // A mutex keyed on StreamExecutor* is ok because StreamExecutors are never
  // destroyed.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::TypedKernel<KernelArgs...>>>
      kernels_ ABSL_GUARDED_BY(mu_);
};

}  // anonymous namespace

Status MakeBatchPointers(se::Stream* stream, const se::GpuAsmOpts& asm_opts,
                         se::DeviceMemoryBase base_ptr, int stride_bytes, int n,
                         se::DeviceMemoryBase ptrs_out) {
#if TENSORFLOW_USE_ROCM
  stream_executor::gpu::rocm_MakeBatchPointers(
      se::gpu::AsGpuStreamValue(stream),
      reinterpret_cast<char*>(base_ptr.opaque()), stride_bytes, n,
      reinterpret_cast<void**>(ptrs_out.opaque()));
#else
  static auto* lazy_kernel =
      new LazyKernel<se::DeviceMemoryBase /*base_ptr*/, int /*stride_bytes*/,
                     int /*n*/, se::DeviceMemoryBase /*ptrs_out*/>(
          "__xla_MakeBatchPointers", kMakeBatchPointersPtx, asm_opts);

  TF_ASSIGN_OR_RETURN(auto kernel, lazy_kernel->Get(stream->parent()));

  constexpr int kThreads = 128;
  TF_RETURN_IF_ERROR(
      stream->ThenLaunch(se::ThreadDim(kThreads, 1, 1),
                         se::BlockDim(CeilOfRatio(n, kThreads), 1, 1), *kernel,
                         base_ptr, stride_bytes, n, ptrs_out));
#endif
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
