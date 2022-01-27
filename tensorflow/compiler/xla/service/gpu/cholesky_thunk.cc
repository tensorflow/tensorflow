/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"

#include <complex>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"

namespace xla {
namespace gpu {

namespace {

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(batch_size)].
//
// Generated from the following CUDA code.
//
// extern "C" {
// __global__ void __xla_SetCholeskyPointers(char* base, int stride, int
//                                           batch_size, void** as) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= batch_size) return;
//   as[idx] = base + idx * stride;
// }
// }
constexpr const char* kSetPotrfBatchedPointersPtx = R"(
.version 4.2
.target sm_35
.address_size 64

.visible .entry __xla_SetCholeskyPointers(
        .param .u64 __xla_SetCholeskyPointers_param_0,
        .param .u32 __xla_SetCholeskyPointers_param_1,
        .param .u32 __xla_SetCholeskyPointers_param_2,
        .param .u64 __xla_SetCholeskyPointers_param_3
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<8>;
        .reg .b64       %rd<8>;

        ld.param.u32    %r2, [__xla_SetCholeskyPointers_param_2];
        mov.u32         %r3, %tid.x;
        mov.u32         %r4, %ctaid.x;
        mov.u32         %r5, %ntid.x;
        mad.lo.s32      %r6, %r4, %r5, %r3;
        setp.ge.s32     %p1, %r6, %r2;
        @%p1 bra        LBB0_2;
        ld.param.u64    %rd3, [__xla_SetCholeskyPointers_param_0];
        ld.param.u64    %rd4, [__xla_SetCholeskyPointers_param_3];
        cvta.to.global.u64      %rd5, %rd4;
        ld.param.u32    %r1, [__xla_SetCholeskyPointers_param_1];
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

StatusOr<GpuSolverContext*> GetContext(se::Stream* stream) {
  // TODO(b/214454412): This global hashtable is incorrect (ABA bug if a Stream
  // is added to the hasthable, then deleted, and then a new Stream is created
  // at the same address).  It also leaks memory!
  static absl::Mutex mu(absl::kConstInit);
  static auto contexts =
      new absl::flat_hash_map<se::Stream*, GpuSolverContext> ABSL_GUARDED_BY(
          mu);

  absl::MutexLock lock(&mu);
  auto result = contexts->emplace(stream, GpuSolverContext());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second, GpuSolverContext::Create(stream));
  }
  return &result.first->second;
}

// Gets a launchable kernel runnable on `executor` that runs
// the function defined by kSetPotrfBatchedPointersPtx.
using SetPotrfBatchedPointersKernel =
    se::TypedKernel<se::DeviceMemoryBase /*a_base*/, int /*stride*/,
                    int /*batch_size*/, se::DeviceMemoryBase /*as*/>;
StatusOr<SetPotrfBatchedPointersKernel*> GetSetPotrfBatchedPointersKernel(
    se::StreamExecutor* executor, const se::GpuAsmOpts& asm_opts) {
  // A global hashtable keyed on StreamExecutor* is ok because StreamExecutors
  // are never destroyed.
  static absl::Mutex mu(absl::kConstInit);
  static auto kernels =
      new absl::flat_hash_map<se::StreamExecutor*,
                              std::unique_ptr<SetPotrfBatchedPointersKernel>>
          ABSL_GUARDED_BY(mu);

  absl::MutexLock lock(&mu);
  auto result = kernels->emplace(executor, nullptr);
  if (result.second) {
    absl::Span<const uint8_t> compiled_ptx;
    StatusOr<absl::Span<const uint8_t>> compiled_ptx_or =
        se::CompileGpuAsmOrGetCached(executor->device_ordinal(),
                                     kSetPotrfBatchedPointersPtx, asm_opts);
    if (compiled_ptx_or.ok()) {
      compiled_ptx = compiled_ptx_or.ConsumeValueOrDie();
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

    StatusOr<std::unique_ptr<SetPotrfBatchedPointersKernel>> kernel =
        executor->CreateTypedKernel<se::DeviceMemoryBase, int, int,
                                    se::DeviceMemoryBase>(
            "__xla_SetCholeskyPointers", kSetPotrfBatchedPointersPtx,
            compiled_ptx);
    if (kernel.ok()) {
      result.first->second = *std::move(kernel);
    } else {
      kernels->erase(result.first);
      return kernel.status();
    }
  }

  return result.first->second.get();
}

}  // namespace

CholeskyThunk::CholeskyThunk(ThunkInfo thunk_info,
                             const CholeskyOptions& options,
                             const se::GpuAsmOpts asm_opts,
                             BufferAllocation::Slice a_buffer,
                             BufferAllocation::Slice workspace_buffer,
                             BufferAllocation::Slice info_buffer,
                             PrimitiveType type, int64_t batch_size, int64_t n)
    : Thunk(Kind::kCholesky, thunk_info),
      asm_opts_(asm_opts),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      a_buffer_(a_buffer),
      workspace_buffer_(workspace_buffer),
      info_buffer_(info_buffer),
      type_(type),
      batch_size_(batch_size),
      n_(n) {}

Status CholeskyThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "type=" << PrimitiveType_Name(type_)
          << " uplo=" << se::blas::UpperLowerString(uplo_)
          << " batch_size=" << batch_size_ << " n=" << n_
          << " a=" << a_buffer_.ToString()
          << " workspace=" << workspace_buffer_.ToString()
          << " info=" << info_buffer_.ToString();

  TF_ASSIGN_OR_RETURN(GpuSolverContext * context, GetContext(params.stream));
  if (context->SupportsPotrfBatched()) {
    switch (type_) {
      case F32:
        return DoPotrfBatched<float>(params, context);
      case F64:
        return DoPotrfBatched<double>(params, context);
      case C64:
        return DoPotrfBatched<std::complex<float>>(params, context);
      case C128:
        return DoPotrfBatched<std::complex<double>>(params, context);
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type_));
    }
  } else {
    switch (type_) {
      case F32:
        return DoPotrfUnbatched<float>(params, context);
      case F64:
        return DoPotrfUnbatched<double>(params, context);
      case C64:
        return DoPotrfUnbatched<std::complex<float>>(params, context);
      case C128:
        return DoPotrfUnbatched<std::complex<double>>(params, context);
      default:
        return InvalidArgument("Invalid type for cholesky %s",
                               PrimitiveType_Name(type_));
    }
  }
}

template <typename T>
Status CholeskyThunk::DoPotrfBatched(const ExecuteParams& params,
                                     GpuSolverContext* context) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  T* a_base = static_cast<T*>(
      params.buffer_allocations->GetDeviceAddress(a_buffer_).opaque());
  se::DeviceMemory<int> infos(
      params.buffer_allocations->GetDeviceAddress(info_buffer_));
  se::DeviceMemory<T*> as(
      params.buffer_allocations->GetDeviceAddress(workspace_buffer_));

  CHECK_GE(as.size(), batch_size_);
  CHECK_GE(infos.size(), batch_size_);

  // Run a kernel that sets as[i] = &a_base[i * stride].
  //
  // A simpler way of doing this would be to create this buffer on the host and
  // then copy it to device.  But using a kernel instead of an H2D copy avoids a
  // few performance pitfalls.
  //
  //  - Only one H2D copy can run on a given GPU at a time.  If there's already
  //    a copy ongoing as part of other work on the GPU, our copy here will
  //    block.  In contrast, multiple kernels can run simultaneously.
  //
  //  - H2D copies from CUDA unpinned memory can acquire a global lock in the
  //    driver and slow down *all* work on the GPU.  So to do this right, we'd
  //    need to allocate the host memory as pinned, one alloc per stream.  Then
  //    we'd need to manage this memory without leaks.  This becomes complex!
  TF_ASSIGN_OR_RETURN(auto kernel,
                      GetSetPotrfBatchedPointersKernel(executor, asm_opts_));

  constexpr int64_t kThreads = 128;
  const int64_t stride_bytes = n_ * n_ * sizeof(T);
  stream->ThenLaunch(se::ThreadDim(kThreads, 1, 1),
                     se::BlockDim(CeilOfRatio(batch_size_, kThreads), 1, 1),
                     *kernel, se::DeviceMemoryBase(a_base),
                     static_cast<int>(stride_bytes),
                     static_cast<int>(batch_size_), se::DeviceMemoryBase(as));

  // Now that we've set up the `as` array, we can finally call cusolver.
  return context->PotrfBatched(uplo_, n_, as, n_, infos, batch_size_);
}

template <typename T>
Status CholeskyThunk::DoPotrfUnbatched(const ExecuteParams& params,
                                       GpuSolverContext* context) {
  T* a_base = static_cast<T*>(
      params.buffer_allocations->GetDeviceAddress(a_buffer_).opaque());
  int* info_base = static_cast<int*>(
      params.buffer_allocations->GetDeviceAddress(info_buffer_).opaque());
  se::DeviceMemoryBase workspace =
      params.buffer_allocations->GetDeviceAddress(workspace_buffer_);

  int64_t stride = n_ * n_;
  for (int64_t i = 0; i < batch_size_; ++i) {
    se::DeviceMemory<T> a_data(
        se::DeviceMemoryBase(&a_base[i * stride], sizeof(T) * stride));
    se::DeviceMemory<int> info_data(
        se::DeviceMemoryBase(&info_base[i], sizeof(int)));
    TF_RETURN_IF_ERROR(
        context->Potrf(uplo_, n_, a_data, n_, info_data, workspace));
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
