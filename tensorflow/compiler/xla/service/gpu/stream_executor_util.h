/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/kernel_spec.h"

// Helper functions for interacting with StreamExecutor.

namespace xla {
namespace gpu {

// Returns (input, filter, output) XLA Layout protos given the StreamExecutor
// layouts.
StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      se::dnn::DataLayout input,
                                      se::dnn::FilterLayout filter,
                                      se::dnn::DataLayout output);

// Returns (input, filter, output) StreamExecutor layouts given the XLA layouts.
StatusOr<
    std::tuple<se::dnn::DataLayout, se::dnn::FilterLayout, se::dnn::DataLayout>>
XlaConvShapesToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                     const Shape& input, const Shape& filter,
                                     const Shape& output);

// Finds the VECT_C dimension in input/filter/output, if present.
//
// A cudnn convolution may have layout NCHW_VECT_C, which means instead of
// [N,C,H,W], the layout is [N,C/k,H,W,k] for some k (usually 4 or 32).
//
// ConvolutionDimensionNumbers doesn't explicitly store which is the `k`
// dimension, because only cudnn convolutions have this feature; it's not
// applicable elsewhere.  We find it by finding a dimension in the
// input/filter/output shape that is *not* in dnums.
std::tuple<absl::optional<int64>, absl::optional<int64>, absl::optional<int64>>
FindVectorizedFeatureDims(const ConvolutionDimensionNumbers& dnums,
                          const Shape& input, const Shape& filter,
                          const Shape& output);

// Generates and returns a unique lock per each provided executor.
// Guarantees that blocks of code both holding a lock for the same provided
// executor (as given by this function) will not be running concurrently.
//
// This is used to prevent other XLA instances from trying to autotune on a
// device while another thread is using it.
tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec);

// Creates a kernel with a provided name, based from provided PTX in ptx.
// The kernel should be executed using the provided executor.
// The argument cubin_data represents compiled PTX and may be left empty.
//
// The canonical storage for both ptx and cubin_data should outlive
// the lifetime of the kernel.
StatusOr<std::unique_ptr<se::KernelBase>> CreateKernel(
    absl::string_view kernel_name, uint64 num_args, absl::string_view ptx,
    absl::Span<const uint8> cubin_data, se::StreamExecutor* stream_exec);

// Runs loaded kernel on the stream with the provided arguments.
Status ExecuteKernelOnStream(const se::KernelBase& kernel,
                             absl::Span<const se::DeviceMemoryBase> args,
                             const LaunchDimensions& dims, se::Stream* stream);

// Create GpuAsmOpts out of HloModuleConfig.
se::GpuAsmOpts PtxOptsFromConfig(const HloModuleConfig& hlo_module_config);

// Initializes `buffer` with random data on `stream`.
// `rng_state` is an inout parameter for the pseudorandom generator state.
// `buffer_type` determines what buffer would be filled out with.
//
// Precondition: `buffer_type` is a floating point type, `rng_state` needs to be
// initialized to zero on the first use.
void InitializeBuffer(se::Stream* stream, PrimitiveType buffer_type,
                      int64* rng_state, se::DeviceMemoryBase buffer);

StatusOr<se::dnn::ConvolutionKind> GetDNNConvKindFromCudnnConvKind(
    CudnnConvKind kind);
StatusOr<se::dnn::DataType> GetDNNDataTypeFromPrimitiveType(PrimitiveType type);

// Returns result with the smallest time which has not failed.
// If deterministic output is requested, returns first (not failing) result.
StatusOr<tensorflow::AutotuneResult> PickBestResult(
    absl::Span<tensorflow::AutotuneResult const> profile_results,
    const HloInstruction& instr);

// Returns whether determinism is required.
//
// The following function allows deterministic ops to be implemented relatively
// quickly using environment variables. It is intended to be temporary. The
// longer-term intention is to enable deterministic ops via tf.config and
// appropriate plumbing. See the discussion on PR 34951 for more information:
// https://github.com/tensorflow/tensorflow/pull/34951#discussion_r355682316
// This function and associated comment are replicated in the following three
// places:
//   1. tensorflow/core/kernels/gpu_utils.cc
//   2. tensorflow/stream_executor/cuda/cuda_dnn.cc
// When implementing the plumbing, you should also search for the use of
// TF_DETERMINISTIC_OPS on its own.
// TODO(duncanriach): move to an API that uses tf.config and implement the first
//                    phase of plumbing.
bool RequireDeterminism(const HloModuleConfig& config);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
