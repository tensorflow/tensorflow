/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
#define XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/layout.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

// Helper functions for interacting with StreamExecutor.

namespace xla {
namespace gpu {

// Returns DNN version info from provided stream executor when possible,
// fallback version otherwise.
se::dnn::VersionInfo GetDnnVersionInfo(
    stream_executor::StreamExecutor* stream_exec,
    se::dnn::VersionInfo fallback_version = se::dnn::VersionInfo{0, 0, 0});

// Returns (input, filter, output) XLA Layout protos given the StreamExecutor
// layouts.
absl::StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      se::dnn::DataLayout input,
                                      se::dnn::FilterLayout filter,
                                      se::dnn::DataLayout output);

// Returns (input, filter, output) StreamExecutor layouts given the XLA layouts.
absl::StatusOr<
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
std::tuple<std::optional<int64_t>, std::optional<int64_t>,
           std::optional<int64_t>>
FindVectorizedFeatureDims(const ConvolutionDimensionNumbers& dnums,
                          const Shape& input, const Shape& filter,
                          const Shape& output);

// Generates and returns a unique lock per the provided executor.
// Guarantees that blocks of code running for the same provided
// executor will not be running concurrently if they lock the returned mutex.
//
// This is used to prevent other XLA instances from trying to autotune on a
// device while another thread is using it.
absl::Mutex& GetGpuMutex(const se::StreamExecutor* stream_exec);

// Creates a kernel with a provided name, based from provided PTX in ptx.
// The kernel should be executed using the provided executor.
// The argument cubin_data represents compiled PTX and may be left empty.
//
// The canonical storage for both ptx and cubin_data should outlive
// the lifetime of the kernel.
absl::StatusOr<std::unique_ptr<se::Kernel>> CreateKernel(
    absl::string_view kernel_name, uint64_t num_args, absl::string_view ptx,
    absl::Span<const uint8_t> cubin_data, se::StreamExecutor* stream_exec,
    uint32_t shared_mem_bytes = 0);

// Runs loaded kernel on the stream with the provided arguments.
absl::Status ExecuteKernelOnStream(const se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   se::Stream* stream);

// Runs loaded kernel on the stream with the provided arguments.
absl::Status ExecuteKernelOnStream(const se::Kernel& kernel,
                                   absl::Span<const se::DeviceMemoryBase> args,
                                   const LaunchDimensions& dims,
                                   const se::ClusterDim& cluster_dim,
                                   se::Stream* stream);

// Initializes `buffer` with random data on `stream`.
// `rng_state` is an inout parameter for the pseudorandom generator state.
// `buffer_type` determines what buffer would be filled out with.
//
// Precondition: `buffer_type` is a floating point type, `rng_state` needs to be
// initialized to zero on the first use.
void InitializeBuffer(se::Stream* stream, PrimitiveType buffer_type,
                      int64_t* rng_state, se::DeviceMemoryBase buffer);

absl::StatusOr<se::dnn::ConvolutionKind> GetDNNConvKindFromCudnnConvKind(
    CudnnConvKind kind);

absl::StatusOr<se::dnn::NormKind> GetDNNNormKindFromCudnnNormKind(
    CudnnNormKind kind);

absl::StatusOr<se::dnn::FusedMHAKind> GetDNNFusedMHAKindFromCudnnfMHAKind(
    CudnnfMHAKind kind);

absl::StatusOr<se::dnn::DataType> GetDNNDataTypeFromPrimitiveType(
    PrimitiveType type);

// Returns result with the smallest time which has not failed.
// If deterministic output is requested, returns first (not failing) result.
absl::StatusOr<AutotuneResult> PickBestResult(
    absl::Span<AutotuneResult const> profile_results,
    std::optional<std::string_view> instr_str,
    HloModuleConfig hlo_module_config);

// Returns whether determinism is required.
bool RequireDeterminism(const HloModuleConfig& config);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
