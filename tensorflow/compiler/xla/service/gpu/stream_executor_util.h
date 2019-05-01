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
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/kernel_spec.h"

// Helper functions for interacting with StreamExecutor.

namespace xla {
namespace gpu {

// Returns true if the given StreamExecutor is for a Volta or newer nvidia GPU.
bool IsVoltaOrLater(const se::StreamExecutor& stream_exec);

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
XlaConvLayoutsToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                      const Layout& input, const Layout& filter,
                                      const Layout& output);

// Generates and returns a unique lock per each provided executor.
// Guarantees that blocks of code both holding a lock for the same provided
// executor (as given by this function) will not be running concurrently.
//
// This is used to prevent other XLA instances from trying to autotune on a
// device while another thread is using it.
tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec);

// Creates a type-safe kernel which can be launched with stream.ThenLaunch.
//
// The kernel has a provided name, and is based from provided PTX in ptx,
// and (optional) compiled PTX in cubin_data.
// The canonical storage for both ptx and cubin_data should outlive the
// lifetime of the kernel.
//
// This is a preferred API since it provides type safety for kernel launches.
template <typename... Args>
StatusOr<std::unique_ptr<se::TypedKernel<Args...>>> CreateTypedKernel(
    absl::string_view kernel_name, uint64 num_args, absl::string_view ptx,
    absl::Span<const uint8> cubin_data, se::StreamExecutor* stream_exec) {
  se::MultiKernelLoaderSpec loader_spec(num_args);
  loader_spec.AddCudaPtxInMemory(ptx, kernel_name);

  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(
        reinterpret_cast<const char*>(cubin_data.data()), kernel_name);
  }

  auto kernel_base = absl::make_unique<se::TypedKernel<Args...>>(stream_exec);
  if (!stream_exec->GetKernel(loader_spec, kernel_base.get())) {
    return InternalError("Unable to load kernel '%s'", kernel_name);
  }

  return std::move(kernel_base);
}

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
                             int64 threads_per_block, int64 block_count,
                             se::Stream* stream);

// Options for compiling with PTX.
struct PtxCompilationOptions {
  bool xla_gpu_disable_ptxas_optimizations;
  std::string xla_gpu_cuda_data_dir;

  using PtxOptionsTuple = std::tuple<bool, std::string>;

  explicit PtxCompilationOptions(const HloModuleConfig& hlo_module_config)
      : xla_gpu_disable_ptxas_optimizations(
            hlo_module_config.debug_options()
                .xla_gpu_disable_ptxas_optimizations()),
        xla_gpu_cuda_data_dir(
            hlo_module_config.debug_options().xla_gpu_cuda_data_dir()) {}

  // For comparison and hashing.
  PtxOptionsTuple ToTuple() {
    return std::make_tuple(xla_gpu_disable_ptxas_optimizations,
                           xla_gpu_cuda_data_dir);
  }
};

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array.
//
// Queries stream executor stream_exec to get CUDA compute capability from the
// device.
//
// compile_ptx_options is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
// It can be constructed from HloModuleConfig.
StatusOr<std::vector<uint8>> CompilePtx(
    se::StreamExecutor* stream_exec, absl::string_view ptx,
    PtxCompilationOptions compile_ptx_options);

// Returns a vector of potential locations of the CUDA root directory.
// Searches through tensorflow CUDA locations AND through the CUDA location
// specified in compile_ptx_options (can be constructed from HloModuleConfig).
std::vector<string> GetCudaRootCandidates(
    PtxCompilationOptions compile_ptx_options);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_STREAM_EXECUTOR_UTIL_H_
