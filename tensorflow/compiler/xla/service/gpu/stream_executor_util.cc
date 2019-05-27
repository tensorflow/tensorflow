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

#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/kernel_spec.h"

namespace xla {
namespace gpu {

using se::dnn::DataLayout;
using se::dnn::DataLayoutString;
using se::dnn::FilterLayout;
using se::dnn::FilterLayoutString;

bool IsVoltaOrLater(const se::StreamExecutor& stream_executor) {
  int major, minor;
  CHECK(stream_executor.GetDeviceDescription().cuda_compute_capability(&major,
                                                                       &minor));
  return major >= 7;
}

StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      DataLayout input, FilterLayout filter,
                                      DataLayout output) {
  std::vector<int64> input_layout;
  switch (input) {
    case DataLayout::kBatchDepthYX:
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.push_back(dnums.input_feature_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      break;
    case DataLayout::kBatchYXDepth:
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      input_layout.push_back(dnums.input_feature_dimension());
      break;
    default:
      return InternalError("Invalid input layout %s for conv with dnums %s",
                           DataLayoutString(input),
                           ConvolutionDimensionNumbersToString(dnums));
  }

  std::vector<int64> filter_layout;
  switch (filter) {
    case FilterLayout::kOutputInputYX:
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      break;
    case FilterLayout::kOutputYXInput:
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      break;
    default:
      return InternalError("Invalid filter layout %s for conv with dnums %s",
                           FilterLayoutString(filter),
                           ConvolutionDimensionNumbersToString(dnums));
  }

  std::vector<int64> output_layout;
  switch (output) {
    case DataLayout::kBatchDepthYX:
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.push_back(dnums.output_feature_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      break;
    case DataLayout::kBatchYXDepth:
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      output_layout.push_back(dnums.output_feature_dimension());
      break;
    default:
      return InternalError("Invalid output layout %s for conv with dnums %s",
                           DataLayoutString(output),
                           ConvolutionDimensionNumbersToString(dnums));
  }

  return std::make_tuple(LayoutUtil::MakeLayoutFromMajorToMinor(input_layout),
                         LayoutUtil::MakeLayoutFromMajorToMinor(filter_layout),
                         LayoutUtil::MakeLayoutFromMajorToMinor(output_layout));
}

StatusOr<std::tuple<DataLayout, FilterLayout, DataLayout>>
XlaConvLayoutsToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                      const Layout& input, const Layout& filter,
                                      const Layout& output) {
  Layout nchw_input, nchw_filter, nchw_output;
  std::tie(nchw_input, nchw_filter, nchw_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchDepthYX,
                                            FilterLayout::kOutputInputYX,
                                            DataLayout::kBatchDepthYX)
          .ConsumeValueOrDie();

  Layout nhwc_input, nhwc_filter, nhwc_output;
  std::tie(nhwc_input, nhwc_filter, nhwc_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchYXDepth,
                                            FilterLayout::kOutputYXInput,
                                            DataLayout::kBatchYXDepth)
          .ConsumeValueOrDie();

  DataLayout input_layout;
  if (LayoutUtil::Equal(input, nchw_input)) {
    input_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(input, nhwc_input)) {
    input_layout = DataLayout::kBatchYXDepth;
  } else {
    return InternalError("Invalid input layout %s for conv with dnums %s",
                         LayoutUtil::HumanString(input),
                         ConvolutionDimensionNumbersToString(dnums));
  }

  FilterLayout filter_layout;
  if (LayoutUtil::Equal(filter, nchw_filter)) {
    filter_layout = FilterLayout::kOutputInputYX;
  } else if (LayoutUtil::Equal(filter, nhwc_filter)) {
    filter_layout = FilterLayout::kOutputYXInput;
  } else {
    return InternalError("Invalid filter layout %s for conv with dnums %s",
                         LayoutUtil::HumanString(filter),
                         ConvolutionDimensionNumbersToString(dnums));
  }

  DataLayout output_layout;
  if (LayoutUtil::Equal(output, nchw_output)) {
    output_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(output, nhwc_output)) {
    output_layout = DataLayout::kBatchYXDepth;
  } else {
    return InternalError("Invalid output layout %s for conv with dnums %s",
                         LayoutUtil::HumanString(output),
                         ConvolutionDimensionNumbersToString(dnums));
  }

  return std::make_tuple(input_layout, filter_layout, output_layout);
}

tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  // se::Platform*s are global singletons guaranteed to live forever.
  static auto* mutexes =
      new std::map<std::pair<const se::Platform*, /*device_ordinal*/ int64>,
                   tensorflow::mutex>();

  tensorflow::mutex_lock global_lock(mu);
  auto it = mutexes
                ->emplace(std::piecewise_construct,
                          std::make_tuple(stream_exec->platform(),
                                          stream_exec->device_ordinal()),
                          std::make_tuple())
                .first;
  return tensorflow::mutex_lock{it->second};
}

StatusOr<std::unique_ptr<se::KernelBase>> CreateKernel(
    absl::string_view kernel_name, uint64 num_args, absl::string_view ptx,
    absl::Span<const uint8> cubin_data, se::StreamExecutor* stream_exec) {
  se::MultiKernelLoaderSpec loader_spec(num_args);
  loader_spec.AddCudaPtxInMemory(ptx, kernel_name);

  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(
        reinterpret_cast<const char*>(cubin_data.data()), kernel_name);
  }

  auto kernel_base = absl::make_unique<se::KernelBase>(stream_exec);
  if (!stream_exec->GetKernel(loader_spec, kernel_base.get())) {
    return InternalError("Unable to load kernel '%s'", kernel_name);
  }

  return std::move(kernel_base);
}

Status ExecuteKernelOnStream(const se::KernelBase& kernel,
                             absl::Span<const se::DeviceMemoryBase> args,
                             int64 threads_per_block, int64 block_count,
                             se::Stream* stream) {
  static constexpr int kKernelArgsLimit = 1024;
  auto kernel_args = absl::make_unique<se::KernelArgsArray<kKernelArgsLimit>>();
  for (const se::DeviceMemoryBase& buf : args) {
    kernel_args->add_device_memory_argument(buf);
  }

  if (!stream->parent()->Launch(stream, se::ThreadDim(threads_per_block),
                                se::BlockDim(block_count), kernel,
                                *kernel_args)) {
    return InternalError("Unable to launch kernel");
  }
  return Status::OK();
}

se::cuda::PtxCompilationOptions PtxOptsFromConfig(
    const HloModuleConfig& hlo_module_config) {
  return se::cuda::PtxCompilationOptions(
      hlo_module_config.debug_options().xla_gpu_disable_ptxas_optimizations(),
      hlo_module_config.debug_options().xla_gpu_cuda_data_dir());
}

}  // namespace gpu
}  // namespace xla
