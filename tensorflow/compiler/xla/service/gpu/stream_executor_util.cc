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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/kernel_spec.h"

namespace xla {
namespace gpu {

namespace {

using se::dnn::DataLayout;
using se::dnn::DataLayoutString;
using se::dnn::FilterLayout;
using se::dnn::FilterLayoutString;
using tensorflow::AutotuneResult;

// Returns the smallest integer >= 0 that's not in the given set of numbers.
//
// For example, FindMissingDnum({1, 0, 3, 4}) returns 2.
//
// This is useful for handling DataLayout::kBatchDepthYX4, which repesents a
// layout [N, C/k, H, W, k] for some constant k, usually 4 or 32.
// ConvolutionDimensionNumbers doesn't explicitly say which dimension is `k`,
// but we can infer it by finding the first dnum that isn't otherwise mentioned
// in the dnums.
int64 FindMissingDnum(absl::Span<const int64> vals) {
  for (int i = 0; i < vals.size(); i++) {
    if (!absl::c_linear_search(vals, i)) {
      return i;
    }
  }
  return vals.size();
}

}  // anonymous namespace

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
    case DataLayout::kBatchDepthYX:  // NCHW
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.push_back(dnums.input_feature_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      break;
    case DataLayout::kBatchDepthYX4:  // NCHW_VECT_C
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.push_back(dnums.input_feature_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      input_layout.push_back(FindMissingDnum(input_layout));
      break;
    case DataLayout::kBatchYXDepth:  // NHWC
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
    case FilterLayout::kOutputInputYX:  // OIHW
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      break;
    case FilterLayout::kOutputInputYX4:  // OIHW_VECT_C
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      filter_layout.push_back(FindMissingDnum(filter_layout));
      break;
    case FilterLayout::kOutputYXInput:  // OHWI
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
    case DataLayout::kBatchDepthYX:  // NCHW
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.push_back(dnums.output_feature_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      break;
    case DataLayout::kBatchDepthYX4:  // NCHW_VECT_C
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.push_back(dnums.output_feature_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      output_layout.push_back(FindMissingDnum(output_layout));
      break;
    case DataLayout::kBatchYXDepth:  // NHWC
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

  Layout nchw4_input, nchw4_filter, nchw4_output;
  std::tie(nchw4_input, nchw4_filter, nchw4_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchDepthYX4,
                                            FilterLayout::kOutputInputYX4,
                                            DataLayout::kBatchDepthYX4)
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
  } else if (LayoutUtil::Equal(input, nchw4_input)) {
    input_layout = DataLayout::kBatchDepthYX4;
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
  } else if (LayoutUtil::Equal(filter, nchw4_filter)) {
    filter_layout = FilterLayout::kOutputInputYX4;
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
  } else if (LayoutUtil::Equal(output, nchw4_output)) {
    output_layout = DataLayout::kBatchDepthYX4;
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
  TF_RETURN_IF_ERROR(stream_exec->GetKernel(loader_spec, kernel_base.get()));
  return std::move(kernel_base);
}

Status ExecuteKernelOnStream(const se::KernelBase& kernel,
                             absl::Span<const se::DeviceMemoryBase> args,
                             const LaunchDimensions& dims, se::Stream* stream) {
  static constexpr int kKernelArgsLimit = 1024;
  auto kernel_args = absl::make_unique<se::KernelArgsArray<kKernelArgsLimit>>();
  for (const se::DeviceMemoryBase& buf : args) {
    kernel_args->add_device_memory_argument(buf);
  }
  LaunchDimensions::Dim3D thread_counts = dims.thread_counts_per_block();
  LaunchDimensions::Dim3D block_counts = dims.block_counts();
  return stream->parent()->Launch(
      stream, se::ThreadDim(thread_counts.x, thread_counts.y, thread_counts.z),
      se::BlockDim(block_counts.x, block_counts.y, block_counts.z), kernel,
      *kernel_args);
}

se::GpuAsmOpts PtxOptsFromConfig(const HloModuleConfig& hlo_module_config) {
  string extra_string =
      hlo_module_config.debug_options().xla_gpu_asm_extra_flags();
  std::vector<std::string> extra_flags;
  extra_flags = absl::StrSplit(extra_string, ',', absl::SkipEmpty());
  return se::GpuAsmOpts(
      hlo_module_config.debug_options().xla_gpu_disable_gpuasm_optimizations(),
      hlo_module_config.debug_options().xla_gpu_cuda_data_dir(), extra_flags);
}

// Unimplemented for integers yet.
template <typename T, typename Generator>
typename std::enable_if<std::is_integral<T>::value,
                        T>::type static UniformDistribution(T lhs, T rhs,
                                                            Generator* gen) =
    delete;

template <typename T, typename Generator>
typename std::enable_if<std::is_floating_point<T>::value,
                        T>::type static UniformDistribution(T lhs, T rhs,
                                                            Generator* gen) {
  return std::uniform_real_distribution<T>(lhs, rhs)(*gen);
}

template <typename T>
static void InitializeTypedBuffer(se::Stream* stream,
                                  se::DeviceMemoryBase buffer,
                                  int64* rng_state) {
  // Accesses to static variables are not locked, since the caller is already
  // in a critical section.
  static std::vector<T>* host_buffer = [] {
    // Use a large prime number to fragment the accesses.
    auto* ret = new std::vector<T>(10069);
    // Default-seeded random numbers.
    std::mt19937 gen;
    for (auto& element : *ret) {
      // Only double gets random values in double.  Other data types get random
      // values in float then cast them to the target data types.
      using RandomFloatingPointType =
          typename std::conditional<std::is_same<T, Eigen::half>::value, float,
                                    T>::type;
      using RandomType =
          typename std::conditional<std::is_integral<T>::value, float,
                                    RandomFloatingPointType>::type;
      // Scale down the values for fp16 to have less overflows.
      auto upper_bound =
          RandomType(std::is_same<T, Eigen::half>::value ? 0.1 : 1.0);
      auto rand_val = UniformDistribution(RandomType(0), upper_bound, &gen);
      // For float or double, it is between [0,1].
      // For fp16, it ranges between [0, 0.1].
      // For integer types, element is either 0 or 1 for less overflows
      // especially for int8.
      element = T(std::is_integral<T>::value ? rand_val + 0.5 : rand_val);
    }
    return ret;
  }();

  int64& host_index = *rng_state;

  char* current_addr = static_cast<char*>(buffer.opaque());
  CHECK_EQ(0, buffer.size() % sizeof(T));
  int64 elements_left = buffer.size() / sizeof(T);
  while (elements_left > 0) {
    CHECK_LE(host_index, host_buffer->size());
    if (host_buffer->size() == host_index) {
      host_index = 0;
    }
    int64 elements_copied =
        std::min<int64>(host_buffer->size() - host_index, elements_left);
    se::DeviceMemoryBase mem(current_addr, elements_copied * sizeof(T));
    stream->ThenMemcpy(&mem, host_buffer->data() + host_index,
                       elements_copied * sizeof(T));
    current_addr += elements_copied * sizeof(T);
    elements_left -= elements_copied;
    host_index += elements_copied;
  }
}

void InitializeBuffer(se::Stream* stream, PrimitiveType buffer_type,
                      int64* rng_state, se::DeviceMemoryBase buffer) {
  switch (buffer_type) {
    case xla::F16:
      return InitializeTypedBuffer<Eigen::half>(stream, buffer, rng_state);
    case xla::F32:
    case xla::C64:
      return InitializeTypedBuffer<float>(stream, buffer, rng_state);
    case xla::F64:
    case xla::C128:
      return InitializeTypedBuffer<double>(stream, buffer, rng_state);
    case xla::S8:
      return InitializeTypedBuffer<int8>(stream, buffer, rng_state);
    default:
      LOG(FATAL) << "Unexpected type";
  }
}

StatusOr<se::dnn::ConvolutionKind> GetDNNConvKindFromCudnnConvKind(
    CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kBackwardFilter:
      return se::dnn::BACKWARD_FILTER;
    case CudnnConvKind::kBackwardInput:
      return se::dnn::BACKWARD_DATA;
    case CudnnConvKind::kForward:
      return se::dnn::FORWARD;
    default:
      break;
  }
  return InternalError("Unexpected convolution kind");
}

StatusOr<se::dnn::DataType> GetDNNDataTypeFromPrimitiveType(
    PrimitiveType type) {
  switch (type) {
    case F16:
      return se::dnn::ToDataType<Eigen::half>::value;
    case F32:
      return se::dnn::ToDataType<float>::value;
    case F64:
      return se::dnn::ToDataType<double>::value;
    default:
      break;
  }
  return InternalError("Unsupported convolution datatype");
}

bool RequireDeterminism(const HloModuleConfig& config) {
  static bool require_cudnn_determinism = [] {
    bool deterministic_ops = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                               /*default_val=*/false,
                                               &deterministic_ops));
    bool cudnn_deterministic = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_CUDNN_DETERMINISTIC",
                                               /*default_val=*/false,
                                               &cudnn_deterministic));
    return deterministic_ops || cudnn_deterministic;
  }();
  return require_cudnn_determinism ||
         config.debug_options().xla_gpu_deterministic_ops();
}

StatusOr<AutotuneResult> PickBestResult(
    absl::Span<AutotuneResult const> profile_results,
    const HloInstruction& instr) {
  std::vector<AutotuneResult> filtered_results;

  // For now, we ignore WRONG_RESULT failures because false-positives are
  // possible (e.g. perhaps the reference algorithm is the one that's
  // incorrect!).  But we don't ignore REDZONE_MODIFIED failures because they're
  // quite severe and can be detected with high accuracy.
  absl::c_copy_if(
      profile_results, std::back_inserter(filtered_results),
      [](const AutotuneResult& r) {
        return !(r.has_failure() &&
                 r.failure().kind() != AutotuneResult::WRONG_RESULT);
      });

  if (filtered_results.empty()) {
    return InternalError(
        "All algorithms tried for %s failed. Falling back to "
        "default algorithm. ",
        instr.ToString());
  }

  auto selected_result = filtered_results.begin();
  if (!RequireDeterminism(instr.parent()->parent()->config())) {
    selected_result = absl::c_min_element(
        filtered_results,
        [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
          return tensorflow::proto_utils::FromDurationProto(lhs.run_time()) <
                 tensorflow::proto_utils::FromDurationProto(rhs.run_time());
        });
  }
  return *selected_result;
}

}  // namespace gpu
}  // namespace xla
