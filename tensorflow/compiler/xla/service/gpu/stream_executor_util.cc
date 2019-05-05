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

// Prints a warning if the ptxas at ptxas_path has known bugs.
//
// Only prints a warning the first time it's called for a particular value of
// ptxas_path.
//
// Locks on entry.
void WarnIfBadPtxasVersion(const string& ptxas_path) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static std::unordered_set<string>* seen_ptxas_paths GUARDED_BY(mu) =
      new std::unordered_set<string>();

  tensorflow::mutex_lock lock(mu);
  if (!seen_ptxas_paths->insert(ptxas_path).second) {
    // Already checked this ptx binary, nothing to do.
    return;
  }

  tensorflow::SubProcess ptxas;
  ptxas.SetProgram(ptxas_path, {ptxas_path, "--version"});
  ptxas.SetChannelAction(tensorflow::CHAN_STDOUT, tensorflow::ACTION_PIPE);
  if (!ptxas.Start()) {
    LOG(WARNING) << "Couldn't invoke " << ptxas_path << " --version";
    return;
  }

  string out;
  int exit_code = ptxas.Communicate(/*stdin_input=*/nullptr, &out,
                                    /*stderr_output=*/nullptr);
  if (exit_code != 0) {
    LOG(WARNING) << "Running " << ptxas_path << " --version returned "
                 << exit_code;
    return;
  }

  int64 vmaj, vmin, vdot;
  string vmaj_str, vmin_str, vdot_str;
  if (!RE2::PartialMatch(out, R"(\bV(\d+)\.(\d+)\.(\d+)\b)", &vmaj_str,
                         &vmin_str, &vdot_str) ||
      !absl::SimpleAtoi(vmaj_str, &vmaj) ||
      !absl::SimpleAtoi(vmin_str, &vmin) ||
      !absl::SimpleAtoi(vdot_str, &vdot)) {
    LOG(WARNING) << "Couldn't parse ptxas version in output of " << ptxas_path
                 << " --version:\n"
                 << out;
    return;
  }

  // We need ptxas >= 9.0 as a hard requirement, because we compile targeting
  // PTX 6.0.  An older ptxas will just fail to compile any of our code.
  //
  // ptxas 9.0 before 9.0.276 and ptxas 9.1 before 9.1.121 miscompile some
  // address calculations with large offsets (e.g. "load ptr + large_constant"),
  // b/70245379.
  //
  // ptxas 9.1.121 miscompiles some large multioutput fusions, again in a way
  // that appears related to address calculations, b/111107644.  ptxas 9.2.88
  // appears to work, as far as we can tell.
  if (vmaj < 9) {
    LOG(ERROR)
        << "You are using ptxas 8.x, but XLA requires ptxas 9.x (and strongly "
           "prefers >= 9.2.88).  Compilation of XLA kernels below will likely "
           "fail.\n\nYou do not need to update CUDA; cherry-picking the ptxas "
           "binary is sufficient.";
  } else if (std::make_tuple(vmaj, vmin, vdot) < std::make_tuple(9, 2, 88)) {
    LOG(WARNING)
        << "*** WARNING *** You are using ptxas " << vmaj << "." << vmin << "."
        << vdot
        << ", which is older than 9.2.88. ptxas 9.x before 9.2.88 is known to "
           "miscompile XLA code, leading to incorrect results or "
           "invalid-address errors.\n\nYou do not need to update to CUDA "
           "9.2.88; cherry-picking the ptxas binary is sufficient.";
  }
}

// Returns a vector of potential locations of the CUDA root directory.
// Searches through tensorflow CUDA locations AND through the CUDA location
// specified in HLO configuration.
std::vector<string> GetCudaRootCandidates(
    PtxCompilationOptions compile_ptx_options) {
  std::vector<string> potential_cuda_roots = tensorflow::CandidateCudaRoots();

  // "." is our last resort, even though it probably won't work.
  potential_cuda_roots.push_back(".");

  // CUDA location explicitly specified by user via --xla_gpu_cuda_data_dir has
  // highest priority.
  string xla_gpu_cuda_data_dir = compile_ptx_options.xla_gpu_cuda_data_dir;
  if (!xla_gpu_cuda_data_dir.empty()) {
    potential_cuda_roots.insert(potential_cuda_roots.begin(),
                                xla_gpu_cuda_data_dir);
  }
  return potential_cuda_roots;
}

StatusOr<std::vector<uint8>> CompilePtx(
    se::StreamExecutor* stream_exec, absl::string_view ptx,
    PtxCompilationOptions compile_ptx_options) {
  int cc_major, cc_minor;
  if (!stream_exec->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                   &cc_minor)) {
    LOG(WARNING)
        << "Couldn't get compute capability for device; assuming sm_20.";
    cc_major = 2;
    cc_minor = 0;
  }

  tensorflow::profiler::TraceMe activity(
      "Compile PTX", tensorflow::profiler::TraceMeLevel::kInfo);
  auto env = tensorflow::Env::Default();
  string ptxas_path;
  for (const string& cuda_root : GetCudaRootCandidates(compile_ptx_options)) {
    ptxas_path = tensorflow::io::JoinPath(cuda_root, "bin", "ptxas");
    VLOG(2) << "Looking for ptxas at " << ptxas_path;
    if (env->FileExists(ptxas_path).ok()) {
      break;
    }
  }
  TF_RETURN_IF_ERROR(env->FileExists(ptxas_path));
  VLOG(2) << "Using ptxas at " << ptxas_path;

  WarnIfBadPtxasVersion(ptxas_path);

  // Write ptx into a temporary file.
  string ptx_path;
  if (!env->LocalTempFilename(&ptx_path)) {
    return InternalError("couldn't get temp PTX file name");
  }
  auto ptx_cleaner = tensorflow::gtl::MakeCleanup([&ptx_path] {
    TF_CHECK_OK(tensorflow::Env::Default()->DeleteFile(ptx_path));
  });

  TF_RETURN_IF_ERROR(tensorflow::WriteStringToFile(env, ptx_path, ptx));
  VLOG(2) << "ptx written to: " << ptx_path;

  // Invoke ptxas and collect its output.
  string cubin_path;
  if (!env->LocalTempFilename(&cubin_path)) {
    return InternalError("couldn't get temp CUBIN file name");
  }
  auto cubin_cleaner = tensorflow::gtl::MakeCleanup([&cubin_path] {
    // CUBIN file may never be created, so the failure to delete it should not
    // produce TF error.
    tensorflow::Env::Default()->DeleteFile(cubin_path).IgnoreError();
  });
  tensorflow::SubProcess ptxas_info_dumper;
  std::vector<string> ptxas_args = {
      ptxas_path, ptx_path, "-o", cubin_path,
      absl::StrCat("-arch=sm_", cc_major, cc_minor)};
  if (VLOG_IS_ON(2)) {
    ptxas_args.push_back("-v");
  }
  if (compile_ptx_options.xla_gpu_disable_ptxas_optimizations) {
    ptxas_args.push_back("-O0");
  }
  ptxas_info_dumper.SetProgram(ptxas_path, ptxas_args);
  ptxas_info_dumper.SetChannelAction(tensorflow::CHAN_STDERR,
                                     tensorflow::ACTION_PIPE);
  if (!ptxas_info_dumper.Start()) {
    return InternalError("Failed to launch ptxas");
  }
  string stderr_output;
  int exit_status = ptxas_info_dumper.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  XLA_LOG_LINES(tensorflow::INFO, stderr_output);
  if (exit_status != 0) {
    return InternalError("ptxas exited with non-zero error code %d",
                         exit_status);
  }

  // Read in the result of compilation and return it as a byte vector.
  string cubin;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  cubin_path, &cubin));
  std::vector<uint8> cubin_vector(cubin.begin(), cubin.end());
  return cubin_vector;
}

}  // namespace gpu
}  // namespace xla
