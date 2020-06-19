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

#include "tensorflow/stream_executor/gpu/asm_compiler.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace stream_executor {

#if TENSORFLOW_USE_ROCM || defined(PLATFORM_WINDOWS)

port::StatusOr<std::vector<uint8>> CompileGpuAsm(int device_ordinal,
                                                 const char* ptx_contents,
                                                 GpuAsmOpts options) {
  // TODO(b/134675935): Subprocess invocation not supported on Windows.
  return port::InternalError(
      "Invoking GPU asm compilation is supported on Cuda non-Windows "
      "platforms only");
}

port::StatusOr<absl::Span<const uint8>> CompileGpuAsmOrGetCached(
    int device_ordinal, const char* ptx, GpuAsmOpts compilation_options) {
  return CompileGpuAsm(device_ordinal, ptx, compilation_options);
}

#else

// Prints a warning if the ptxas at ptxas_path has known bugs.
//
// Only prints a warning the first time it's called for a particular value of
// ptxas_path.
//
// Locks on entry.
static void WarnIfBadPtxasVersion(const std::string& ptxas_path) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static std::unordered_set<std::string>* seen_ptxas_paths TF_GUARDED_BY(mu) =
      new std::unordered_set<std::string>();

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

  std::string out;
  int exit_code = ptxas.Communicate(/*stdin_input=*/nullptr, &out,
                                    /*stderr_output=*/nullptr);
  if (exit_code != 0) {
    LOG(WARNING) << "Running " << ptxas_path << " --version returned "
                 << exit_code;
    return;
  }

  int64 vmaj, vmin, vdot;
  std::string vmaj_str, vmin_str, vdot_str;
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
        << "You are using ptxas 8.x, but TF requires ptxas 9.x (and strongly "
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

port::StatusOr<absl::Span<const uint8>> CompileGpuAsmOrGetCached(
    int device_ordinal, const char* ptx, GpuAsmOpts compilation_options) {
  using PtxCacheKey = std::tuple<int, std::string, GpuAsmOpts::PtxOptionsTuple>;
  static tensorflow::mutex ptx_cache_mutex(tensorflow::LINKER_INITIALIZED);
  static auto& ptx_cache TF_GUARDED_BY(ptx_cache_mutex) =
      *new absl::flat_hash_map<PtxCacheKey, std::vector<uint8>>();

  tensorflow::mutex_lock lock(ptx_cache_mutex);
  PtxCacheKey cache_key{device_ordinal, std::string(ptx),
                        compilation_options.ToTuple()};
  auto it = ptx_cache.find(cache_key);
  if (it == ptx_cache.end()) {
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8> compiled,
        CompileGpuAsm(device_ordinal, ptx, compilation_options));
    it = ptx_cache.emplace(cache_key, std::move(compiled)).first;
  }

  CHECK(it != ptx_cache.end());
  const std::vector<uint8>& compiled = it->second;
  return absl::MakeSpan(compiled);
}

port::StatusOr<std::vector<uint8>> CompileGpuAsm(int device_ordinal,
                                                 const char* ptx_contents,
                                                 GpuAsmOpts options) {
  gpu::GpuDeviceHandle handle;
  TF_RETURN_IF_ERROR(gpu::GpuDriver::GetDevice(device_ordinal, &handle));
  int cc_major;
  int cc_minor;
  TF_RETURN_IF_ERROR(
      gpu::GpuDriver::GetComputeCapability(&cc_major, &cc_minor, handle));
  return CompileGpuAsm(cc_major, cc_minor, ptx_contents, options);
}

port::StatusOr<std::vector<uint8>> CompileGpuAsm(int cc_major, int cc_minor,
                                                 const char* ptx_contents,
                                                 GpuAsmOpts options) {
  std::string ptxas_path;
  auto env = tensorflow::Env::Default();
  for (const std::string& cuda_root :
       tensorflow::CandidateCudaRoots(options.preferred_cuda_dir)) {
    ptxas_path = tensorflow::io::JoinPath(cuda_root, "bin", "ptxas");
    VLOG(2) << "Looking for ptxas at " << ptxas_path;
    if (env->FileExists(ptxas_path).ok()) {
      break;
    }
  }
  if (!env->FileExists(ptxas_path).ok()) {
    // Rely on subprocess invocation to find the correct binary.
    ptxas_path = "ptxas";
  }
  VLOG(2) << "Using ptxas at " << ptxas_path;

  WarnIfBadPtxasVersion(ptxas_path);

  // Write ptx into a temporary file.
  std::string ptx_path;
  if (!env->LocalTempFilename(&ptx_path)) {
    return port::InternalError("couldn't get temp PTX file name");
  }
  TF_RETURN_IF_ERROR(
      tensorflow::WriteStringToFile(env, ptx_path, ptx_contents));
  VLOG(2) << "ptx written to: " << ptx_path;

  auto ptx_cleaner = tensorflow::gtl::MakeCleanup([&ptx_path] {
    TF_CHECK_OK(tensorflow::Env::Default()->DeleteFile(ptx_path));
  });

  // Invoke ptxas and collect its output.
  std::string cubin_path;
  if (!env->LocalTempFilename(&cubin_path)) {
    return port::InternalError("couldn't get temp CUBIN file name");
  }
  auto cubin_cleaner = tensorflow::gtl::MakeCleanup([&cubin_path] {
    // CUBIN file may never be created, so the failure to delete it should not
    // produce TF error.
    tensorflow::Env::Default()->DeleteFile(cubin_path).IgnoreError();
  });
  tensorflow::SubProcess ptxas_info_dumper;
  std::vector<std::string> ptxas_args = {
      ptxas_path, ptx_path, "-o", cubin_path,
      absl::StrCat("-arch=sm_", cc_major, cc_minor)};
  if (VLOG_IS_ON(2)) {
    ptxas_args.push_back("-v");
  }
  if (options.disable_gpuasm_optimizations) {
    ptxas_args.push_back("-O0");
  }
  ptxas_args.insert(ptxas_args.end(), options.extra_flags.begin(),
                    options.extra_flags.end());
  if (VLOG_IS_ON(3)) {
    VLOG(3) << absl::StrJoin(ptxas_args, " ");
  }

  ptxas_info_dumper.SetProgram(ptxas_path, ptxas_args);
  ptxas_info_dumper.SetChannelAction(tensorflow::CHAN_STDERR,
                                     tensorflow::ACTION_PIPE);
  if (!ptxas_info_dumper.Start()) {
    return port::InternalError("Failed to launch ptxas");
  }
  std::string stderr_output;
  int exit_status = ptxas_info_dumper.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  if (exit_status != 0) {
    return port::InternalError(
        absl::StrFormat("ptxas exited with non-zero error code %d, output: %s",
                        exit_status, stderr_output));
  }
  // Print the verbose output of ptxas.
  if (!stderr_output.empty()) {
    VLOG(2) << stderr_output;
  }

  // Read in the result of compilation and return it as a byte vector.
  std::string cubin;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  cubin_path, &cubin));
  std::vector<uint8> cubin_vector(cubin.begin(), cubin.end());
  return cubin_vector;
}

#endif

}  // namespace stream_executor
