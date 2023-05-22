/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_COMPILER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

void WarnIfBadDriverJITVersion();

// NVPTXCompiler generates efficient GPU executables for NVPTX target.
class NVPTXCompiler : public GpuCompiler {
 public:
  NVPTXCompiler();

  Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, GpuVersion gpu_version,
      se::dnn::VersionInfo dnn_version,
      se::DeviceMemoryAllocator* device_allocator) override;

  Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
      const AutotuneResults* autotune_results) override;

  HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() override;

  GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) override;

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      GpuVersion gpu_version, bool relocatable,
      const HloModule* debug_module) override;

 private:
  StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& module_config) override;

  StatusOr<std::vector<uint8_t>> LinkModules(
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) override;

  absl::Mutex mutex_;

  // When compiling an HLO module, we need to find a path to the nvvm libdevice
  // files.  We search in the module's config.debug_options().cuda_data_dir()
  // and in tensorflow::LibdeviceRoot(), the latter of which is a constant.
  //
  // We cache the cuda_data_dir() and the result of our search, so that if the
  // next module we have to compile has the same cuda_data_dir(), we can skip
  // the search.
  std::string cached_cuda_data_dir_ ABSL_GUARDED_BY(mutex_);
  std::string cached_libdevice_dir_ ABSL_GUARDED_BY(mutex_);

  // Tries to compile the given ptx string to cubin.  Returns a vector with the
  // compiled cubin.  If compilation was unsuccessful, returns an empty vector.
  std::vector<uint8_t> CompileGpuAsmOrGetCachedResult(
      const std::string& ptx, se::CudaComputeCapability cc,
      const HloModuleConfig& hlo_module_config, absl::string_view module_name,
      bool relocatable);

  // The compilation_cache_ map is a cache from {ptx string, cc_major, cc_minor}
  // -> cubin so we don't recompile the same ptx twice.  This is important for
  // some interactive workflows.  (We also cache at the HLO level, but sometimes
  // we can't realize that two modules are the same until we lower to ptx.)
  //
  // Compilation of distinct PTX happens in parallel. If more than one thread
  // attempts to compile the same PTX, the fist thread to obtain
  // cache_value_->mutex_ performs the compilation. The rest wait() on
  // cache_value_->compilation_done_cv_ until the compilation is done.
  //
  // If compiling the ptx fails, we return an empty cubin, cross our fingers,
  // and leave compilation up to the driver.
  struct CompilationCacheKey {
    CompilationCacheKey(std::string ptx, int cc_major, int cc_minor,
                        bool relocatable)
        : ptx(std::move(ptx)),
          cc_major(cc_major),
          cc_minor(cc_minor),
          relocatable(relocatable) {}
    template <typename H>
    friend H AbslHashValue(H h, const CompilationCacheKey& key) {
      return H::combine(std::move(h), key.ptx, key.cc_major, key.cc_minor,
                        key.relocatable);
    }
    friend bool operator==(const CompilationCacheKey& a,
                           const CompilationCacheKey& b) {
      return a.cc_major == b.cc_major && a.cc_minor == b.cc_minor &&
             a.ptx == b.ptx && a.relocatable == b.relocatable;
    }
    std::string ptx;
    int cc_major;
    int cc_minor;
    bool relocatable;
  };
  struct CompilationCacheValue {
    bool compilation_done = false;
    std::vector<uint8_t> cubin_data;
    // mutex and condition variable to serialize compilation completing.
    absl::Mutex mutex;
    absl::CondVar compilation_done_cv;
  };

  // Don't even think about switching this to flat_hash_map; iterator stability
  // is critical here.
  absl::node_hash_map<CompilationCacheKey, CompilationCacheValue>
      compilation_cache_ ABSL_GUARDED_BY(mutex_);

  NVPTXCompiler(const NVPTXCompiler&) = delete;
  NVPTXCompiler& operator=(const NVPTXCompiler&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_COMPILER_H_
