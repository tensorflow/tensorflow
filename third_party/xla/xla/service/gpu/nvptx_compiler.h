/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_NVPTX_COMPILER_H_
#define XLA_SERVICE_GPU_NVPTX_COMPILER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/IR/Module.h"
#include "xla/autotune_results.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

void WarnIfBadDriverJITVersion();

// NVPTXCompiler generates efficient GPU executables for NVPTX target.
class NVPTXCompiler : public GpuCompiler {
 public:
  NVPTXCompiler();

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::GpuComputeCapability gpu_version,
      se::dnn::VersionInfo dnn_version,
      se::DeviceMemoryAllocator* device_allocator) override;

  absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const TargetConfig& gpu_target_config,
      tsl::thread::ThreadPool* thread_pool) override;

  bool RequiresCollectiveScheduleLinearizer(
      const HloModule* module, se::StreamExecutor* stream_exec) override;

  absl::Status AddConvAndGemmAutotuningPasses(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      AutotuneConfig& autotune_config,
      tsl::thread::ThreadPool* thread_pool) override;

  absl::Status AddTritonGemmAutotuningPasses(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      AutotuneConfig& autotune_config,
      tsl::thread::ThreadPool* thread_pool) override;

  absl::Status AddCustomKernelReplacementPasses(
      HloPassPipeline* pipeline, const DebugOptions& debug_options) override;

  HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() const override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      se::GpuComputeCapability gpu_version, bool relocatable,
      const HloModule* debug_module, const CompileOptions& options) override;

  enum class LinkingMethod {
    kNone,
    kNvLink,
    kDriver,
  };

 private:
  absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& module_config) override;

  absl::StatusOr<std::vector<uint8_t>> LinkModules(
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) override;

  absl::Mutex mutex_;

  absl::flat_hash_map<std::string, LinkingMethod> linking_methods_
      ABSL_GUARDED_BY(mutex_);

  absl::StatusOr<LinkingMethod> ChooseLinkingMethod(
      const DebugOptions& debug_options);

  // Tries to compile the given ptx string to cubin.  Returns a vector with the
  // compiled cubin if compilation succeeded.
  absl::StatusOr<std::vector<uint8_t>> CompileGpuAsmOrGetCachedResult(
      const std::string& ptx, se::CudaComputeCapability cc,
      const HloModuleConfig& hlo_module_config, absl::string_view module_name,
      bool relocatable, const CompileOptions& options);

  struct CompilationCacheFlags {
    template <typename H>
    friend H AbslHashValue(H h, const CompilationCacheFlags& flags) {
      return H::combine(std::move(h),
                        flags.filter_kernels_spilling_registers_on_autotuning);
    }

    friend bool operator==(const CompilationCacheFlags& a,
                           const CompilationCacheFlags& b) {
      return a.filter_kernels_spilling_registers_on_autotuning ==
             b.filter_kernels_spilling_registers_on_autotuning;
    }

    bool filter_kernels_spilling_registers_on_autotuning;
  };

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
                        bool relocatable, CompilationCacheFlags flags)
        : ptx(std::move(ptx)),
          cc_major(cc_major),
          cc_minor(cc_minor),
          relocatable(relocatable),
          flags(std::move(flags)) {}

    template <typename H>
    friend H AbslHashValue(H h, const CompilationCacheKey& key) {
      return H::combine(std::move(h), key.ptx, key.cc_major, key.cc_minor,
                        key.relocatable, key.flags);
    }

    friend bool operator==(const CompilationCacheKey& a,
                           const CompilationCacheKey& b) {
      return a.cc_major == b.cc_major && a.cc_minor == b.cc_minor &&
             a.ptx == b.ptx && a.relocatable == b.relocatable &&
             a.flags == b.flags;
    }

    std::string ptx;
    int cc_major;
    int cc_minor;
    bool relocatable;
    CompilationCacheFlags flags;
  };

  struct CompilationCacheValue {
    bool compilation_done = false;
    absl::StatusOr<std::vector<uint8_t>> maybe_cubin;
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

#endif  // XLA_SERVICE_GPU_NVPTX_COMPILER_H_
