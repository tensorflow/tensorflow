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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {
namespace gpu {

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler();
  ~GpuCompiler() override {}

  // Bring in
  // StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
  //     std::vector<std::unique_ptr<HloModule>> modules,
  //     std::vector<std::vector<se::StreamExecutor*>>
  //        stream_execs)
  using LLVMCompiler::Compile;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> module,
                     AotCompilationOptions const& options) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    // Capture just the pointer size, not the entire GpuCompiler object.
    int64 pointer_size = pointer_size_;
    return [pointer_size](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape, pointer_size);
    };
  }

  // The triple that represents our target.
  static const char* kTargetTriple;

  // The data layout of the emitted module. Copied from computeDataLayout in
  // NVPTXTargetMachine.cpp.
  static const char* kDataLayout;

 private:
  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64 pointer_size_;

  tensorflow::mutex mutex_;

  // When compiling an HLO module, we need to find a path to the nvvm libdevice
  // files.  We search in the module's config.debug_options().cuda_data_dir()
  // and in tensorflow::LibdeviceRoot(), the latter of which is a constant.
  //
  // We cache the cuda_data_dir() and the result of our search, so that if the
  // next module we have to compile has the same cuda_data_dir(), we can skip
  // the search.
  string cached_cuda_data_dir_ GUARDED_BY(mutex_);
  string cached_libdevice_dir_ GUARDED_BY(mutex_);

  // Tries to compile the given ptx string to cubin.  Returns a vector with the
  // compiled cubin.  If compilation was unsuccessful, returns an empty vector.
  std::vector<uint8> CompilePtxOrGetCachedResult(const string& ptx,
                                                 int cc_major, int cc_minor);

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
    CompilationCacheKey(std::string ptx, int cc_major, int cc_minor)
        : ptx(std::move(ptx)), cc_major(cc_major), cc_minor(cc_minor) {}
    string ptx;
    int cc_major;
    int cc_minor;
  };
  struct CompilationCacheHash {
    size_t operator()(const CompilationCacheKey& key) const {
      return tensorflow::Hash64Combine(
          tensorflow::Hash64Combine(tensorflow::Hash64(key.ptx), key.cc_major),
          key.cc_minor);
    }
  };
  struct CompilationCacheEq {
    size_t operator()(const CompilationCacheKey& a,
                      const CompilationCacheKey& b) const {
      return a.cc_major == b.cc_major && a.cc_minor == b.cc_minor &&
             a.ptx == b.ptx;
    }
  };
  struct CompilationCacheValue {
    bool compilation_done = false;
    std::vector<uint8> cubin_data;
    // mutex and condition variable to serialize compilation completing.
    tensorflow::mutex mutex_;
    tensorflow::condition_variable compilation_done_cv_;
  };

  // Don't even think about switching this to FlatMap; iterator stability is
  // critical here.
  std::unordered_map<CompilationCacheKey, CompilationCacheValue,
                     CompilationCacheHash, CompilationCacheEq>
      compilation_cache_ GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(GpuCompiler);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
