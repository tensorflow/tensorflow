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

// The compiler API is used by the XLA service to generate executables that
// run on a given platform. This is a registry and abstract interface, for
// pluggability by the various platforms.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILER_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace xla {

// The following types are used for ahead of time compilation.

// Contains the object file data created as a result of ahead-of-time
// compuation.
using ObjectFileData = std::vector<char>;

// Contains the buffer sizes information needed to allocate buffers to execute
// an ahead-of-time computation.  Entries which contain -1 designate a parameter
// which should be skipped over during allocation.
using BufferSizes = std::vector<int64>;

// Abstract superclass describing the result of an ahead-of-time compilation.
class AotCompilationResult {
 public:
  AotCompilationResult(const AotCompilationResult&) = delete;
  AotCompilationResult& operator=(AotCompilationResult const&) = delete;

  virtual ~AotCompilationResult() = default;

 protected:
  AotCompilationResult() = default;
};

// Abstract superclass describing options to an ahead-of-time compilation.
class AotCompilationOptions {
 public:
  AotCompilationOptions(const AotCompilationOptions&) = delete;
  AotCompilationOptions& operator=(AotCompilationOptions const&) = delete;

  virtual ~AotCompilationOptions() = default;

  // Returns the ID of the platform to which these options apply.
  virtual perftools::gputools::Platform::Id PlatformId() const = 0;

 protected:
  AotCompilationOptions() = default;
};

// Abstract compiler interface that is subclassed for compilation on a
// particular platform.
//
// The compiler ties together high level optimization (HLO) and low level
// optimization (LLO) / codegen (CG) to generate efficient executables for the
// target platform.
//
// The platform-based compiler singletons are registered via module initializers
// in their corresponding XLA compiler libraries, and are registered via the
// RegisterCompilerFactory API below.
//
// Thread-safety: subclasses of Compiler must be thread-safe, as multiple
// XLA clients may be requesting compilation concurrently for a given
// platform.
class Compiler {
 public:
  virtual ~Compiler() {}

  // Returns the ID of the platform that this compiler targets.
  virtual perftools::gputools::Platform::Id PlatformId() const = 0;

  // Compiles the HLO module for execution on a device given by the executor,
  // and returns an executable object or an error status. Takes ownership of the
  // HLO module and is free to transform it.
  //
  // The compiler may optionally specialize to the individual device
  // (not just type of device) indicated by the executor.
  //
  // Use the overload below to compile computations that run in parallel.
  virtual StatusOr<std::unique_ptr<Executable>> Compile(
      std::unique_ptr<HloModule> module,
      perftools::gputools::StreamExecutor* executor) = 0;

  // Compiles a set of HLO modules that can run in parallel, potentially
  // communicating data between the modules, and returns a corresponding
  // sequence of executable objects.
  virtual StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::vector<std::unique_ptr<HloModule>> modules,
      std::vector<perftools::gputools::StreamExecutor*> stream_exec) = 0;

  // Compiles the HLO module for ahead-of-time execution.  This is intended for
  // use in static compilation.
  virtual StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> modules,
                     const AotCompilationOptions& options) = 0;

  /////
  // The Compiler class also serves as a point to register compiler objects
  // for the various platforms.

  using CompilerFactory = std::function<std::unique_ptr<Compiler>()>;

  // Registers the compiler singleton for the platform. This is assumed to
  // be a singleton, so no ownership is transferred.
  //
  // Precondition: a platform kind must not be registered more than once.
  static void RegisterCompilerFactory(
      perftools::gputools::Platform::Id platform_id,
      CompilerFactory compiler_factory);

  // Returns the compiler singleton pointer if it is available for the given
  // platform, or an error status if it is not.
  static StatusOr<Compiler*> GetForPlatform(
      const perftools::gputools::Platform* platform);

  // Returns a function that computes the size in bytes of the logical
  // buffer that contains a shape.
  virtual HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const = 0;

  // Returns a function that computes the size in bytes of a given
  // logical buffer.
  std::function<int64(const LogicalBuffer&)> BufferSizeBytesFunction() {
    HloCostAnalysis::ShapeSizeFunction shape_size = ShapeSizeBytesFunction();
    return [shape_size](const LogicalBuffer& buffer) {
      return shape_size(buffer.shape());
    };
  }

 private:
  // Mutex that guards the platform-compiler map.
  static tensorflow::mutex* platform_compiler_mutex_;
  static void LazyInitMutex();

  // Map from platform kind to compiler factory.
  static std::map<perftools::gputools::Platform::Id, CompilerFactory>*
  GetPlatformCompilerFactories();

  // Map from platform kind to compiler instance, if we made one already (based
  // on the factories above).
  static std::map<perftools::gputools::Platform::Id, std::unique_ptr<Compiler>>*
  GetPlatformCompilers();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILER_H_
