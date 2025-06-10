/* Copyright 2015 The OpenXLA Authors.

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

// Kernel-loader specs are structures that describe how to load a data-parallel
// kernel on a given platform for subsequent launching. Headers that instantiate
// these data structures will typically be auto-generated. However, users can
// also instantiate them by hand.
//
// A kernel with the same exact functionality and type signature may be
// implemented on several different platforms. Typical usage is to create a
// singleton that describes how to load a kernel on the various supported
// platforms:
//
//  static const MultiKernelLoaderSpec &SaxpySpec() {
//    static auto *mkls =
//        (new MultiKernelLoaderSpec{4 /* = arity */})
//            ->AddCudaPtxInMemory(ptx_bytes, ptx_kernel_name);
//    };
//
//    return *mkls;
//  }
//
// This lazily instantiates an object that describes how to load CUDA PTX
// present on disk that implements saxpy for the CUDA platform. The
// CudaPtxInMemory object is a subtype of KernelLoaderSpec -- KernelLoaderSpec
// describes how to load a kernel for subsequent launching on a single platform.
//
// For the loader functionality that accepts these KernelLoaderSpecs in order
// to grab the kernel appropriately, see StreamExecutor::GetKernel().

#ifndef XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_
#define XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace stream_executor {

class Kernel;                     // defined in kernel.h
class KernelArgs;                 // defined in kernel.h
class KernelArgsPackedArrayBase;  // defined in kernel.h

// Describes how to load a kernel on a target platform.
//
// This is an abstract base class, subclassed for specific platforms.
// The filename_or_text field represents the program location (i.e. PTX or
// OpenCL loadable translation unit path) and is simply stored; whether it is a
// filename or text is exposed via more specifically named accessors in
// subclasses.
//
// These kernel loader specifications are typically auto-generated into header
// files at build time, but can also be specified manually.
class KernelLoaderSpec {
 public:
  virtual ~KernelLoaderSpec() = default;

  // Returns the kernel name to load out of the program.
  const std::string &kernel_name() const { return kernel_name_; }

 protected:
  explicit KernelLoaderSpec(absl::string_view kernel_name);

 private:
  // The kernel name that should be loaded out of the program description given
  // above.
  std::string kernel_name_;

  KernelLoaderSpec(const KernelLoaderSpec &) = delete;
  void operator=(const KernelLoaderSpec &) = delete;
};

// Loads kernel from in process symbol pointer (e.g. pointer to C++ device
// function).
class InProcessSymbol : public KernelLoaderSpec {
 public:
  InProcessSymbol(void *symbol, std::string kernel_name);

  void *symbol() const { return symbol_; }

 private:
  void *symbol_;
};

// Kernel loader specification for PTX text that resides in memory.
class CudaPtxInMemory : public KernelLoaderSpec {
 public:
  // Components: compute capability major number, compute capability minor
  // number, and PTX source.
  typedef std::tuple<int, int, absl::string_view> PtxSpec;

  // Single-PTX constructor. Adds the provided PTX version with an unknown
  // compute capability. Since the CC is unknown, the PTX is assumed to be very
  // generally usable - in other words, PTX specified in this manner is VERY
  // likely to be used as the default! Note that the PTX can be compressed,
  // which is indicated by the argument ptx_compressed.
  //
  // Warning: the string backing the provided absl::string_view ptx must outlive
  // this instance.
  CudaPtxInMemory(absl::string_view ptx, absl::string_view kernel_name);

  // Multiple-PTX-version constructor. Adds each item in spec_list to this
  // object. Note that the PTX can be compressed, which is indicated by the
  // argument ptx_compressed.
  CudaPtxInMemory(const std::initializer_list<PtxSpec> &spec_list,
                  absl::string_view kernel_name);

  // Add the PTX implementation described by ptx_spec to this object. On
  // collision (i.e., if a version with the same compute_capability already
  // exists), the existing implementation will be overwritten.
  void AddSpec(PtxSpec ptx_spec);

  // Returns pointer to the ptx of available implementation with the
  // lowest-valued compute capability. For example, if PTX written to CC2.0,
  // 3.0, and 3.5 are all available, the version for CC2.0 will be set. Returns
  // nullptr on failed lookup (if any version is not available).
  const char *default_text() const;

  // Returns pointer to the ptx for the requested compute capability.
  // Returns nullptr on failed lookup (if the requested version is not
  // available).
  const char *text(int compute_capability_major,
                   int compute_capability_minor) const;

 private:
  // PTX translation unit text contents in memory. The key is of as a tuple
  // "<cc_major>,<cc_minor>", i.e., "2,0", "3,0", "3,5". Because CC's
  // represented in this way have a clear sorting order, map::begin() will give
  // the lowest-numbered version available, i.e. the default.
  std::map<std::tuple<int, int>, const char *> ptx_by_compute_capability_;

  // Defines the minimum compute capability possible. Used when PTX has no
  // compute capability specified (in the single-PTX constructor).
  static const std::tuple<int, int> kMinimumCapability;

  CudaPtxInMemory(const CudaPtxInMemory &) = delete;
  void operator=(const CudaPtxInMemory &) = delete;
};

// Kernel loader specification for a CUBIN blob that resides in memory.
class CudaCubinInMemory : public KernelLoaderSpec {
 public:
  CudaCubinInMemory(absl::Span<const uint8_t> cubin_bytes,
                    absl::string_view kernel_name);

  absl::Span<const uint8_t> cubin_bytes() const { return cubin_bytes_; }

 private:
  absl::Span<const uint8_t> cubin_bytes_;

  CudaCubinInMemory(const CudaCubinInMemory &) = delete;
  void operator=(const CudaCubinInMemory &) = delete;
};

// Describes how to load a kernel on any subset of a number of target platforms.
class MultiKernelLoaderSpec {
 public:
  // A function for converting kernel arguments into a packed kernels arguments
  // that can be directly passed to a device kernel. This indirection allows
  // registering custom CUDA C++ kernels with non-trivial C++ API with a
  // StreamExecutor as a generic `Kernel`.
  using KernelArgsPacking =
      std::function<absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>(
          const Kernel &kernel, const KernelArgs &args)>;

  explicit MultiKernelLoaderSpec(
      size_t arity, KernelArgsPacking kernel_args_packing = nullptr);

  // Returns the number of arguments that this kernel accepts.
  size_t arity() const { return arity_; }

  // Convenience getters for testing whether these platform variants have
  // kernel loader specifications available.
  bool has_in_process_symbol() const { return in_process_symbol_ != nullptr; }
  bool has_cuda_cubin_in_memory() const {
    return cuda_cubin_in_memory_ != nullptr;
  }
  bool has_cuda_ptx_in_memory() const { return cuda_ptx_in_memory_ != nullptr; }

  // Accessors for platform variant kernel load specifications.
  // Precondition: corresponding has_* is true.
  const InProcessSymbol &in_process_symbol() const {
    CHECK(has_in_process_symbol());
    return *in_process_symbol_;
  }
  const CudaCubinInMemory &cuda_cubin_in_memory() const {
    CHECK(has_cuda_cubin_in_memory());
    return *cuda_cubin_in_memory_;
  }
  const CudaPtxInMemory &cuda_ptx_in_memory() const {
    CHECK(has_cuda_ptx_in_memory());
    return *cuda_ptx_in_memory_;
  }
  // Builder-pattern-like methods for use in initializing a
  // MultiKernelLoaderSpec. Each of these should be used at most once for a
  // single MultiKernelLoaderSpec object. See file comment for example usage.
  //
  // Note that the kernel_name parameter must be consistent with the kernel in
  // the PTX being loaded. Also be aware that in CUDA C++ the kernel name may be
  // mangled by the compiler if it is not declared in an extern "C" scope.
  MultiKernelLoaderSpec *AddInProcessSymbol(void *symbol,
                                            absl::string_view kernel_name);
  MultiKernelLoaderSpec *AddCudaCubinInMemory(
      absl::Span<const uint8_t> cubin_bytes, absl::string_view kernel_name);
  MultiKernelLoaderSpec *AddCudaPtxInMemory(absl::string_view ptx,
                                            absl::string_view kernel_name);

  void set_kernel_args_packing(KernelArgsPacking kernel_args_packing) {
    kernel_args_packing_ = std::move(kernel_args_packing);
  }

  const KernelArgsPacking &kernel_args_packing() const {
    return kernel_args_packing_;
  }

 private:
  std::shared_ptr<InProcessSymbol>
      in_process_symbol_;  // In process symbol pointer.
  std::shared_ptr<CudaCubinInMemory>
      cuda_cubin_in_memory_;  // Binary CUDA program in memory.
  std::shared_ptr<CudaPtxInMemory>
      cuda_ptx_in_memory_;  // PTX text that resides in memory.

  // Number of parameters that the kernel takes. (This is nicer to have in a
  // constexpr than having to determine it from the types via template
  // metaprogramming).
  size_t arity_;

  // Custom kernel arguments packing.
  KernelArgsPacking kernel_args_packing_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_
