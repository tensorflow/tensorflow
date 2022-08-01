/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
//            ->AddCudaPtxOnDisk(ptx_file_path, ptx_kernelname)
//            ->AddOpenCLTextOnDisk(opencl_text_file_path, ocl_kernelname);
//    };
//
//    return *mkls;
//  }
//
// This lazily instantiates an object that describes how to load CUDA PTX
// present on disk that implements saxpy for the for the CUDA platform, or
// OpenCL text present on disk that implements saxpy for an OpenCL-based
// platform. The CudaPtxOnDisk and OpenCLTextOnDisk objects are subtypes of
// KernelLoaderSpec -- KernelLoaderSpec describes how to load a kernel for
// subsequent launching on a single platform.
//
// For the loader functionality that accepts these KernelLoaderSpecs in order
// to grab the kernel appropriately, see StreamExecutor::GetKernel().

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_

#include <stddef.h>

#include <map>
#include <memory>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

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
  virtual ~KernelLoaderSpec() {}

  // Returns the kernel name to load out of the program.
  const std::string &kernelname() const { return kernelname_; }

 protected:
  explicit KernelLoaderSpec(absl::string_view kernelname);

 private:
  // The kernel name that should be loaded out of the program description given
  // above.
  std::string kernelname_;

  SE_DISALLOW_COPY_AND_ASSIGN(KernelLoaderSpec);
};

// An abstract kernel loader spec that has an associated file path, where
// there's a canonical suffix for the filename; e.g. see CudaPtxOnDisk whose
// canonical filename suffix is ".ptx".
class OnDiskKernelLoaderSpec : public KernelLoaderSpec {
 public:
  ~OnDiskKernelLoaderSpec() override {}

  // Returns the path to the on-disk loadable kernel file.
  const std::string &filename() const { return filename_; }

  // Returns the canonical suffix for this on-disk kernel loader spec format;
  // e.g. PTX files on disk have a canonical suffix of ".ptx".
  virtual const char *CanonicalSuffix() const = 0;

 protected:
  OnDiskKernelLoaderSpec(absl::string_view filename,
                         absl::string_view kernelname);

  std::string filename_;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(OnDiskKernelLoaderSpec);
};

// Kernel loader specification for PTX text that resides on disk.
class CudaPtxOnDisk : public OnDiskKernelLoaderSpec {
 public:
  CudaPtxOnDisk(absl::string_view filename, absl::string_view kernelname);
  ~CudaPtxOnDisk() override {}

  const char *CanonicalSuffix() const override { return ".ptx"; }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(CudaPtxOnDisk);
};

// Kernel loader specification for CUBIN binary that resides on disk.
class CudaCubinOnDisk : public OnDiskKernelLoaderSpec {
 public:
  CudaCubinOnDisk(absl::string_view filename, absl::string_view kernelname);
  ~CudaCubinOnDisk() override {}

  const std::string &filename() const { return filename_; }

  const char *CanonicalSuffix() const override { return ".cubin"; }

 private:
  std::string filename_;

  SE_DISALLOW_COPY_AND_ASSIGN(CudaCubinOnDisk);
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
  CudaPtxInMemory(absl::string_view ptx, absl::string_view kernelname,
                  bool ptx_compressed = false);

  // Multiple-PTX-version constructor. Adds each item in spec_list to this
  // object. Note that the PTX can be compressed, which is indicated by the
  // argument ptx_compressed.
  CudaPtxInMemory(const std::initializer_list<PtxSpec> &spec_list,
                  absl::string_view kernel_name, bool ptx_compressed = false);
  ~CudaPtxInMemory() override {}

  // Add the PTX implementation described by ptx_spec to this object. On
  // collision (i.e., if a version with the same compute_capability already
  // exists), the existing implementation will be overwritten.
  void AddSpec(PtxSpec ptx_spec);

  // Returns pointer to the ptx of available implementation with the
  // lowest-valued compute capability. For example, if PTX written to CC2.0,
  // 3.0, and 3.5 are all available, the version for CC2.0 will be set. Returns
  // nullptr on failed lookup (if any version is not available).
  // When the ptx is compressed, returns the decompressed ptx.
  const char *default_text() const;

  // Similar to default_text().
  // When the ptx is compressed, returns the decompressed ptx.
  const char *original_default_text() const;

  // Returns pointer to the ptx for the requested compute capability.
  // Returns nullptr on failed lookup (if the requested version is not
  // available).
  // When the ptx is compressed, returns the decompressed ptx.
  const char *text(int compute_capability_major,
                   int compute_capability_minor) const;

  // Similar to text().
  // When the ptx is compressed, returns the original compressed ptx.
  const char *original_text(int compute_capability_major,
                            int compute_capability_minor) const;

  // Decompresses the PTX string using bzip2.
  static std::string DecompressPtx(const char *ptx);

 private:
  // PTX translation unit text contents in memory. The key is of as a tuple
  // "<cc_major>,<cc_minor>", i.e., "2,0", "3,0", "3,5". Because CC's
  // represented in this way have a clear sorting order, map::begin() will give
  // the lowest-numbered version available, i.e. the default.
  std::map<std::tuple<int, int>, const char *,
           bool (*)(const std::tuple<int, int> &, const std::tuple<int, int> &)>
      ptx_by_compute_capability_;

  // Stores all decompressed ptx strings, with original ptx string as keys.
  // It is marked as mutable for lazy decompression.
  mutable std::map<const char *, std::string> decompressed_ptx_;
  mutable absl::Mutex mu_;

  // Defines the minimum compute capability possible. Used when PTX has no
  // compute capability specified (in the single-PTX constructor).
  static const std::tuple<int, int> kMinimumCapability;

  SE_DISALLOW_COPY_AND_ASSIGN(CudaPtxInMemory);
};

// Kernel loader specification for OpenCL text that resides on disk.
class OpenCLTextOnDisk : public OnDiskKernelLoaderSpec {
 public:
  OpenCLTextOnDisk(absl::string_view filename, absl::string_view kernelname);
  ~OpenCLTextOnDisk() override {}

  const char *CanonicalSuffix() const override { return ".ocl"; }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(OpenCLTextOnDisk);
};

// Kernel loader specification for OpenCL binary that resides on disk.
class OpenCLBinaryOnDisk : public OnDiskKernelLoaderSpec {
 public:
  OpenCLBinaryOnDisk(absl::string_view filename, absl::string_view kernelname);
  ~OpenCLBinaryOnDisk() override {}

  const char *CanonicalSuffix() const override { return ".aocx"; }

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(OpenCLBinaryOnDisk);
};

// Kernel loader specification for OpenCL text that resides in memory.
class OpenCLTextInMemory : public KernelLoaderSpec {
 public:
  OpenCLTextInMemory(absl::string_view text, absl::string_view kernelname);
  ~OpenCLTextInMemory() override {}

  // Returns the OpenCL text contents.
  const std::string &text() const { return text_; }

 private:
  // OpenCL translation unit text contents in memory.
  std::string text_;

  SE_DISALLOW_COPY_AND_ASSIGN(OpenCLTextInMemory);
};

// Kernel loader specification for a CUBIN blob that resides in memory.
class CudaCubinInMemory : public KernelLoaderSpec {
 public:
  CudaCubinInMemory(const char *bytes, absl::string_view kernelname);
  ~CudaCubinInMemory() override {}

  const char *bytes() const { return bytes_; }

 private:
  const char *bytes_;

  SE_DISALLOW_COPY_AND_ASSIGN(CudaCubinInMemory);
};

// Describes how to load a kernel on any subset of a number of target platforms.
class MultiKernelLoaderSpec {
 public:
  explicit MultiKernelLoaderSpec(size_t arity);

  // Returns the number of arguments that this kernel accepts.
  size_t arity() const { return arity_; }

  // Convenience getters for testing whether these platform variants have
  // kernel loader specifications available.
  bool has_cuda_ptx_on_disk() const { return cuda_ptx_on_disk_ != nullptr; }
  bool has_cuda_cubin_on_disk() const { return cuda_cubin_on_disk_ != nullptr; }
  bool has_cuda_cubin_in_memory() const {
    return cuda_cubin_in_memory_ != nullptr;
  }
  bool has_cuda_ptx_in_memory() const { return cuda_ptx_in_memory_ != nullptr; }
  bool has_ocl_text_on_disk() const { return ocl_text_on_disk_ != nullptr; }
  bool has_ocl_binary_on_disk() const { return ocl_binary_on_disk_ != nullptr; }
  bool has_ocl_text_in_memory() const { return ocl_text_in_memory_ != nullptr; }

  // Accessors for platform variant kernel load specifications.
  // Precondition: corresponding has_* is true.
  const CudaPtxOnDisk &cuda_ptx_on_disk() const {
    CHECK(has_cuda_ptx_on_disk());
    return *cuda_ptx_on_disk_;
  }
  const CudaCubinOnDisk &cuda_cubin_on_disk() const {
    CHECK(has_cuda_cubin_on_disk());
    return *cuda_cubin_on_disk_;
  }
  const CudaCubinInMemory &cuda_cubin_in_memory() const {
    CHECK(has_cuda_cubin_in_memory());
    return *cuda_cubin_in_memory_;
  }
  const CudaPtxInMemory &cuda_ptx_in_memory() const {
    CHECK(has_cuda_ptx_in_memory());
    return *cuda_ptx_in_memory_;
  }
  const OpenCLTextOnDisk &ocl_text_on_disk() const {
    CHECK(has_ocl_text_on_disk());
    return *ocl_text_on_disk_;
  }
  const OpenCLBinaryOnDisk &ocl_binary_on_disk() const {
    CHECK(has_ocl_binary_on_disk());
    return *ocl_binary_on_disk_;
  }
  const OpenCLTextInMemory &ocl_text_in_memory() const {
    CHECK(has_ocl_text_in_memory());
    return *ocl_text_in_memory_;
  }

  // Builder-pattern-like methods for use in initializing a
  // MultiKernelLoaderSpec. Each of these should be used at most once for a
  // single MultiKernelLoaderSpec object. See file comment for example usage.
  //
  // Note that the kernelname parameter must be consistent with the kernel in
  // the PTX or OpenCL being loaded. Also be aware that in CUDA C++ the kernel
  // name may be mangled by the compiler if it is not declared in an
  // extern "C" scope.
  MultiKernelLoaderSpec *AddOpenCLTextOnDisk(absl::string_view filename,
                                             absl::string_view kernelname);
  MultiKernelLoaderSpec *AddOpenCLBinaryOnDisk(absl::string_view filename,
                                               absl::string_view kernelname);
  MultiKernelLoaderSpec *AddOpenCLTextInMemory(absl::string_view ocl_text,
                                               absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaPtxOnDisk(absl::string_view filename,
                                          absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaCubinOnDisk(absl::string_view filename,
                                            absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaCubinInMemory(const char *cubin_bytes,
                                              absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaPtxInMemory(absl::string_view ptx,
                                            absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaCompressedPtxInMemory(
      absl::string_view ptx, absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaPtxInMemory(
      std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
      absl::string_view kernelname);
  MultiKernelLoaderSpec *AddCudaCompressedPtxInMemory(
      std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
      absl::string_view kernelname);

 private:
  std::unique_ptr<CudaPtxOnDisk>
      cuda_ptx_on_disk_;  // PTX text that resides in a file.
  std::unique_ptr<CudaCubinOnDisk>
      cuda_cubin_on_disk_;  // Binary CUDA program in a file.
  std::unique_ptr<CudaCubinInMemory>
      cuda_cubin_in_memory_;  // Binary CUDA program in memory.
  std::unique_ptr<CudaPtxInMemory>
      cuda_ptx_in_memory_;  // PTX text that resides in memory.
  std::unique_ptr<OpenCLTextOnDisk>
      ocl_text_on_disk_;  // OpenCL text that resides on disk.
  std::unique_ptr<OpenCLBinaryOnDisk>
      ocl_binary_on_disk_;  // OpenCL binary that resides on disk.
  std::unique_ptr<OpenCLTextInMemory>
      ocl_text_in_memory_;  // OpenCL text that resides in memory.

  // Number of parameters that the kernel takes. (This is nicer to have in a
  // constexpr than having to determine it from the types via template
  // metaprogramming).
  size_t arity_;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_
