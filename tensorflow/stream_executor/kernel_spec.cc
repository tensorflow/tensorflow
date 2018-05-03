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

#include "tensorflow/stream_executor/kernel_spec.h"

namespace stream_executor {

KernelLoaderSpec::KernelLoaderSpec(port::StringPiece kernelname)
    : kernelname_(std::string(kernelname)) {}

OnDiskKernelLoaderSpec::OnDiskKernelLoaderSpec(port::StringPiece filename,
                                               port::StringPiece kernelname)
    : KernelLoaderSpec(kernelname), filename_(std::string(filename)) {}

CudaPtxOnDisk::CudaPtxOnDisk(port::StringPiece filename,
                             port::StringPiece kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

CudaCubinOnDisk::CudaCubinOnDisk(port::StringPiece filename,
                                 port::StringPiece kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

CudaCubinInMemory::CudaCubinInMemory(const char *bytes,
                                     port::StringPiece kernelname)
    : KernelLoaderSpec(kernelname), bytes_(bytes) {}

bool CompareComputeCapability(const std::tuple<int, int> &lhs,
                              const std::tuple<int, int> &rhs) {
  return std::get<0>(lhs) < std::get<0>(rhs) ||
         (std::get<0>(lhs) == std::get<0>(rhs) &&
          std::get<1>(lhs) < std::get<1>(rhs));
}

const std::tuple<int, int> CudaPtxInMemory::kMinimumCapability{1, 0};

CudaPtxInMemory::CudaPtxInMemory(port::StringPiece ptx,
                                 port::StringPiece kernel_name,
                                 bool ptx_compressed)
    : KernelLoaderSpec(kernel_name),
      ptx_by_compute_capability_(CompareComputeCapability) {
  if (ptx_compressed) {
    // Lazy decompression. Put an empty string in decompressed_ptx_ showing that
    // the original ptx is compressed.
    decompressed_ptx_[ptx.data()] = "";
  }
  ptx_by_compute_capability_[kMinimumCapability] = ptx.data();
}

CudaPtxInMemory::CudaPtxInMemory(
    const std::initializer_list<CudaPtxInMemory::PtxSpec> &spec_list,
    port::StringPiece kernel_name, bool ptx_compressed)
    : KernelLoaderSpec(kernel_name),
      ptx_by_compute_capability_(CompareComputeCapability) {
  for (const auto &spec : spec_list) {
    int major, minor;
    port::StringPiece ptx;
    std::tie(major, minor, ptx) = spec;
    if (ptx_compressed) {
      // Lazy decompression. Put an empty string in decompressed_ptx_ showing
      // that the original ptx is compressed.
      decompressed_ptx_[ptx.data()] = "";
    }
    ptx_by_compute_capability_[std::tuple<int, int>{major, minor}] = ptx.data();
  }
}

string CudaPtxInMemory::DecompressPtx(const char *ptx) {
  // Get the length of the PTX string from the beginning of the buffer.
  uint64 ptx_length = *reinterpret_cast<const uint64 *>(ptx);
  // Get the PTX string from the buffer with offset and length.
  string compressed_ptx(ptx + sizeof(uint64),
                        ptx + sizeof(uint64) + ptx_length);
  string decompressed_ptx;
  // Decompress the PTX string with bzip2.
  LOG(FATAL) << "bzip2 decompression is not supported yet.";
  return decompressed_ptx;
}

const char *CudaPtxInMemory::default_text() const {
  if (ptx_by_compute_capability_.empty()) {
    return nullptr;
  }

  mutex_lock lock{mu_};

  auto ptx = ptx_by_compute_capability_.begin()->second;
  // Check if there is an entry in decompressed ptx table.
  auto decompressed_ptx_iter = decompressed_ptx_.find(ptx);
  if (decompressed_ptx_iter != decompressed_ptx_.end()) {
    // If the decompressed string is empty, which means the ptx hasn't been
    // decompressed, decompress it here.
    if (decompressed_ptx_iter->second.empty()) {
      decompressed_ptx_iter->second = DecompressPtx(ptx);
    }
    return decompressed_ptx_iter->second.c_str();
  }
  return ptx;
}

const char *CudaPtxInMemory::original_default_text() const {
  if (ptx_by_compute_capability_.empty()) {
    return nullptr;
  }

  return ptx_by_compute_capability_.begin()->second;
}

const char *CudaPtxInMemory::text(int compute_capability_major,
                                  int compute_capability_minor) const {
  std::tuple<int, int> capability{compute_capability_major,
                                  compute_capability_minor};

  auto ptx_iter = ptx_by_compute_capability_.find(capability);
  if (ptx_iter == ptx_by_compute_capability_.end()) {
    return nullptr;
  }

  mutex_lock lock{mu_};

  // Check if there is an entry in decompressed ptx table.
  auto decompressed_ptx_iter = decompressed_ptx_.find(ptx_iter->second);
  if (decompressed_ptx_iter != decompressed_ptx_.end()) {
    // If the decompressed string is empty, which means the ptx hasn't been
    // decompressed, decompress it here.
    if (decompressed_ptx_iter->second.empty()) {
      decompressed_ptx_iter->second = DecompressPtx(ptx_iter->second);
    }
    return decompressed_ptx_iter->second.c_str();
  }
  return ptx_iter->second;
}

const char *CudaPtxInMemory::original_text(int compute_capability_major,
                                           int compute_capability_minor) const {
  std::tuple<int, int> capability{compute_capability_major,
                                  compute_capability_minor};

  auto ptx_iter = ptx_by_compute_capability_.find(capability);
  if (ptx_iter == ptx_by_compute_capability_.end()) {
    return nullptr;
  }

  return ptx_iter->second;
}

OpenCLTextOnDisk::OpenCLTextOnDisk(port::StringPiece filename,
                                   port::StringPiece kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

OpenCLTextInMemory::OpenCLTextInMemory(port::StringPiece text,
                                       port::StringPiece kernelname)
    : KernelLoaderSpec(kernelname), text_(std::string(text)) {}

OpenCLBinaryOnDisk::OpenCLBinaryOnDisk(port::StringPiece filename,
                                       port::StringPiece kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextOnDisk(
    port::StringPiece filename, port::StringPiece kernelname) {
  CHECK(ocl_text_on_disk_ == nullptr);
  ocl_text_on_disk_.reset(new OpenCLTextOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLBinaryOnDisk(
    port::StringPiece filename, port::StringPiece kernelname) {
  CHECK(ocl_binary_on_disk_ == nullptr);
  ocl_binary_on_disk_.reset(new OpenCLBinaryOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextInMemory(
    port::StringPiece filename, port::StringPiece kernelname) {
  CHECK(ocl_text_in_memory_ == nullptr);
  ocl_text_in_memory_.reset(new OpenCLTextInMemory{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxOnDisk(
    port::StringPiece filename, port::StringPiece kernelname) {
  CHECK(cuda_ptx_on_disk_ == nullptr);
  cuda_ptx_on_disk_.reset(new CudaPtxOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinInMemory(
    const char *bytes, port::StringPiece kernelname) {
  CHECK(cuda_cubin_in_memory_ == nullptr);
  cuda_cubin_in_memory_.reset(new CudaCubinInMemory{bytes, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinOnDisk(
    port::StringPiece filename, port::StringPiece kernelname) {
  CHECK(cuda_cubin_on_disk_ == nullptr);
  cuda_cubin_on_disk_.reset(new CudaCubinOnDisk{filename, kernelname});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
    port::StringPiece ptx, port::StringPiece kernelname) {
  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{ptx, kernelname, false /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory(
    port::StringPiece ptx, port::StringPiece kernelname) {
  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{ptx, kernelname, true /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
    std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
    port::StringPiece kernelname) {
  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{spec_list, kernelname, false /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory(
    std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
    port::StringPiece kernelname) {
  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(
      new CudaPtxInMemory{spec_list, kernelname, true /* ptx_compressed */});
  return this;
}

MultiKernelLoaderSpec::MultiKernelLoaderSpec(size_t arity) : arity_(arity) {}

}  // namespace stream_executor
