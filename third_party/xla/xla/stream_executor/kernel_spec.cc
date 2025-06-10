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

#include "xla/stream_executor/kernel_spec.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace stream_executor {

KernelLoaderSpec::KernelLoaderSpec(absl::string_view kernel_name)
    : kernel_name_(std::string(kernel_name)) {}

InProcessSymbol::InProcessSymbol(void *symbol, std::string kernel_name)
    : KernelLoaderSpec(std::move(kernel_name)), symbol_(symbol) {}

CudaCubinInMemory::CudaCubinInMemory(absl::Span<const uint8_t> cubin_bytes,
                                     absl::string_view kernel_name)
    : KernelLoaderSpec(kernel_name), cubin_bytes_(cubin_bytes) {}

const std::tuple<int, int> CudaPtxInMemory::kMinimumCapability{1, 0};

CudaPtxInMemory::CudaPtxInMemory(absl::string_view ptx,
                                 absl::string_view kernel_name)
    : KernelLoaderSpec(kernel_name) {
  ptx_by_compute_capability_[kMinimumCapability] = ptx.data();
}

CudaPtxInMemory::CudaPtxInMemory(
    const std::initializer_list<CudaPtxInMemory::PtxSpec> &spec_list,
    absl::string_view kernel_name)
    : KernelLoaderSpec(kernel_name) {
  for (const auto &spec : spec_list) {
    int major, minor;
    absl::string_view ptx;
    std::tie(major, minor, ptx) = spec;
    ptx_by_compute_capability_[std::tuple<int, int>{major, minor}] = ptx.data();
  }
}

const char *CudaPtxInMemory::default_text() const {
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

  return ptx_iter->second;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddInProcessSymbol(
    void *symbol, absl::string_view kernel_name) {
  CHECK(in_process_symbol_ == nullptr);
  in_process_symbol_ =
      std::make_shared<InProcessSymbol>(symbol, std::string(kernel_name));
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinInMemory(
    absl::Span<const uint8_t> cubin_bytes, absl::string_view kernel_name) {
  CHECK(cuda_cubin_in_memory_ == nullptr);
  cuda_cubin_in_memory_.reset(new CudaCubinInMemory{cubin_bytes, kernel_name});
  return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
    absl::string_view ptx, absl::string_view kernel_name) {
  CHECK(cuda_ptx_in_memory_ == nullptr);
  cuda_ptx_in_memory_.reset(new CudaPtxInMemory{ptx, kernel_name});
  return this;
}

MultiKernelLoaderSpec::MultiKernelLoaderSpec(
    size_t arity, KernelArgsPacking kernel_args_packing)
    : arity_(arity), kernel_args_packing_(std::move(kernel_args_packing)) {}

}  // namespace stream_executor
