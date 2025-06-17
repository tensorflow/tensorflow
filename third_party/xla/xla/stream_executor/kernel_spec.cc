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
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace stream_executor {

KernelLoaderSpec KernelLoaderSpec::CreateInProcessSymbolSpec(
    void *symbol, std::string kernel_name, size_t arity,
    KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{InProcessSymbol{symbol}, std::move(kernel_name),
                          arity, kernel_args_packing};
}

KernelLoaderSpec KernelLoaderSpec::CreateCudaCubinInMemorySpec(
    absl::Span<const uint8_t> cubin_bytes, std::string kernel_name,
    size_t arity, KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{CudaCubinInMemory{cubin_bytes},
                          std::move(kernel_name), arity, kernel_args_packing};
}

KernelLoaderSpec KernelLoaderSpec::CreateCudaPtxInMemorySpec(
    absl::string_view ptx, std::string kernel_name, size_t arity,
    KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{CudaPtxInMemory{ptx}, std::move(kernel_name), arity,
                          kernel_args_packing};
}

}  // namespace stream_executor
