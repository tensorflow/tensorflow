/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TYPED_KERNEL_FACTORY_H_
#define XLA_STREAM_EXECUTOR_TYPED_KERNEL_FACTORY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

// This class creates TypedKernel objects for stream executors based on the
// specification.
template <typename... Params>
class TypedKernelFactory {
 public:
  // Creates a typed kernel on a given executor from a kernel specification.
  static absl::StatusOr<TypedKernel<Params...>> Create(
      StreamExecutor *executor, const MultiKernelLoaderSpec &spec) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Kernel> kernel,
                        executor->LoadKernel(spec));
    return TypedKernel<Params...>(std::move(kernel));
  }

  // Creates a kernel which can be launched on a stream from a
  // PTX (and optional CUBIN), such that the types of the arguments provided for
  // launch would have to match types of the arguments provided at creation
  // time. The canonical storage for both ptx and cubin_data should outlive the
  // lifetime of the kernel.
  static absl::StatusOr<TypedKernel<Params...>> Create(StreamExecutor *executor,
                                                       std::string kernel_name,
                                                       absl::string_view ptx) {
    MultiKernelLoaderSpec loader_spec =
        MultiKernelLoaderSpec::CreateCudaPtxInMemorySpec(
            ptx, std::move(kernel_name),
            TypedKernel<Params...>::kNumberOfParameters);

    return Create(executor, loader_spec);
  }

  static absl::StatusOr<TypedKernel<Params...>> Create(
      StreamExecutor *executor, std::string kernel_name,
      absl::Span<const uint8_t> cubin_data) {
    MultiKernelLoaderSpec loader_spec =
        MultiKernelLoaderSpec::CreateCudaCubinInMemorySpec(
            cubin_data, std::move(kernel_name),
            TypedKernel<Params...>::kNumberOfParameters);

    return Create(executor, loader_spec);
  }

  // Creates a kernel which can be launched on a stream from
  // an in-process symbol pointer.
  static absl::StatusOr<TypedKernel<Params...>> Create(StreamExecutor *executor,
                                                       std::string kernel_name,
                                                       void *symbol) {
    MultiKernelLoaderSpec loader_spec =
        MultiKernelLoaderSpec::CreateInProcessSymbolSpec(
            symbol, std::move(kernel_name),
            TypedKernel<Params...>::kNumberOfParameters);

    return Create(executor, loader_spec);
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TYPED_KERNEL_FACTORY_H_
