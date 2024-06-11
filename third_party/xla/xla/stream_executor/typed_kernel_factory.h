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
#include "xla/stream_executor/kernel_factory.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/statusor.h"

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
                        KernelFactory::Create(executor, spec));
    return TypedKernel<Params...>(std::move(kernel));
  }

  // Creates a kernel which can be launched with `stream.ThenLaunch(...)` from a
  // PTX (and optional CUBIN), such that the types of the arguments provided for
  // launch would have to match types of the arguments provided at creation
  // time. The canonical storage for both ptx and cubin_data should outlive the
  // lifetime of the kernel.
  static absl::StatusOr<TypedKernel<Params...>> Create(
      StreamExecutor *executor, absl::string_view kernel_name,
      absl::string_view ptx, absl::Span<const uint8_t> cubin_data) {
    MultiKernelLoaderSpec loader_spec(
        TypedKernel<Params...>::kNumberOfParameters);
    loader_spec.AddCudaPtxInMemory(ptx, kernel_name);

    if (!cubin_data.empty()) {
      loader_spec.AddCudaCubinInMemory(cubin_data, kernel_name);
    }

    return Create(executor, loader_spec);
  }

  // Creates a kernel which can be launched with `stream.ThenLaunch(...)` from
  // an in-process symbol pointer.
  static absl::StatusOr<TypedKernel<Params...>> Create(
      StreamExecutor *executor, absl::string_view kernel_name, void *symbol) {
    MultiKernelLoaderSpec loader_spec(
        TypedKernel<Params...>::kNumberOfParameters);
    loader_spec.AddInProcessSymbol(symbol, kernel_name);

    return Create(executor, loader_spec);
  }

  // Creates a kernel which can be launched with `stream.ThenLaunch(...)` from
  // an LLVM IR.
  static absl::StatusOr<TypedKernel<Params...>> Create(
      StreamExecutor *executor, absl::string_view ir,
      absl::string_view entrypoint, absl::string_view kernel_name,
      absl::Span<std::string> options) {
    MultiKernelLoaderSpec loader_spec(
        TypedKernel<Params...>::kNumberOfParameters);
    loader_spec.AddLlvmHostKernel(ir, entrypoint, kernel_name, options);

    return Create(executor, loader_spec);
  }
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TYPED_KERNEL_FACTORY_H_
