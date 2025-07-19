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
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel_spec.pb.h"

namespace stream_executor {

KernelLoaderSpec KernelLoaderSpec::CreateInProcessSymbolSpec(
    void* symbol, std::string kernel_name, size_t arity,
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

KernelLoaderSpec KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
    std::vector<uint8_t> cubin_bytes, std::string kernel_name, size_t arity,
    KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{OwningCudaCubinInMemory{std::move(cubin_bytes)},
                          std::move(kernel_name), arity, kernel_args_packing};
}

KernelLoaderSpec KernelLoaderSpec::CreateCudaPtxInMemorySpec(
    absl::string_view ptx, std::string kernel_name, size_t arity,
    KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{CudaPtxInMemory{ptx}, std::move(kernel_name), arity,
                          kernel_args_packing};
}

KernelLoaderSpec KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
    std::string ptx, std::string kernel_name, size_t arity,
    KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{OwningCudaPtxInMemory{std::move(ptx)},
                          std::move(kernel_name), arity, kernel_args_packing};
}

absl::StatusOr<KernelLoaderSpecProto> KernelLoaderSpec::ToProto() const {
  if (kernel_args_packing_ != nullptr) {
    return absl::UnimplementedError(
        "KernelLoaderSpecs with KernelArgsPacking are not currently"
        "serializable.");
  }

  if (has_in_process_symbol()) {
    return absl::InvalidArgumentError(
        "KernelLoaderSpec referencing in process device functions can't "
        "be serialized.");
  }

  KernelLoaderSpecProto proto{};
  proto.set_arity(arity_);
  proto.set_kernel_name(kernel_name_);

  if (has_cuda_cubin_in_memory()) {
    absl::Span<const uint8_t> data = cuda_cubin_in_memory()->cubin_bytes;
    proto.mutable_cubin()->mutable_data()->assign(data.begin(), data.end());
  }

  if (has_cuda_ptx_in_memory()) {
    proto.mutable_ptx()->set_data(cuda_ptx_in_memory()->ptx);
  }

  CHECK(proto.has_cubin() || proto.has_ptx());

  return proto;
}

absl::StatusOr<KernelLoaderSpec> KernelLoaderSpec::FromProto(
    const KernelLoaderSpecProto& proto) {
  if (proto.has_cubin()) {
    const std::string& data = proto.cubin().data();
    return KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
        std::vector<uint8_t>{data.begin(), data.end()}, proto.kernel_name(),
        proto.arity());
  }

  if (proto.has_ptx()) {
    return KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
        proto.ptx().data(), proto.kernel_name(), proto.arity());
  }

  return absl::InvalidArgumentError(
      "Invalid KernelLoaderSpecProto. Neither PTX nor CUBIN payload has been "
      "found.");
}

}  // namespace stream_executor
