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
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

KernelLoaderSpec KernelLoaderSpec::CreateInProcessSymbolSpec(
    void* symbol, std::string kernel_name, size_t arity,
    KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{InProcessSymbol{symbol}, std::move(kernel_name),
                          arity, kernel_args_packing};
}

KernelLoaderSpec KernelLoaderSpec::CreateSerializableInProcessSymbolSpec(
    std::string persistent_kernel_name, void* symbol, std::string kernel_name,
    size_t arity, KernelArgsPacking kernel_args_packing) {
  return KernelLoaderSpec{
      InProcessSymbol{symbol, std::move(persistent_kernel_name)},
      std::move(kernel_name), arity, kernel_args_packing};
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
  if (std::holds_alternative<KernelArgsPackingFunc>(kernel_args_packing_) &&
      std::get<KernelArgsPackingFunc>(kernel_args_packing_) != nullptr) {
    return absl::UnimplementedError(
        "KernelLoaderSpecs with a function for argument packing is not "
        "serializable.");
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

  if (has_in_process_symbol()) {
    if (in_process_symbol()->persistent_name.empty()) {
      return absl::InvalidArgumentError(
          "KernelLoaderSpec referencing in process device functions can't "
          "be serialized without a persistent kernel name.");
    }
    proto.mutable_in_process_symbol()->set_persistent_name(
        in_process_symbol()->persistent_name);
  }

  CHECK(has_cuda_cubin_in_memory() || has_cuda_ptx_in_memory() ||
        has_in_process_symbol());

  if (std::holds_alternative<KernelArgsPackingSpec>(kernel_args_packing_)) {
    TF_ASSIGN_OR_RETURN(
        *proto.mutable_kernel_args_packing_spec(),
        std::get<KernelArgsPackingSpec>(kernel_args_packing_).ToProto());
  }

  return proto;
}

absl::StatusOr<KernelLoaderSpec> KernelLoaderSpec::FromProto(
    const KernelLoaderSpecProto& proto,
    std::optional<SymbolResolver> symbol_resolver) {
  KernelArgsPacking kernel_args_packing;
  if (proto.has_kernel_args_packing_spec()) {
    TF_ASSIGN_OR_RETURN(
        kernel_args_packing,
        KernelArgsPackingSpec::FromProto(proto.kernel_args_packing_spec()));
  }

  switch (proto.payload_case()) {
    case KernelLoaderSpecProto::kCubin: {
      const std::string& data = proto.cubin().data();
      return KernelLoaderSpec::CreateOwningCudaCubinInMemorySpec(
          std::vector<uint8_t>{data.begin(), data.end()}, proto.kernel_name(),
          proto.arity(), std::move(kernel_args_packing));
    }

    case KernelLoaderSpecProto::kPtx: {
      return KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
          proto.ptx().data(), proto.kernel_name(), proto.arity(),
          std::move(kernel_args_packing));
    }

    case KernelLoaderSpecProto::kInProcessSymbol: {
      if (!symbol_resolver.has_value()) {
        return absl::InvalidArgumentError(
            "KernelLoaderSpecProto references in process symbol, but no symbol "
            "registry has been provided.");
      }
      if (proto.in_process_symbol().persistent_name().empty()) {
        return absl::InvalidArgumentError(
            "KernelLoaderSpecProto references in process symbol, but no "
            "persistent name has been provided.");
      }

      TF_ASSIGN_OR_RETURN(
          void* symbol,
          (*symbol_resolver)(proto.in_process_symbol().persistent_name()));
      return KernelLoaderSpec::CreateSerializableInProcessSymbolSpec(
          proto.in_process_symbol().persistent_name(), symbol,
          proto.kernel_name(), proto.arity(), kernel_args_packing);
    }

    default:
      return absl::InvalidArgumentError(
          "Invalid KernelLoaderSpecProto. Neither PTX nor CUBIN payload has "
          "been "
          "found.");
  }
}

}  // namespace stream_executor
