/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_FFI_API_KERNEL_API_H_
#define XLA_FFI_API_KERNEL_API_H_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <variant>
#include <vector>

#include "xla/ffi/api/kernel_c_api.h"  // IWYU pragma: export

namespace xla::ffi {

// Device pointers are just passed as is by pointer value.
struct DevicePointer {
  void* ptr;
};

// Host values are passed by value to the kernel.
// The size must be non-zero.
// In the end the runtime will copy the value to a temporary buffer.
struct HostValue {
  const void* ptr;
  size_t size;
};

// A kernel argument is either a pointer to a device buffer or a host value.
using KernelArg = std::variant<DevicePointer, HostValue>;

enum class SourceFormat {
  kFunctionPtr = XLA_FFI_SourceFormat_FUNCTION_PTR,
};

namespace internal {

template <class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
// Deduction guide.
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

// Converts a span of `KernelArg` or `void*` to a vector of
// `XLA_FFI_KernelArg`.
template <typename KernelArgSpan>
std::vector<XLA_FFI_KernelArg> ConvertKernelArgs(const KernelArgSpan& args) {
  std::vector<XLA_FFI_KernelArg> raw_args;
  const size_t num_args = std::size(args);
  raw_args.reserve(num_args);
  for (const auto& arg : args) {
    // Can't use std/absl::remove_cvref because we don't have absl OR cpp20.
    // NOLINTNEXTLINE(modernize-type-traits)
    using ArgT = std::remove_cv_t<std::remove_reference_t<decltype(arg)>>;
    if constexpr (std::is_same_v<ArgT, KernelArg>) {
      std::visit(Overload{[&](const DevicePointer& arg) {
                            raw_args.push_back(
                                {arg.ptr, 0, XLA_FFI_KernelArgType_DevicePtr});
                          },
                          [&](const HostValue& arg) {
                            raw_args.push_back(
                                {arg.ptr, static_cast<int64_t>(arg.size),
                                 XLA_FFI_KernelArgType_HostValue});
                          }},
                 arg);
    } else {
      // Handle void* arguments for convenience.
      raw_args.push_back({arg, 0, XLA_FFI_KernelArgType_DevicePtr});
    }
  }
  return raw_args;
}

}  // namespace internal

}  // namespace xla::ffi

#endif  // XLA_FFI_API_KERNEL_API_H_
