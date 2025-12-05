/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_ARGUMENT_PACKING_SPEC_H_
#define XLA_STREAM_EXECUTOR_KERNEL_ARGUMENT_PACKING_SPEC_H_

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"
#include "xla/stream_executor/kernel_argument_packing_spec.pb.h"

namespace stream_executor {

// Represents a relocation of an argument to a specific location in the
// packed argument buffer. Imagine an arbitrary buffer of bytes with a
// placeholder that later on will be replaced by the runtime with a
// pointer to the actual argument. Currently only supports 64 bit absolute
// pointers to device memory.
class ArgumentPackingRelocation {
 public:
  enum class Type { kBits64Absolute };

  explicit ArgumentPackingRelocation(Type type, int argument_index,
                                     size_t offset)
      : type_(type), argument_index_(argument_index), offset_(offset) {}
  Type type() const { return type_; }
  int argument_index() const { return argument_index_; }
  size_t offset() const { return offset_; }

  absl::StatusOr<ArgumentPackingRelocationProto> ToProto() const;

  static absl::StatusOr<ArgumentPackingRelocation> FromProto(
      const ArgumentPackingRelocationProto& proto);

 private:
  Type type_;
  int argument_index_;
  size_t offset_;
};

// Represents the packing spec for a single argument of a kernel. So this is a
// buffer of bytes and a list of relocations (placeholders) that describe
// where later on argument buffer addresses will be written to.
class SingleArgumentPackingSpec {
 public:
  SingleArgumentPackingSpec() = default;
  SingleArgumentPackingSpec(std::vector<char> storage,
                            std::vector<ArgumentPackingRelocation> relocations)
      : storage_(std::move(storage)), relocations_(std::move(relocations)) {}

  // Materializes the argument buffer for this packing spec. The `args` span
  // must contain at least the number of arguments referenced in the packing
  // spec, otherwise an error will be returned.
  absl::StatusOr<std::vector<char>> BuildArgument(
      absl::Span<const DeviceAddressBase> args) const;

  // Writes a placeholder to the argument packing spec that will be replaced
  // by the runtime with the address of the argument `argument_index`th
  // argument. Currently this is always a 64bits absolute pointer to device
  // memory. Other types of relocations will be added in the future if needed.
  void WriteArgumentAddress(int argument_index);

  // Writes a constant value to the argument packing spec. The value must be
  // trivially copyable.
  template <typename T>
  void WriteConstant(T value) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "The given value must be trivially copyable");
    std::array<char, sizeof(T)> temp_storage;
    std::memcpy(temp_storage.data(), &value, sizeof(T));
    storage_.insert(storage_.end(), temp_storage.begin(), temp_storage.end());
  }

  absl::StatusOr<SingleArgumentPackingSpecProto> ToProto() const;

  static absl::StatusOr<SingleArgumentPackingSpec> FromProto(
      const SingleArgumentPackingSpecProto& proto);

 private:
  std::vector<char> storage_;
  std::vector<ArgumentPackingRelocation> relocations_;
};

// `KernelArgumentsPackingSpec` defines how to convert a list of device buffer
// pointers into a packed argument buffer that can be passed to a device kernel.
//
// When calling a custom kernel from XLA each HLO parameter and HLO result is
// represented as a device buffer pointer and by default the custom kernel gets
// launched with those pointers as kernel arguments in a predefined order -
// input parameters first, then output parameters.
//
// This is very inflexible so KernelArgumentsPackingSpec allows to specify a
// transformation from a list of device buffer pointers (usually created by
// xla::emitters::KernelArguments) to a packed argument buffer (list of byte
// arrays). Each argument of the custom kernel can be a buffer of arbitrary
// bytes with a list of placeholders (which we call relocations - similar to
// linker relocations) that will be replaced by the runtime with the address of
// the corresponding device buffer.
//
// Since this is all declarative it is also possible to serialize
// KernelArgumentsPackingSpec to a proto.
//
// Usage example: We want to launch a kernel that has the following launch
// arguments:
// - Output buffer pointer
// - Input buffer pointer
// - Constant value 42
//
// KernelArgumentsPackingSpec packing_spec;
// // `1` refers to the second argument as defined by
// // xla::emitters::KernelArguments. In case of 1 HLO input this is the first
// // output buffer.
// packing_spec.AddAddressArgument(1);
// // `0` refers to the first argument as defined by
// // xla::emitters::KernelArguments.
// packing_spec.AddAddressArgument(0);
// packing_spec.AddConstantArgument<int64_t>(42);
//
// custom_kernel.SetArgumentsPackingSpec(packing_spec);
//
// Now the custom kernel gets launched with a packed argument buffer that looks
// like this: | output_ptr | input_ptr | 42 |
class KernelArgumentsPackingSpec {
 public:
  KernelArgumentsPackingSpec() = default;
  explicit KernelArgumentsPackingSpec(
      std::vector<SingleArgumentPackingSpec> kernel_arguments)
      : kernel_arguments_(std::move(kernel_arguments)) {}

  // Adds a single argument packing spec to the kernel arguments packing spec.
  void AddArgument(SingleArgumentPackingSpec spec) {
    kernel_arguments_.push_back(std::move(spec));
  }

  // Adds a an argument that only contains a pointer to the `argument_index`th
  // argument.
  void AddAddressArgument(int argument_index) {
    kernel_arguments_.push_back(SingleArgumentPackingSpec());
    kernel_arguments_.back().WriteArgumentAddress(argument_index);
  }

  template <typename T>
  void AddConstantArgument(T value) {
    kernel_arguments_.push_back(SingleArgumentPackingSpec());
    kernel_arguments_.back().WriteConstant(value);
  }

  // Materializes the argument buffers for this packing spec. The `args` span
  // must contain at least the number of arguments referenced in the packing
  // spec, otherwise an error will be returned.
  absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>> BuildArguments(
      absl::Span<const DeviceAddressBase> thunk_arguments,
      size_t shared_memory_bytes) const;

  absl::StatusOr<KernelArgumentsPackingSpecProto> ToProto() const;

  static absl::StatusOr<KernelArgumentsPackingSpec> FromProto(
      const KernelArgumentsPackingSpecProto& proto);

 private:
  std::vector<SingleArgumentPackingSpec> kernel_arguments_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_ARGUMENT_PACKING_SPEC_H_
