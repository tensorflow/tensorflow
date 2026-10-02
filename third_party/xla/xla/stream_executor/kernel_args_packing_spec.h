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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_ARGS_PACKING_SPEC_H_
#define XLA_STREAM_EXECUTOR_KERNEL_ARGS_PACKING_SPEC_H_

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"
#include "xla/stream_executor/kernel_args_packing_spec.pb.h"

namespace stream_executor {

// Represents a relocation of an argument.
class KernelArgPackingRelocation {
 public:
  enum class Kind { kBits64Absolute };

  KernelArgPackingRelocation(Kind kind, int argument_index)
      : kind_(kind), argument_index_(argument_index) {}
  Kind kind() const { return kind_; }
  int argument_index() const { return argument_index_; }

  absl::StatusOr<KernelArgPackingRelocationProto> ToProto() const;

  static absl::StatusOr<KernelArgPackingRelocation> FromProto(
      const KernelArgPackingRelocationProto& proto);

  friend bool operator==(const KernelArgPackingRelocation& a,
                         const KernelArgPackingRelocation& b) {
    return a.kind_ == b.kind_ && a.argument_index_ == b.argument_index_;
  }
  friend bool operator!=(const KernelArgPackingRelocation& a,
                         const KernelArgPackingRelocation& b) {
    return !(a == b);
  }
  template <typename H>
  friend H AbslHashValue(H h, const KernelArgPackingRelocation& r) {
    return H::combine(std::move(h), r.kind_, r.argument_index_);
  }

 private:
  Kind kind_;
  int argument_index_;
};

// Represents the packing spec for a single argument of a kernel.
class KernelArgPackingSpec {
 public:
  KernelArgPackingSpec() = default;

  // Materializes the argument buffer for this packing spec. The `args` span
  // must contain at least the number of arguments referenced in the packing
  // spec, otherwise an error will be returned.
  absl::StatusOr<std::vector<char>> BuildArgument(
      absl::Span<const std::unique_ptr<PackedArgBase>> args) const;

  // Builds KernelArgPackingSpec that refers to the given arg number.
  static KernelArgPackingSpec BuildArgRelocation(int argument_index);

  // Returns the index of the argument this spec relocates, or `std::nullopt`
  // if this spec holds a constant.
  std::optional<int> relocation_argument_index() const {
    return relocation_.has_value()
               ? std::make_optional(relocation_->argument_index())
               : std::nullopt;
  }

  // Builds KernelArgPackingSpec that refers to a constant. The value must be
  // trivially copyable.
  template <typename T>
  static KernelArgPackingSpec BuildFor(const T& value) {
    using Packed = typename KernelArgPacking<T>::Type;

    static_assert(std::is_trivially_copyable_v<Packed>,
                  "The given value must be trivially copyable");
    Packed packed = KernelArgPacking<T>::Pack(value);

    std::vector<char> temp_storage(sizeof(Packed));
    std::memcpy(temp_storage.data(), &packed, sizeof(Packed));
    return KernelArgPackingSpec(std::move(temp_storage), {});
  }

  absl::StatusOr<KernelArgPackingSpecProto> ToProto() const;

  static absl::StatusOr<KernelArgPackingSpec> FromProto(
      const KernelArgPackingSpecProto& proto);

  friend bool operator==(const KernelArgPackingSpec& a,
                         const KernelArgPackingSpec& b) {
    return a.constant_ == b.constant_ && a.relocation_ == b.relocation_;
  }
  friend bool operator!=(const KernelArgPackingSpec& a,
                         const KernelArgPackingSpec& b) {
    return !(a == b);
  }
  template <typename H>
  friend H AbslHashValue(H h, const KernelArgPackingSpec& s) {
    return H::combine(std::move(h), s.constant_, s.relocation_);
  }

 private:
  KernelArgPackingSpec(std::vector<char> constant,
                       std::optional<KernelArgPackingRelocation> relocation)
      : constant_(std::move(constant)), relocation_(std::move(relocation)) {}

  std::vector<char> constant_;
  std::optional<KernelArgPackingRelocation> relocation_;
};

// `KernelArgsPackingSpec` defines how to convert a list of device buffer
// pointers into a packed argument buffer that can be passed to a device kernel.
//
// When calling a custom kernel from XLA each HLO parameter and HLO result is
// represented as a device buffer pointer and by default the custom kernel gets
// launched with those pointers as kernel arguments in a predefined order -
// input parameters first, then output parameters.
//
// This is very inflexible so KernelArgsPackingSpec allows to specify a
// transformation from a list of device buffer pointers (usually created by
// xla::emitters::KernelArguments) to a packed argument buffer (list of byte
// arrays). Each argument of the custom kernel can be a buffer of arbitrary
// bytes with a list of placeholders (which we call relocations - similar to
// linker relocations) that will be replaced by the runtime with the address of
// the corresponding device buffer.
//
// Since this is all declarative it is also possible to serialize
// KernelArgsPackingSpec to a proto.
//
// Usage example: We want to launch a kernel that has the following launch
// arguments:
//
// - Output buffer pointer
// - Input buffer pointer
// - Constant value 42
//
// Source code:
//
//   KernelArgsPackingSpec packing_spec;
//
//   // `1` refers to the second argument as defined by
//   // xla::emitters::KernelArguments. In case of 1 HLO input this is the first
//   // output buffer.
//   packing_spec.AddAddressArgument(1);
//
//   // `0` refers to the first argument as defined by
//   // xla::emitters::KernelArguments.
//   packing_spec.AddAddressArgument(0);
//   packing_spec.AddConstantArgument<int64_t>(42);
//
//   custom_kernel.SetArgumentsPackingSpec(packing_spec);
//
// Now the custom kernel gets launched with a packed argument buffer that looks
// like this: | output_ptr | input_ptr | 42 |
class KernelArgsPackingSpec {
 public:
  KernelArgsPackingSpec() = default;
  explicit KernelArgsPackingSpec(
      std::vector<KernelArgPackingSpec> kernel_arguments)
      : kernel_arguments_(std::move(kernel_arguments)) {}

  // Builds the packing spec that passes the first `num_args` device buffer
  // pointers to the kernel unchanged, in order. This is the right spec for
  // kernels whose signature is exactly "all inputs, then all outputs".
  static KernelArgsPackingSpec Identity(int num_args);

  // Adds a single argument packing spec to the kernel arguments packing spec.
  void AddArgument(KernelArgPackingSpec spec) {
    kernel_arguments_.push_back(std::move(spec));
  }

  // Adds an argument that only contains a pointer to the `argument_index`th
  // argument.
  void AddAddressArgument(int argument_index) {
    kernel_arguments_.push_back(
        KernelArgPackingSpec::BuildArgRelocation(argument_index));
  }

  template <typename T>
  void AddConstantArgument(const T& value) {
    kernel_arguments_.push_back(KernelArgPackingSpec::BuildFor(value));
  }

  // Number of arguments the kernel will be launched with.
  size_t size() const { return kernel_arguments_.size(); }

  // Returns the largest device buffer index referenced by any relocation, or
  // `std::nullopt` if the spec contains no relocations.
  std::optional<int> MaxArgumentIndex() const;

  // Returns an error if any relocation refers to a device buffer index that is
  // out of range for `num_available_args` buffers. `BuildArguments` performs
  // the same check, but only once the kernel is actually launched; calling
  // `Validate` while emitting lets a mismatch between the packing spec and the
  // kernel arguments surface at compile time instead.
  absl::Status Validate(size_t num_available_args) const;

  // Materializes the argument buffers for this packing spec. The `args` span
  // must contain at least the number of arguments referenced in the packing
  // spec, otherwise an error will be returned.
  absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>> BuildArguments(
      absl::Span<const std::unique_ptr<PackedArgBase>> args,
      size_t shared_memory_bytes) const;

  absl::StatusOr<KernelArgsPackingSpecProto> ToProto() const;

  static absl::StatusOr<KernelArgsPackingSpec> FromProto(
      const KernelArgsPackingSpecProto& proto);

  friend bool operator==(const KernelArgsPackingSpec& a,
                         const KernelArgsPackingSpec& b) {
    return a.kernel_arguments_ == b.kernel_arguments_;
  }
  friend bool operator!=(const KernelArgsPackingSpec& a,
                         const KernelArgsPackingSpec& b) {
    return !(a == b);
  }
  template <typename H>
  friend H AbslHashValue(H h, const KernelArgsPackingSpec& s) {
    return H::combine(std::move(h), s.kernel_arguments_);
  }

 private:
  std::vector<KernelArgPackingSpec> kernel_arguments_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_ARGS_PACKING_SPEC_H_
