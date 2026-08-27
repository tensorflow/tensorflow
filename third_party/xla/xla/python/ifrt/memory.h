/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_MEMORY_H_
#define XLA_PYTHON_IFRT_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {
namespace ifrt {

// `MemoryKind` uniquely identifies a group of memory spaces with a
// platform-dependent string. When no specific memory kind is chosen,
// the universal default "device" memory kind is used.
class MemoryKind {
 public:
  // Creates `MemoryKind` with default "device" memory kind.
  MemoryKind();

  // Creates `MemoryKind` from a platform-dependent identifier of a memory kind.
  // `MemoryKind` will be stable even after the string referenced by
  // `memory_kind` is deallocated.
  explicit MemoryKind(std::optional<absl::string_view> memory_kind);

  bool operator==(const MemoryKind& other) const {
    // Use a pointer comparison. `memory_kind_` always points to an interned
    // string. We can only check the beginning of the string because having a
    // different length will lead to a different pointer during interning.
    return memory_kind_.data() == other.memory_kind_.data();
  }
  bool operator!=(const MemoryKind& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const MemoryKind& memory_kind) {
    return H::combine(std::move(h), memory_kind.memory_kind_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MemoryKind& memory_kind) {
    sink.Append(memory_kind.memory_kind_);
  }

  ABSL_DEPRECATED(
      "Use `value()` instead. If it is required to check if the memory kind is "
      "default, use `is_default()`.")
  std::optional<absl::string_view> memory_kind() const { return memory_kind_; }

  bool is_default() const;
  absl::string_view value() const { return memory_kind_; }

 private:
  std::string ToString() const;

  absl::string_view memory_kind_;
};

// Canonicalizes `MemoryKind`. If `MemoryKind` has no memory kind chosen,
// returns a default `MemoryKind` chosen for the device. If there is no default
// indicated by the device, simply returns `MemoryKind` with no memory kind
// chosen.
//
// TODO(b/356623715): Remove this function.
MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind, const Device* device);

TSL_LIB_GTL_DEFINE_INT_TYPE(MemoryId, int32_t);

// `Memory` represents a memory space that one or more devices can be attached
// to. A platform may have multiple memory spaces with different backing
// hardware or memory region types.
class Memory : public RTTIExtends<Memory, RTTIRoot> {
 public:
  Memory() = default;

  // Not copyable or movable.
  Memory(const Memory&) = delete;
  Memory(Memory&&) = delete;
  Memory& operator=(const Memory&) = delete;
  Memory& operator=(Memory&&) = delete;

  virtual MemoryId Id() const = 0;

  // A platform-dependent string that uniquely identifies the kind of the
  // memory.
  virtual const MemoryKind& Kind() const = 0;

  // Debug string suitable for reading by end users, should be reasonably terse.
  virtual absl::string_view ToString() const = 0;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  //
  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  ABSL_DEPRECATED("Memory implements AbslStringify; rely on that instead.")
  virtual absl::string_view DebugString() const = 0;

  // The devices to which this memory space is attached.
  virtual absl::Span<Device* const> Devices() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Memory& memory) {
    sink.Append(memory.DebugString());
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Memory* memory) {
    if (memory == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(memory->DebugString());
    }
  }

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MEMORY_H_
