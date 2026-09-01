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

#ifndef XLA_CORE_COLLECTIVES_REGISTERED_MEMORY_H_
#define XLA_CORE_COLLECTIVES_REGISTERED_MEMORY_H_

#include <string>

#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_address.h"

namespace xla {

// An RAII handle for a device address range that has been registered with a
// communicator for accelerated ("zero-copy") collective operations.
//
// Unlike SymmetricMemory, registration makes NO symmetry assumption about the
// buffer's address across ranks: the same logical buffer may live at a
// different virtual address, and at a different offset within its allocation,
// on every rank. Registration is a purely local optimization hint to the
// backend; it does not change collective semantics or results, only speed. As
// such, registration is *not* a collective operation -- each rank may register
// (and deregister) independently.
//
// Because no symmetry is assumed, a RegisteredMemory exposes nothing usable
// from a device kernel (no peer pointers, no multimem address, no packed kernel
// argument). It is purely a host-side lifetime handle: the range stays
// registered for as long as this object is alive, and is deregistered from the
// owning communicator when the object is destroyed.
class RegisteredMemory {
 public:
  virtual ~RegisteredMemory() = default;

  // The device address range this handle keeps registered.
  virtual stream_executor::DeviceAddressBase addr() const = 0;

  virtual std::string ToString() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RegisteredMemory& mem) {
    absl::Format(&sink, "%s", mem.ToString());
  }
};

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_REGISTERED_MEMORY_H_
