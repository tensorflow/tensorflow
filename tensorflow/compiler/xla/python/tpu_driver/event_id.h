// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_EVENT_ID_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_EVENT_ID_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"

namespace tpu_driver {

// For gRPC serialization, events are represented as a pair of
// {client, operation} ids. To simplify serialization, these are encoded as a
// single integer field.
//
// This class provides a typed interface for these values as well as support for
// hashing and ostreams (for logging).
struct EventId {
  uint64_t client_id;
  uint64_t operation_id;

  template <typename H>
  friend H AbslHashValue(H h, const EventId& c) {
    return H::combine(std::move(h), c.client_id, c.operation_id);
  }

  bool operator==(const EventId& r) const {
    return r.client_id == client_id && r.operation_id == operation_id;
  }

  friend std::ostream& operator<<(std::ostream& os, EventId r) {
    return os << r.client_id << ":" << r.operation_id;
  }

  std::string ToString() const {
    return absl::StrCat(client_id, ":", operation_id);
  }

  uint64_t AsInt() const { return client_id << 44 | operation_id; }

  static EventId FromInt(uint64_t value) {
    return EventId{value >> 44, value & 0xfffffffffff};
  }
};

}  // namespace tpu_driver

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_EVENT_ID_H_
