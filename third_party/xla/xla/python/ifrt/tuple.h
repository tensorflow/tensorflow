/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_TUPLE_H_
#define XLA_PYTHON_IFRT_TUPLE_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

// A tuple containing zero or more Values.
//
// IsDeleted() returns true if either the tuple itself has been deleted or
// any of its elements have been deleted.
// The future returned by Delete() triggers with an Ok status when the tuple and
// all of its elements have been deleted.
class Tuple : public llvm::RTTIExtends<Tuple, Value> {
 public:
  Tuple() = default;

  // Not copyable or movable.
  Tuple(const Tuple&) = delete;
  Tuple(Tuple&&) = delete;
  Tuple& operator=(const Tuple&) = delete;
  Tuple& operator=(Tuple&&) = delete;

  // Returns the arity of the tuple.
  virtual int Arity() = 0;

  // Unpacks the tuple into its constituent pieces.
  virtual absl::Status Unpack(absl::Span<tsl::RCReference<Value>> values) = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_TUPLE_H_
