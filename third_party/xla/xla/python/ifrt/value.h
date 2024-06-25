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

#ifndef XLA_PYTHON_IFRT_VALUE_H_
#define XLA_PYTHON_IFRT_VALUE_H_

#include <string>

#include "absl/status/status.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

// Abstract superclass of values such as arrays.
class Value : public tsl::ReferenceCounted<Value>,
              public llvm::RTTIExtends<Value, llvm::RTTIRoot> {
 public:
  Value() = default;

  // Not copyable or movable.
  Value(const Value&) = delete;
  Value(Value&&) = delete;
  Value& operator=(const Value&) = delete;
  Value& operator=(Value&&) = delete;

  virtual Client* client() const = 0;

  // Returns a future that becomes ready when the buffer is computed or has an
  // error.
  virtual Future<> GetReadyFuture() const = 0;

  // Deletes the value from the devices. The operation may be asynchronous. The
  // returned future will have the result of the deletion on the devices, and
  // will be triggered after all values have been deleted.
  // Implementations that do not track the completion of the deletion operation
  // may make the future immediately ready with an OK status.
  // TODO(phawkins): decide if we want Delete() to be idempotent.
  virtual Future<> Delete() = 0;

  // Returns whether the value has been enqueued for deletion from the devices.
  virtual bool IsDeleted() const = 0;

  virtual std::string DebugString() const = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_VALUE_H_
