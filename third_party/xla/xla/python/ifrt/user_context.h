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

#ifndef XLA_PYTHON_IFRT_USER_CONTEXT_H_
#define XLA_PYTHON_IFRT_USER_CONTEXT_H_

#include <cstdint>
#include <string>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"  // IWYU pragma: keep
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// UserContext is an interface that must be implemented by any object that the
// user would like to be associated with the runtime operations triggered by an
// IFRT call. For example, a UserContext can be based on a stack trace for
// Python frameworks (e.g.: JAX), or on a "request_id" in case of request
// serving applications.
class UserContext : public tsl::ReferenceCounted<UserContext>,
                    public llvm::RTTIExtends<UserContext, llvm::RTTIRoot> {
 public:
  static tsl::RCReference<UserContext> Default();

  ~UserContext() override = default;

  // Returns a fingerprint of the UserContext. The returned fingerprint is must
  // be non-zero, as the special value of zero is reserved for the IFRT
  // implementations for their internal default UserContext.  IFRT
  // implementations may use internally. IFRT implementations
  // may also use this as a key for holding the UserContexts in a container, and
  // so this should be efficient enough to called multiple times.
  virtual uint64_t Fingerprint() const = 0;

  // Returns a human readable string. Meant for debugging, logging, and for
  // putting together statusz-like pages.
  virtual std::string DebugString() const = 0;

  // For llvm::RTTI
  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_USER_CONTEXT_H_
