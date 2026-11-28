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

#ifndef XLA_PYTHON_IFRT_SIMPLE_USER_CONTEXT_H_
#define XLA_PYTHON_IFRT_SIMPLE_USER_CONTEXT_H_

#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {

// Base class for `UserContext`s with a low-cost call.
//
// `SimpleUserContextInterface::DebugString()` calls are expected to be
// inexpensive and never block. Thus, IFRT runtimes may call them inline and
// frequently without using a separate thread.
//
// One of the use cases of `UserContext`s implementing this base class is to
// provide IFRT runtimes with user-level scope information so that the runtimes
// can internally record `SimpleUserContextInterface::DebugString()` for tracing
// and debugging purposes.
class SimpleUserContext
    : public llvm::RTTIExtends<SimpleUserContext, UserContext> {
 public:
  // Returns true if `user_context` is a `SimpleUserContext` and the runtimes
  // may call `DebugString()`, expecting it to be inexpensive.
  //
  // It wraps `llvm::isa_and_nonnull` for those do not directly depend on LLVM
  // casting utilities.
  //
  // TODO(hyeontaek): Remove this method once IFRT moves away from using LLVM
  // RTTI APIs.
  static bool IsSimpleUserContext(const UserContext* user_context) {
    return llvm::isa_and_nonnull<SimpleUserContext>(user_context);
  }

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SIMPLE_USER_CONTEXT_H_
