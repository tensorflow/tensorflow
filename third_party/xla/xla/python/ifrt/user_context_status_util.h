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

#ifndef XLA_PYTHON_IFRT_USER_CONTEXT_STATUS_UTIL_H_
#define XLA_PYTHON_IFRT_USER_CONTEXT_STATUS_UTIL_H_

#include "absl/status/status.h"
#include "xla/python/ifrt/user_context.h"

namespace xla {
namespace ifrt {

// Attaches a reference to a user context to the status payload. This is used
// when a user context ID is available, but `UserContextRef` is not available.
// This happens when only the ID can be plumbed, e.g., within a low-level
// runtime below IFRT.
//
// The low-level runtimes often cannot use this function directly because of
// layering constraint. Then, they may fork the implementation because the
// implementation does not rely on IFRT features.
//
// `status` may be OK. If so, this function call will return the original
// status as-is.
//
// If a user context was already attached to the status, it will be overwritten
// with the new user context.
absl::Status AttachUserContextId(absl::Status status, UserContextId id);

// Attaches a reference to a user context to the status payload. As long as the
// status is alive, the attached user context will also be kept alive.
// This is used within IFRT implementations where it has a
// `UserContextRef` available to use.
//
// When the returned status is serialized and deserialized, the user context
// will be ignored, but its ID will be preserved, giving the same result of
// using `AttachUserContextId()`.
//
// `status` may be OK. If so, this function call will return the original
// status as-is.
//
// If a user context was already attached to the status, it will be overwritten
// with the new user context.
absl::Status AttachUserContextRef(absl::Status status,
                                  UserContextRef user_context);

// Re-attaches any user contexts referenced by ID in the status payload if the
// user contexts are found in the `UserContextRegistry`. This is used when a
// status was originally updated with `AttachedUserContextId` or went through
// serialization and deserialization, and do not have `TrackedUserContextRef`s
// attached.
//
// This is expected to be called when an IFRT implementation receives a status
// with user context IDs from a low-level runtime, and wants to expand the user
// contexts on the same process.
//
// `status` may be OK. If so, this function call will return the original
// status as-is.
absl::Status ReattachUserContextRefs(absl::Status status);

// Expands any user contexts attached to the status and appends them to the
// status message. This is used above IFRT, when the user knows when it is
// desirable to do the expansion (e.g., not on a scarce fiber and already
// holding Python GIL, in case the user context represents a Python traceback
// object).
//
// `status` may be OK. If so, this function call will return the original
// status as-is.
absl::Status ExpandUserContexts(absl::Status status);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_USER_CONTEXT_STATUS_UTIL_H_
