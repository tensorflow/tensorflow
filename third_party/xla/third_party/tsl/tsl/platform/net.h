/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_NET_H_
#define TENSORFLOW_TSL_PLATFORM_NET_H_

#include "absl/base/macros.h"
namespace tsl {
namespace net {

// Return a port number that is not currently bound to any TCP or UDP port.
// On success returns the assigned port number. Otherwise returns -1.
int PickUnusedPort();

// Same as PickUnusedPort(), but fails a CHECK() if a port can't be found. In
// that case, the error message is logged to FATAL.
int PickUnusedPortOrDie();

// Relinquish a claim on the given port which was previously returned by
// PickUnusedPort[OrDie](). This allows PickUnusedPort[OrDie]() to return
// the given port to another caller in the future. Since the number of
// ports the portserver will give to a process is limited (typically 200),
// recycling ports after they are no longer needed can help avoid
// exhausting them. 'port' must be a positive number that was previously
// returned by PickUnusedPort[OrDie](), and not yet recycled, otherwise an
// abort may occur.
void RecycleUnusedPort(int port);
}  // namespace net

namespace internal {
ABSL_DEPRECATE_AND_INLINE()
inline int PickUnusedPort() { return tsl::net::PickUnusedPort(); }

ABSL_DEPRECATE_AND_INLINE()
inline int PickUnusedPortOrDie() { return tsl::net::PickUnusedPortOrDie(); }

ABSL_DEPRECATE_AND_INLINE()
inline void RecycleUnusedPort(int port) { tsl::net::RecycleUnusedPort(port); }
}  // namespace internal
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_NET_H_
