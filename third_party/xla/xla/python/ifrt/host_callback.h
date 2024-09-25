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

#ifndef XLA_PYTHON_IFRT_HOST_CALLBACK_H_
#define XLA_PYTHON_IFRT_HOST_CALLBACK_H_

#include <string>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

// Abstract (unloaded) host callback. It wraps a serializable host computation
// that can be loaded as`LoadedHostCallback`.
//
// TODO(hyeontaek): Unify `HostCallback` with `Executable` once `Executable` is
// added.
class HostCallback : public llvm::RTTIExtends<HostCallback, llvm::RTTIRoot> {
 public:
  // Returns a serialized host callback.
  virtual std::string Serialize() const = 0;

  static char ID;  // NOLINT
};

// Abstract loaded host callback. It wraps a host computation that may be called
// during an execution of a `LoadedExecutable`. This interface only represents
// an opaque reference of the host computation; the details of the host
// computation call are implementation specific.
//
// TODO(hyeontaek): Merge `LoadedHostCallback` into `LoadedExecutable`. They
// share a similar lifecycle, and only how their execution is invoked:
// `LoadedExecutable` runs as a top-level standalone runnable, while
// `LoadedHostCallback` runs as a sub-computation of another `LoadedExecutable`
// execution.
class LoadedHostCallback
    : public tsl::ReferenceCounted<LoadedHostCallback>,
      public llvm::RTTIExtends<LoadedHostCallback, llvm::RTTIRoot> {
 public:
  virtual Client* client() const = 0;

  // Returns a serialized host callback.
  //
  // The implementation may return an error if this `LoadedHostCallback` is not
  // serializable, or the information required for serialization is not
  // preserved within this `LoadedHostCallback`.
  //
  // TODO(hyeontaek): Change `Serialize()` to return `HostCallback` instead of a
  // serialized host callback directly.
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_HOST_CALLBACK_H_
