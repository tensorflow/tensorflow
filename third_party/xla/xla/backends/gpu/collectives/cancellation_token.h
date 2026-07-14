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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_CANCELLATION_TOKEN_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_CANCELLATION_TOKEN_H_

#include <atomic>

namespace xla::gpu {

// Cancellation token shared between all communicators in a GPU clique and
// allows coordinated cancellation of all of them in presence of communication
// errors. When one communicator in a clique fails, all communicators in the
// same clique are safely cancelled. This class is thread safe.
//
// See JAX details: https://docs.jax.dev/en/latest/fault_tolerance.html
class CancellationToken {
 public:
  explicit CancellationToken(bool cancelled = false) : cancelled_(cancelled) {}

  bool IsCancelled() const { return cancelled_.load(); }
  void Cancel() { cancelled_.store(true); }

 private:
  std::atomic<bool> cancelled_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_CANCELLATION_TOKEN_H_
