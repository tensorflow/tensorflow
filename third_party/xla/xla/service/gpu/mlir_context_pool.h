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

#ifndef XLA_SERVICE_GPU_MLIR_CONTEXT_POOL_H_
#define XLA_SERVICE_GPU_MLIR_CONTEXT_POOL_H_

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/runtime/object_pool.h"

namespace xla::gpu {

// A pool of MLIR contexts. Contexts borrowed from the pool are owned by a
// single thread at a time, which allows them to be single-threaded (i.e. to
// skip StorageUniquer locking).
using MlirContextPool = ObjectPool<std::unique_ptr<mlir::MLIRContext>>;

class PooledOrFallbackMlirContext;

// Borrows a context from `pool` if it is non-null; otherwise uses `fallback`.
// `fallback` may be null only if `pool` is non-null.
absl::StatusOr<PooledOrFallbackMlirContext> BorrowMlirContextOr(
    MlirContextPool* absl_nullable pool,
    mlir::MLIRContext* absl_nullable fallback);

// An MLIR context that is either borrowed from a `MlirContextPool` (and
// returned to it on destruction) or a non-owning reference to a caller-owned
// fallback context. Must outlive every use of `get()`.
class PooledOrFallbackMlirContext {
 public:
  PooledOrFallbackMlirContext(PooledOrFallbackMlirContext&&) = default;
  PooledOrFallbackMlirContext& operator=(PooledOrFallbackMlirContext&&) =
      default;

  mlir::MLIRContext* get() const {
    return borrowed_ != nullptr ? borrowed_->get() : fallback_;
  }

 private:
  friend absl::StatusOr<PooledOrFallbackMlirContext> BorrowMlirContextOr(
      MlirContextPool* absl_nullable pool,
      mlir::MLIRContext* absl_nullable fallback);

  explicit PooledOrFallbackMlirContext(MlirContextPool::BorrowedObject borrowed)
      : borrowed_(std::move(borrowed)) {}
  explicit PooledOrFallbackMlirContext(
      mlir::MLIRContext* absl_nullable fallback)
      : fallback_(fallback) {}

  // Null when using the fallback context.
  MlirContextPool::BorrowedObject borrowed_;
  mlir::MLIRContext* absl_nullable fallback_ = nullptr;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MLIR_CONTEXT_POOL_H_
