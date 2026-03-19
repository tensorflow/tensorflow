/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_LITERAL_POOL_H_
#define XLA_LITERAL_POOL_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/literal.h"
#include "xla/shape.h"

namespace xla {

// Literal pool provides a mechanism to deduplicate identical literals and
// share them across multiple HLO modules.
class LiteralPool {
 public:
  // Returns a default literal pool that can be used across multiple HLO modules
  // in a process.
  static LiteralPool* Default();

  // Returns a canonical literal from the pool. If the literal is not in the
  // pool, it is added to the pool and returned back.
  std::shared_ptr<Literal> GetCanonicalLiteral(const Literal& literal);

  // Returns a canonical literal from the pool. If the literal is not in the
  // pool, it is added to the pool and returned back.
  std::shared_ptr<Literal> GetCanonicalLiteral(
      std::shared_ptr<Literal> literal);

  // Runs garbage collection on all the literals in the pool. Returns the number
  // of literals that were garbage collected.
  size_t GarbageCollect();

  // Runs garbage collection on literals with the given shape. Returns the
  // number of literals that were garbage collected.
  size_t GarbageCollect(Shape shape);

 private:
  // We keep weak pointers to the literals in the pool to allow for garbage
  // collection when owning HLO modules are destroyed. We run periodic garbage
  // collection to clean up the literals that are no longer referenced.
  absl::Mutex mu_;
  absl::flat_hash_map<Shape, std::vector<std::weak_ptr<Literal>>> literals_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_LITERAL_POOL_H_
