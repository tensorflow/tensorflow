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

#ifndef XLA_SHAPE_POOL_H_
#define XLA_SHAPE_POOL_H_

#include <cstddef>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/shape.h"

namespace xla {

// Shape pool provides a mechanism to deduplicate identical shapes and
// share them across multiple HLO instructions (and HLO modules).
class ShapePool {
 public:
  // Returns a default shape pool that can be used across multiple HLO modules
  // in a process.
  static ShapePool* Default();

  // Returns a canonical shape from the pool. If the shape is not in the
  // pool, it is added to the pool and returned back.
  std::shared_ptr<Shape> GetCanonicalShape(const Shape& shape);

  // Returns a canonical shape from the pool. If the shape is not in the
  // pool, it is added to the pool and returned back.
  std::shared_ptr<Shape> GetCanonicalShape(std::shared_ptr<Shape> shape);

  // Runs garbage collection on all shapes in the pool. Returns the number
  // of shapes that were garbage collected.
  size_t GarbageCollect();

 private:
  // We keep weak pointers to the shapes in the pool to allow for garbage
  // collection when owning HLO instructions are destroyed. We run periodic
  // garbage collection to clean up the shapes that are no longer referenced.
  absl::Mutex mu_;
  absl::flat_hash_map<Shape, std::weak_ptr<Shape>> canonical_shapes_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_SHAPE_POOL_H_
