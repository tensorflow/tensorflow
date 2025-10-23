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

#include "xla/shape_pool.h"

#include <cstddef>
#include <memory>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/shape.h"

namespace xla {

ShapePool* ShapePool::Default() {
  static absl::NoDestructor<ShapePool> pool;
  return &*pool;
}

size_t ShapePool::GarbageCollect() {
  absl::MutexLock lock(mu_);
  size_t num_erased = absl::erase_if(
      canonical_shapes_, [](auto& entry) { return entry.second.expired(); });
  VLOG(3) << "Garbage collected " << num_erased << " shapes";
  return num_erased;
}

std::shared_ptr<Shape> ShapePool::GetCanonicalShape(const Shape& shape) {
  absl::MutexLock lock(mu_);

  auto [it, _] = canonical_shapes_.try_emplace(shape, std::weak_ptr<Shape>());

  std::shared_ptr<Shape> ptr = it->second.lock();
  if (ABSL_PREDICT_TRUE(ptr != nullptr)) {
    return ptr;
  }

  std::shared_ptr<Shape> canonical_shape = std::make_shared<Shape>(shape);
  it->second = canonical_shape;
  return canonical_shape;
}

std::shared_ptr<Shape> ShapePool::GetCanonicalShape(
    std::shared_ptr<Shape> shape) {
  absl::MutexLock lock(mu_);

  auto [it, _] = canonical_shapes_.try_emplace(*shape, std::weak_ptr<Shape>());

  std::shared_ptr<Shape> ptr = it->second.lock();
  if (ABSL_PREDICT_TRUE(ptr != nullptr)) {
    return ptr;
  }

  it->second = shape;
  return shape;
}

}  // namespace xla
