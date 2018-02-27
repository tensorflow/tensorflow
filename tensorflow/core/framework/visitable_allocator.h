/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VISITABLE_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_VISITABLE_ALLOCATOR_H_

#include <functional>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tracking_allocator.h"

namespace tensorflow {

// Subclass VisitableAllocator instead of Allocator when a memory
// allocator needs to enable some kind of registration/deregistration
// of memory areas.
class VisitableAllocator : public Allocator {
 public:
  // Visitor gets called with a pointer to a memory area and its
  // size in bytes.
  typedef std::function<void(void*, size_t)> Visitor;

  // Register a visitor guaranteed to be called exactly once on each
  // chunk of memory newly allocated from the underlying device.
  // Typically, chunks will be reused and possibly sub-divided by a
  // pool manager, so the calls will happen only once per process
  // execution, not once per tensor (re)allocation.
  virtual void AddAllocVisitor(Visitor visitor) = 0;

  // Register a visitor guaranteed to be called on each chunk of
  // memory returned to the underlying device.
  virtual void AddFreeVisitor(Visitor visitor) = 0;
};

// Needed for cases when a VisitableAllocator gets wrapped for tracking.
// Multiple-inheritance is considered acceptable in this case because
// VisitableAllocator is a pure virtual interface and only TrackingAllocator
// has default implementation.
class TrackingVisitableAllocator : public TrackingAllocator,
                                   public VisitableAllocator {
 public:
  TrackingVisitableAllocator(VisitableAllocator* allocator, bool track_ids)
      : TrackingAllocator(allocator, track_ids), allocator_(allocator) {}
  ~TrackingVisitableAllocator() override {}

  string Name() override { return TrackingAllocator::Name(); }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return TrackingAllocator::AllocateRaw(alignment, num_bytes);
  }

  void DeallocateRaw(void* ptr) override {
    TrackingAllocator::DeallocateRaw(ptr);
  }

  void AddAllocVisitor(Visitor visitor) override {
    allocator_->AddAllocVisitor(visitor);
  }

  void AddFreeVisitor(Visitor visitor) override {
    allocator_->AddFreeVisitor(visitor);
  }

 protected:
  VisitableAllocator* allocator_;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_VISITABLE_ALLOCATOR_H_
