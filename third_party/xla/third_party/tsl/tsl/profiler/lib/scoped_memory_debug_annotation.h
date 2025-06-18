/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
#define TENSORFLOW_TSL_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"

namespace tsl {
namespace profiler {

std::string DefaultPendingShapeFunc();

// Annotations for memory profiling and debugging purpose.
// ScopedMemoryDebugAnnotation will cache the annotations in thread-local
// memory, and some allocators will try to tag allocations with the annotations.
struct MemoryDebugAnnotation {
  absl::string_view pending_op_name;
  int64_t pending_step_id = 0;
  absl::string_view pending_region_type;
  int32_t pending_data_type = 0;
  // A lambda function, when invoked, it will generate the string that describe
  // the shape of the pending tensor. By default, the TensorShape string is an
  // empty string.
  std::function<std::string()> pending_shape_func = DefaultPendingShapeFunc;
};

// Wrapper class of MemoryDebugAnnotation for RAII.
class ScopedMemoryDebugAnnotation {
 public:
  static const MemoryDebugAnnotation& CurrentAnnotation() {
    return *ThreadMemoryDebugAnnotation();
  }

  explicit ScopedMemoryDebugAnnotation(absl::string_view op_name) {
    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    *thread_local_annotation = MemoryDebugAnnotation();
    thread_local_annotation->pending_op_name = op_name;
  }

  explicit ScopedMemoryDebugAnnotation(absl::string_view op_name,
                                       int64_t step_id) {
    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    *thread_local_annotation = MemoryDebugAnnotation();
    thread_local_annotation->pending_op_name = op_name;
    thread_local_annotation->pending_step_id = step_id;
  }

  // This constructor keeps the pending_op_name and pending_step_id from parent
  // (if any).  Otherwise it overwrites with op_name.
  explicit ScopedMemoryDebugAnnotation(
      absl::string_view op_name, absl::string_view region_type,
      int32_t data_type, std::function<std::string()>&& pending_shape_func) {
    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    if (thread_local_annotation->pending_op_name.empty()) {
      thread_local_annotation->pending_op_name = op_name;
    }
    thread_local_annotation->pending_region_type = region_type;
    thread_local_annotation->pending_data_type = data_type;
    thread_local_annotation->pending_shape_func = std::move(pending_shape_func);
  }

  explicit ScopedMemoryDebugAnnotation(
      absl::string_view op_name, int64_t step_id, absl::string_view region_type,
      int32_t data_type, std::function<std::string()>&& pending_shape_func) {
    MemoryDebugAnnotation* thread_local_annotation =
        ThreadMemoryDebugAnnotation();
    last_annotation_ = *thread_local_annotation;
    thread_local_annotation->pending_op_name = op_name;
    thread_local_annotation->pending_step_id = step_id;
    thread_local_annotation->pending_region_type = region_type;
    thread_local_annotation->pending_data_type = data_type;
    thread_local_annotation->pending_shape_func = std::move(pending_shape_func);
  }

  ~ScopedMemoryDebugAnnotation() {
    *ThreadMemoryDebugAnnotation() = last_annotation_;
  }

 private:
  // Returns a pointer to the MemoryDebugAnnotation for the current thread.
  static MemoryDebugAnnotation* ThreadMemoryDebugAnnotation();

  // Stores the previous values in case the annotations are nested.
  MemoryDebugAnnotation last_annotation_;

  ScopedMemoryDebugAnnotation(const ScopedMemoryDebugAnnotation&) = delete;
  ScopedMemoryDebugAnnotation& operator=(const ScopedMemoryDebugAnnotation&) =
      delete;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_SCOPED_MEMORY_DEBUG_ANNOTATION_H_
