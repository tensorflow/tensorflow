/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/annotation_stack.h"

namespace tensorflow {
namespace profiler {
namespace internal {

std::atomic<bool> g_annotation_enabled;

}  // namespace internal

/*static*/ string* AnnotationStack::ThreadAnnotationStack() {
  static thread_local string annotation_stack;
  return &annotation_stack;
}

}  // namespace profiler
}  // namespace tensorflow
