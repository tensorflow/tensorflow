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

// This header file contains the macro definitions for thread safety
// annotations that allow the developers to document the locking policies
// of their multi-threaded code. The annotations can also help program
// analysis tools to identify potential thread safety issues.
//
// The primary documentation on these annotations is external:
// http://clang.llvm.org/docs/ThreadSafetyAnalysis.html
//
// The annotations are implemented using compiler attributes.
// Using the macros defined here instead of the raw attributes allows
// for portability and future compatibility.
//
// When referring to mutexes in the arguments of the attributes, you should
// use variable names or more complex expressions (e.g. my_object->mutex_)
// that evaluate to a concrete mutex object whenever possible. If the mutex
// you want to refer to is not in scope, you may use a member pointer
// (e.g. &MyClass::mutex_) to refer to a mutex in some (unknown) object.
//

#ifndef TENSORFLOW_CORE_PLATFORM_THREAD_ANNOTATIONS_H_
#define TENSORFLOW_CORE_PLATFORM_THREAD_ANNOTATIONS_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/thread_annotations.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/thread_annotations.h

#include "tensorflow/tsl/platform/thread_annotations.h"

#endif  // TENSORFLOW_CORE_PLATFORM_THREAD_ANNOTATIONS_H_
