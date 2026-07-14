/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Classes to maintain a static registry of memory allocator factories.
#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_

#include <string>
#include <vector>

#include "xla/tsl/framework/allocator_registry.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"

namespace tensorflow {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::AllocatorFactory;
using tsl::AllocatorFactoryRegistration;
using tsl::AllocatorFactoryRegistry;
using tsl::ProcessStateInterface;
// NOLINTEND(misc-unused-using-decls)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
