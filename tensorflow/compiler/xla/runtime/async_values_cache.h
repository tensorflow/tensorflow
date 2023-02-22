/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_ASYNC_VALUES_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_ASYNC_VALUES_CACHE_H_

#include "tensorflow/tsl/platform/platform.h"

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/compiler/xla/runtime/google/async_values_cache.h"
#else
#include "tensorflow/compiler/xla/runtime/default/async_values_cache.h"
#endif

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_ASYNC_VALUES_CACHE_H_
