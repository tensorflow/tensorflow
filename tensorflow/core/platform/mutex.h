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

#ifndef TENSORFLOW_CORE_PLATFORM_MUTEX_H_
#define TENSORFLOW_CORE_PLATFORM_MUTEX_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/mutex.h"

namespace tensorflow {

using tsl::Condition;
using tsl::condition_variable;
using tsl::ConditionResult;
using tsl::kCond_MaybeNotified;
using tsl::kCond_Timeout;
using tsl::LINKER_INITIALIZED;
using tsl::LinkerInitialized;
using tsl::mutex;
using tsl::mutex_lock;
using tsl::tf_shared_lock;
using tsl::WaitForMilliseconds;
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_MUTEX_H_
