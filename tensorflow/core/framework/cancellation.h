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

#ifndef TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
#define TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_

#include "xla/tsl/framework/cancellation.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::CancelCallback;
using tsl::CancellationManager;
using tsl::CancellationToken;
using tsl::RegisterCancellationCallback;
// NOLINTEND(misc-unused-using-decls)
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
