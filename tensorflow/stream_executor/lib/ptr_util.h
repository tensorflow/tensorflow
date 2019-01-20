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

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_PTR_UTIL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_PTR_UTIL_H_

#include <memory>
#include "tensorflow/core/util/ptr_util.h"

namespace stream_executor {
using tensorflow::MakeUnique;
using tensorflow::WrapUnique;
}  // namespace stream_executor

namespace perftools {
namespace gputools {

// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(jlebar): Remove this once we've completed
// the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_PTR_UTIL_H_
