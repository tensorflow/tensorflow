/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_TF_XLA_STUB_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_TF_XLA_STUB_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
// Returns an error if the XLA JIT is enabled via `session_options` or if the
// TF_XLA_FLAGS or XLA_FLAGS environment variables are set, but neither the
// XLA CPU JIT nor the XLA GPU JIT are linked in.
//
// If `session_options` is null then only the environment variables are checked.
Status CheckXlaJitOptimizerOptions(const SessionOptions* session_options);

// The XLA CPU JIT creates a static instance of this class to notify
// `CheckXlaJitOptimizerOptions` that the XLA CPU JIT is linked in.
//
// NB!  The constructor of this class (if run at all) needs to be ordered (via
// happens before) before any call to `CheckXlaJitOptimizerOptions`.
class XlaCpuJitIsLinkedIn {
 public:
  XlaCpuJitIsLinkedIn();
};

// The XLA GPU JIT creates a static instance of this class to notify
// `CheckXlaJitOptimizerOptions` that the XLA GPU JIT is linked in.
//
// NB!  The constructor of this class (if run at all) needs to be ordered (via
// happens before) before any call to `CheckXlaJitOptimizerOptions`.
class XlaGpuJitIsLinkedIn {
 public:
  XlaGpuJitIsLinkedIn();
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_TF_XLA_STUB_H_
