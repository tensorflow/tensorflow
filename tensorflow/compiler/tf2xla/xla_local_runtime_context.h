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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_LOCAL_RUNTIME_CONTEXT_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_LOCAL_RUNTIME_CONTEXT_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

// Forward-declare the ThreadPoolDevice so that it can be ignored unless it's
// actually used.  E.g. some ahead-of-time compiled computations don't need a
// thread pool.
namespace Eigen {
struct ThreadPoolDevice;
}

namespace tensorflow {

// An instance of this class is passed to each call from tensorflow into a
// compiled XLA computation. See xla_launch_ops.cc.
struct XlaLocalRuntimeContext {
 public:
  XlaLocalRuntimeContext() {}

  // Kernels implemented using custom call ops set this if they encounter an
  // error. The error is checked after the entire XLA computation is
  // complete.
  //
  // error+error_msg are used instead of Status to reduce the binary size
  // overhead for ahead-of-time compiled binaries.
  bool error = false;
  string error_msg;

  // Kernels that need a thread pool can get it from here.
  const Eigen::ThreadPoolDevice* thread_pool = nullptr;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaLocalRuntimeContext);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_LOCAL_RUNTIME_CONTEXT_H_
