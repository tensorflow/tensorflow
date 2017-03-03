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

#ifndef TENSORFLOW_PLATFORM_DENORMAL_H_
#define TENSORFLOW_PLATFORM_DENORMAL_H_

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace port {

// While this class is active, denormal floating point numbers are flushed
// to zero.  The destructor restores the original flags.
class ScopedFlushDenormal {
 public:
  ScopedFlushDenormal();
  ~ScopedFlushDenormal();

 private:
  bool flush_zero_mode_;
  bool denormals_zero_mode_;
  TF_DISALLOW_COPY_AND_ASSIGN(ScopedFlushDenormal);
};

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DENORMAL_H_
