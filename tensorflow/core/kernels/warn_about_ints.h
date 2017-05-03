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

#ifndef TENSORFLOW_KERNELS_WARN_ABOUT_INTS_H_
#define TENSORFLOW_KERNELS_WARN_ABOUT_INTS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// Warn if a kernel is being created using ints
// TODO(irving): Remove in TF 2.0 along with the bad op registrations.
void WarnAboutInts(OpKernelConstruction* context);

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_WARN_ABOUT_INTS_H_
