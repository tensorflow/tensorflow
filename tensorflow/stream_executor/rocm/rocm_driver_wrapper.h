/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file wraps rocm driver calls with dso loader so that we don't need to
// have explicit linking to librocm. All TF rocm driver usage should route
// through this wrapper.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_

#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_driver_wrapper.h"

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_
