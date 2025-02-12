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

#ifndef XLA_TSL_PLATFORM_ROCM_ROCDL_PATH_H_
#define XLA_TSL_PLATFORM_ROCM_ROCDL_PATH_H_

#include "xla/tsl/platform/types.h"

namespace tsl {

// Returns the root directory of the ROCM SDK, which contains sub-folders such
// as bin, lib, and rocdl.
string RocmRoot();

// Returns the directory that contains ROCm-Device-Libs files in the ROCm SDK.
string RocdlRoot();

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_ROCM_ROCDL_PATH_H_
