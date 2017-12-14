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
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"

#ifndef USE_NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#endif  //  defined(__ARM_NEON__) || defined(__ARM_NEON)
#endif  //  USE_NEON

#ifdef USE_NEON
#include "tensorflow/contrib/lite/kernels/internal/optimized/neon_tensor_utils.h"
#else
#include "tensorflow/contrib/lite/kernels/internal/reference/portable_tensor_utils.h"
#endif  // USE_NEON
