/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_HAVE_BUILT_PATH_FOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_HAVE_BUILT_PATH_FOR_H_

#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {

#if RUY_PLATFORM(X86)
bool HaveBuiltPathForSse42();
bool HaveBuiltPathForAvx2();
bool HaveBuiltPathForAvx512();
bool HaveBuiltPathForAvxVnni();
#endif  // RUY_PLATFORM(X86)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_HAVE_BUILT_PATH_FOR_H_
