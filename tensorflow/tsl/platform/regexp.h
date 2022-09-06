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

#ifndef TENSORFLOW_TSL_PLATFORM_REGEXP_H_
#define TENSORFLOW_TSL_PLATFORM_REGEXP_H_

#include "tensorflow/tsl/platform/platform.h"


#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_PORTABLE_GOOGLE)
#include "third_party/re2/re2.h"
#else
#include "re2/re2.h"
#endif

#endif  // TENSORFLOW_TSL_PLATFORM_REGEXP_H_
