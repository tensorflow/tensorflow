/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_JPEG_H_
#define TENSORFLOW_CORE_PLATFORM_JPEG_H_

#include "tensorflow/core/platform/platform.h"

#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/build_config/jpeg.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID)
extern "C" {
#include "jpeg-9a/jerror.h"
#include "jpeg-9a/jinclude.h"
#include "jpeg-9a/jpeglib.h"
#include "jpeg-9a/transupp.h"  // for rotations
}
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // TENSORFLOW_CORE_PLATFORM_JPEG_H_
