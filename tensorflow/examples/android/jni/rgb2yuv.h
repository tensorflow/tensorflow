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

#ifndef ORG_TENSORFLOW_JNI_IMAGEUTILS_RGB2YUV_H_
#define ORG_TENSORFLOW_JNI_IMAGEUTILS_RGB2YUV_H_

#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

#ifdef __cplusplus
extern "C" {
#endif

void ConvertARGB8888ToYUV420SP(const uint32* const input, uint8* const output,
                               int width, int height);

void ConvertRGB565ToYUV420SP(const uint16* const input,
                             uint8* const output,
                             const int width, const int height);

#ifdef __cplusplus
}
#endif

#endif  // ORG_TENSORFLOW_JNI_IMAGEUTILS_RGB2YUV_H_
