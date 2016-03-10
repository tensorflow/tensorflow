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

// This is a collection of routines which converts various YUV image formats
// to (A)RGB.

#ifndef ORG_TENSORFLOW_JNI_IMAGEUTILS_YUV2RGB_H_
#define ORG_TENSORFLOW_JNI_IMAGEUTILS_YUV2RGB_H_

#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

#ifdef __cplusplus
extern "C" {
#endif

void ConvertYUV420ToARGB8888(const uint8* const yData, const uint8* const uData,
                             const uint8* const vData, uint32* const output,
                             const int width, const int height,
                             const int y_row_stride, const int uv_row_stride,
                             const int uv_pixel_stride);

// Converts YUV420 semi-planar data to ARGB 8888 data using the supplied width
// and height. The input and output must already be allocated and non-null.
// For efficiency, no error checking is performed.
void ConvertYUV420SPToARGB8888(const uint8* const pY, const uint8* const pUV,
                               uint32* const output, const int width,
                               const int height);

// The same as above, but downsamples each dimension to half size.
void ConvertYUV420SPToARGB8888HalfSize(const uint8* const input,
                                       uint32* const output,
                                       int width, int height);

// Converts YUV420 semi-planar data to RGB 565 data using the supplied width
// and height. The input and output must already be allocated and non-null.
// For efficiency, no error checking is performed.
void ConvertYUV420SPToRGB565(const uint8* const input, uint16* const output,
                             const int width, const int height);

#ifdef __cplusplus
}
#endif

#endif  // ORG_TENSORFLOW_JNI_IMAGEUTILS_YUV2RGB_H_
