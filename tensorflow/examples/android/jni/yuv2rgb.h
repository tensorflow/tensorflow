// This is a collection of routines which converts various YUV image formats
// to (A)RGB.

#ifndef ORG_TENSORFLOW_JNI_IMAGEUTILS_YUV2RGB_H_
#define ORG_TENSORFLOW_JNI_IMAGEUTILS_YUV2RGB_H_

#include "tensorflow/core/platform/port.h"

using namespace tensorflow;

#ifdef __cplusplus
extern "C" {
#endif

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
