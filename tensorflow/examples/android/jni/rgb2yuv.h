#ifndef ORG_TENSORFLOW_JNI_IMAGEUTILS_RGB2YUV_H_
#define ORG_TENSORFLOW_JNI_IMAGEUTILS_RGB2YUV_H_

#include "tensorflow/core/platform/port.h"

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
