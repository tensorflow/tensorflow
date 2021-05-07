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

// These utility functions allow for the conversion of RGB data to YUV data.

#include "tensorflow/tools/android/test/jni/rgb2yuv.h"

static inline void WriteYUV(const int x, const int y, const int width,
                            const int r8, const int g8, const int b8,
                            uint8_t* const pY, uint8_t* const pUV) {
  // Using formulas from http://msdn.microsoft.com/en-us/library/ms893078
  *pY = ((66 * r8 + 129 * g8 + 25 * b8 + 128) >> 8) + 16;

  // Odd widths get rounded up so that UV blocks on the side don't get cut off.
  const int blocks_per_row = (width + 1) / 2;

  // 2 bytes per UV block
  const int offset = 2 * (((y / 2) * blocks_per_row + (x / 2)));

  // U and V are the average values of all 4 pixels in the block.
  if (!(x & 1) && !(y & 1)) {
    // Explicitly clear the block if this is the first pixel in it.
    pUV[offset] = 0;
    pUV[offset + 1] = 0;
  }

  // V (with divide by 4 factored in)
#ifdef __APPLE__
  const int u_offset = 0;
  const int v_offset = 1;
#else
  const int u_offset = 1;
  const int v_offset = 0;
#endif
  pUV[offset + v_offset] += ((112 * r8 - 94 * g8 - 18 * b8 + 128) >> 10) + 32;

  // U (with divide by 4 factored in)
  pUV[offset + u_offset] += ((-38 * r8 - 74 * g8 + 112 * b8 + 128) >> 10) + 32;
}

void ConvertARGB8888ToYUV420SP(const uint32_t* const input,
                               uint8_t* const output, int width, int height) {
  uint8_t* pY = output;
  uint8_t* pUV = output + (width * height);
  const uint32_t* in = input;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const uint32_t rgb = *in++;
#ifdef __APPLE__
      const int nB = (rgb >> 8) & 0xFF;
      const int nG = (rgb >> 16) & 0xFF;
      const int nR = (rgb >> 24) & 0xFF;
#else
      const int nR = (rgb >> 16) & 0xFF;
      const int nG = (rgb >> 8) & 0xFF;
      const int nB = rgb & 0xFF;
#endif
      WriteYUV(x, y, width, nR, nG, nB, pY++, pUV);
    }
  }
}

void ConvertRGB565ToYUV420SP(const uint16_t* const input, uint8_t* const output,
                             const int width, const int height) {
  uint8_t* pY = output;
  uint8_t* pUV = output + (width * height);
  const uint16_t* in = input;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const uint32_t rgb = *in++;

      const int r5 = ((rgb >> 11) & 0x1F);
      const int g6 = ((rgb >> 5) & 0x3F);
      const int b5 = (rgb & 0x1F);

      // Shift left, then fill in the empty low bits with a copy of the high
      // bits so we can stretch across the entire 0 - 255 range.
      const int r8 = r5 << 3 | r5 >> 2;
      const int g8 = g6 << 2 | g6 >> 4;
      const int b8 = b5 << 3 | b5 >> 2;

      WriteYUV(x, y, width, r8, g8, b8, pY++, pUV);
    }
  }
}
