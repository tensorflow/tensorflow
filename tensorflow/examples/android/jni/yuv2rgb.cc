// This is a collection of routines which converts various YUV image formats
// to ARGB.

#include "tensorflow/examples/android/jni/yuv2rgb.h"

#ifndef MAX
#define MAX(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#define MIN(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#endif

// This value is 2 ^ 18 - 1, and is used to clamp the RGB values before their ranges
// are normalized to eight bits.
static const int kMaxChannelValue = 262143;

//  Accepts a YUV 4:2:0 image with a plane of 8 bit Y samples followed by an
//  interleaved U/V plane containing 8 bit 2x2 subsampled chroma samples,
//  except the interleave order of U and V is reversed. Converts to a packed
//  ARGB 32 bit output of the same pixel dimensions.
void ConvertYUV420SPToARGB8888(const uint8* const yData,
                               const uint8* const uvData,
                               uint32* const output, const int width,
                               const int height) {
  const uint8* pY = yData;
  const uint8* pUV = uvData;
  uint32* out = output;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = *pY++;
      int offset = (y >> 1) * width + 2 * (x >> 1);
#ifdef __APPLE__
      int nU = pUV[offset];
      int nV = pUV[offset + 1];
#else
      int nV = pUV[offset];
      int nU = pUV[offset + 1];
#endif

      nY -= 16;
      nU -= 128;
      nV -= 128;
      if (nY < 0) nY = 0;

      // This is the floating point equivalent. We do the conversion in integer
      // because some Android devices do not have floating point in hardware.
      // nR = (int)(1.164 * nY + 2.018 * nU);
      // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
      // nB = (int)(1.164 * nY + 1.596 * nV);

      int nR = (int)(1192 * nY + 1634 * nV);
      int nG = (int)(1192 * nY - 833 * nV - 400 * nU);
      int nB = (int)(1192 * nY + 2066 * nU);

      nR = MIN(kMaxChannelValue, MAX(0, nR));
      nG = MIN(kMaxChannelValue, MAX(0, nG));
      nB = MIN(kMaxChannelValue, MAX(0, nB));

      nR = (nR >> 10) & 0xff;
      nG = (nG >> 10) & 0xff;
      nB = (nB >> 10) & 0xff;
      *out++ = 0xff000000 | (nR << 16) | (nG << 8) | nB;
    }
  }
}

// The same as above, but downsamples each dimension to half size.
void ConvertYUV420SPToARGB8888HalfSize(const uint8* const input,
                                       uint32* const output,
                                       int width, int height) {
  const uint8* pY = input;
  const uint8* pUV = input + (width * height);
  uint32* out = output;
  int stride = width;
  width >>= 1;
  height >>= 1;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = (pY[0] + pY[1] + pY[stride] + pY[stride + 1]) >> 2;
      pY += 2;
#ifdef __APPLE__
      int nU = *pUV++;
      int nV = *pUV++;
#else
      int nV = *pUV++;
      int nU = *pUV++;
#endif

      nY -= 16;
      nU -= 128;
      nV -= 128;
      if (nY < 0) nY = 0;

      int nR = (int)(1192 * nY + 1634 * nV);
      int nG = (int)(1192 * nY - 833 * nV - 400 * nU);
      int nB = (int)(1192 * nY + 2066 * nU);

      nR = MIN(kMaxChannelValue, MAX(0, nR));
      nG = MIN(kMaxChannelValue, MAX(0, nG));
      nB = MIN(kMaxChannelValue, MAX(0, nB));

      nR = (nR >> 10) & 0xff;
      nG = (nG >> 10) & 0xff;
      nB = (nB >> 10) & 0xff;
      *out++ = 0xff000000 | (nR << 16) | (nG << 8) | nB;
    }
    pY += stride;
  }
}

//  Accepts a YUV 4:2:0 image with a plane of 8 bit Y samples followed by an
//  interleaved U/V plane containing 8 bit 2x2 subsampled chroma samples,
//  except the interleave order of U and V is reversed. Converts to a packed
//  RGB 565 bit output of the same pixel dimensions.
void ConvertYUV420SPToRGB565(const uint8* const input, uint16* const output,
                             const int width, const int height) {
  const uint8* pY = input;
  const uint8* pUV = input + (width * height);
  uint16 *out = output;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = *pY++;
      int offset = (y >> 1) * width + 2 * (x >> 1);
#ifdef __APPLE__
      int nU = pUV[offset];
      int nV = pUV[offset + 1];
#else
      int nV = pUV[offset];
      int nU = pUV[offset + 1];
#endif

      nY -= 16;
      nU -= 128;
      nV -= 128;
      if (nY < 0) nY = 0;

      // This is the floating point equivalent. We do the conversion in integer
      // because some Android devices do not have floating point in hardware.
      // nR = (int)(1.164 * nY + 2.018 * nU);
      // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
      // nB = (int)(1.164 * nY + 1.596 * nV);

      int nR = (int)(1192 * nY + 1634 * nV);
      int nG = (int)(1192 * nY - 833 * nV - 400 * nU);
      int nB = (int)(1192 * nY + 2066 * nU);

      nR = MIN(kMaxChannelValue, MAX(0, nR));
      nG = MIN(kMaxChannelValue, MAX(0, nG));
      nB = MIN(kMaxChannelValue, MAX(0, nB));

      // Shift more than for ARGB8888 and apply appropriate bitmask.
      nR = (nR >> 13) & 0x1f;
      nG = (nG >> 12) & 0x3f;
      nB = (nB >> 13) & 0x1f;

      // R is high 5 bits, G is middle 6 bits, and B is low 5 bits.
      *out++ = (nR << 11) | (nG << 5) | nB;
    }
  }
}
