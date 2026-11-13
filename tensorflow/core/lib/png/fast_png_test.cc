/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/png/fast_png.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace png {
namespace {

// Row padding is filled with an asymmetric byte that MakeTestImage never
// produces at (0, 0, 0), so an encoder that reads past `width * channels` into
// the stride tail surfaces as a decode mismatch.
constexpr uint8_t kPaddingByte = 0xA5;

// Per-axis increments are distinct odd primes (coprime with 256), so swapped
// axes, a wrong stride, or a transposed read mismatch at almost every byte,
// and no axis is constant modulo 256.
constexpr int kXStep = 37;
constexpr int kYStep = 13;
constexpr int kChannelStep = 97;

// Odd, non-lane-aligned stride tails so fpnge cannot treat rows as packed.
constexpr int kOddStridePadding = 19;
constexpr int kSmallStridePadding = 5;

// Prime dimensions keep rows off every SIMD lane multiple (4, 8, 16, 32), so
// fpnge's partial-vector tail path runs on each row.
constexpr int kPrimeWidth = 67;
constexpr int kPrimeHeight = 43;

// Matches kMaxFastPngDim in fast_png.cc (PNG_USER_WIDTH_MAX).
constexpr int kMaxFastPngDim = 1000000;

// Any zlib level fpnge handles; used where the level is not under test.
constexpr int kAnyFastLevel = 3;

std::vector<uint8_t> MakeTestImage(int width, int height, int channels,
                                   int row_bytes) {
  std::vector<uint8_t> src(height * row_bytes, kPaddingByte);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        src[y * row_bytes + x * channels + c] =
            static_cast<uint8_t>(x * kXStep + y * kYStep + c * kChannelStep);
      }
    }
  }
  return src;
}

void VerifyDecodedPixels(absl::string_view encoded,
                         const std::vector<uint8_t>& src, int width, int height,
                         int channels, int row_bytes) {
  DecodeContext decode;
  ASSERT_TRUE(
      CommonInitDecode(encoded, channels, /*desired_channel_bits=*/8, &decode));
  EXPECT_EQ(decode.width, width);
  EXPECT_EQ(decode.height, height);
  EXPECT_EQ(decode.channels, channels);

  std::vector<uint8_t> decoded(height * width * channels);
  ASSERT_TRUE(CommonFinishDecode(decoded.data(), width * channels, &decode));

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        ASSERT_EQ(decoded[(y * width + x) * channels + c],
                  src[y * row_bytes + x * channels + c]);
      }
    }
  }
}

void VerifyLosslessRoundTrip(int width, int height, int channels,
                             int extra_stride_bytes, int compression) {
  const int row_bytes = width * channels + extra_stride_bytes;
  const std::vector<uint8_t> src =
      MakeTestImage(width, height, channels, row_bytes);

  tstring encoded;
  ASSERT_TRUE(WriteImageToBuffer(src.data(), width, height, row_bytes, channels,
                                 /*channel_bits=*/8, compression, &encoded,
                                 /*metadata=*/nullptr));
  ASSERT_FALSE(encoded.empty());
  VerifyDecodedPixels(absl::string_view(encoded.data(), encoded.size()), src,
                      width, height, channels, row_bytes);
}

TEST(FastPngTest, LosslessRoundTripAcrossChannelsStridesAndLevels) {
  for (int channels : {1, 2, 3, 4}) {
    for (int extra_stride : {0, kOddStridePadding}) {
      // Covers every fpnge-eligible level plus the libpng-only ones (0, 9).
      for (int compression : {-1, 0, 1, 2, 3, 4, 5, 6, 9}) {
        VerifyLosslessRoundTrip(kPrimeWidth, kPrimeHeight, channels,
                                extra_stride, compression);
      }
    }
  }
}

TEST(FastPngTest, RejectsUnsupportedParametersWithoutMutatingOutput) {
  constexpr int kSide = 16;
  constexpr int kChannels = 3;
  constexpr int kRowBytes = kSide * kChannels;
  const std::vector<uint8_t> pixels =
      MakeTestImage(kSide, kSide, kChannels, kRowBytes);
  std::string out = "untouched";
  // 16-bit samples.
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), kSide, kSide, kRowBytes,
                                kChannels, /*channel_bits=*/16, kAnyFastLevel,
                                &out));
  // Uncompressed and high-compression levels stay on libpng.
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), kSide, kSide, kRowBytes,
                                kChannels, /*channel_bits=*/8,
                                /*compression=*/0, &out));
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), kSide, kSide, kRowBytes,
                                kChannels, /*channel_bits=*/8,
                                /*compression=*/7, &out));
  // Stride shorter than one packed row.
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), kSide, kSide,
                                /*row_bytes=*/kRowBytes - 1, kChannels,
                                /*channel_bits=*/8, kAnyFastLevel, &out));
  // Width past libpng's decoder limit.
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), kMaxFastPngDim + 1,
                                /*height=*/1,
                                /*row_bytes=*/(kMaxFastPngDim + 1) * kChannels,
                                kChannels, /*channel_bits=*/8, kAnyFastLevel,
                                &out));
  EXPECT_EQ(out, "untouched");

  // WriteImageToBuffer must also reject row_bytes < width * num_channels *
  // (channel_bits / 8) without reading out of bounds in the libpng fallback.
  EXPECT_FALSE(WriteImageToBuffer(pixels.data(), kSide, kSide,
                                  /*row_bytes=*/kRowBytes - 1, kChannels,
                                  /*channel_bits=*/8, kAnyFastLevel, &out,
                                  /*metadata=*/nullptr));
}

#if defined(TF_PNG_ENABLE_FPNGE)
TEST(FastPngTest, DirectTryFastEncodePngCoversEdgeCasesAndLargeBuffers) {
  // Both std::string and tstring overloads, overwriting non-empty output,
  // narrow widths (including 1x1), and levels 2 and 5.
  for (int width : {1, 2, 3, 7, 16}) {
    for (int height : {1, 5}) {
      for (int channels : {1, 2, 3, 4}) {
        const int row_bytes = width * channels + kSmallStridePadding;
        const std::vector<uint8_t> src =
            MakeTestImage(width, height, channels, row_bytes);

        std::string std_out = "previous_contents_to_overwrite";
        ASSERT_TRUE(TryFastEncodePng(src.data(), width, height, row_bytes,
                                     channels, /*channel_bits=*/8,
                                     /*compression=*/2, &std_out));
        VerifyDecodedPixels(std_out, src, width, height, channels, row_bytes);

        tstring tstr_out = "previous_tstring_contents";
        ASSERT_TRUE(TryFastEncodePng(src.data(), width, height, row_bytes,
                                     channels, /*channel_bits=*/8,
                                     /*compression=*/5, &tstr_out));
        VerifyDecodedPixels(absl::string_view(tstr_out.data(), tstr_out.size()),
                            src, width, height, channels, row_bytes);
      }
    }
  }

  // Image whose FPNGEOutputAllocSize exceeds the 8 MiB thread-local scratch cap
  // (1200 x 1200 x 4 = 5.76 MiB raw, ~11.5 MiB alloc_size > 8 MiB).
  constexpr int kLargeW = 1200;
  constexpr int kLargeH = 1200;
  constexpr int kLargeC = 4;
  constexpr int kLargeRowBytes = kLargeW * kLargeC;
  const std::vector<uint8_t> large_src =
      MakeTestImage(kLargeW, kLargeH, kLargeC, kLargeRowBytes);
  tstring large_encoded;
  ASSERT_TRUE(TryFastEncodePng(large_src.data(), kLargeW, kLargeH,
                               kLargeRowBytes, kLargeC, /*channel_bits=*/8,
                               /*compression=*/-1, &large_encoded));
  VerifyDecodedPixels(
      absl::string_view(large_encoded.data(), large_encoded.size()), large_src,
      kLargeW, kLargeH, kLargeC, kLargeRowBytes);
}
#endif

TEST(FastPngTest, MetadataFallsBackToLibpngAndPreservesTextChunks) {
  constexpr int kSide = 8;
  constexpr int kChannels = 3;
  const std::vector<uint8_t> pixels =
      MakeTestImage(kSide, kSide, kChannels, kSide * kChannels);
  std::vector<std::pair<std::string, std::string>> metadata = {
      {"Author", "TensorFlow"}};
  std::string encoded;
  ASSERT_TRUE(WriteImageToBuffer(pixels.data(), kSide, kSide, kSide * kChannels,
                                 kChannels, /*channel_bits=*/8, kAnyFastLevel,
                                 &encoded, &metadata));
  EXPECT_NE(encoded.find("Author"), std::string::npos);
  EXPECT_NE(encoded.find("TensorFlow"), std::string::npos);
}

}  // namespace
}  // namespace png
}  // namespace tensorflow
