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

#include <cstddef>
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
  std::vector<uint8_t> src(height * row_bytes, 0xA5);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        src[y * row_bytes + x * channels + c] =
            static_cast<uint8_t>((x * 37 + y * 13 + c * 97) & 0xFF);
      }
    }
  }

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
    for (int extra_stride : {0, 19}) {
      for (int compression : {-1, 0, 1, 2, 3, 4, 5, 6, 9}) {
        VerifyLosslessRoundTrip(/*width=*/67, /*height=*/43, channels,
                                extra_stride, compression);
      }
    }
  }
}

TEST(FastPngTest, RejectsUnsupportedParametersWithoutMutatingOutput) {
  std::vector<uint8_t> pixels(16 * 16 * 3, 42);
  std::string out = "untouched";
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), 16, 16, 16 * 3, 3,
                                /*channel_bits=*/16, /*compression=*/-1, &out));
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), 16, 16, 16 * 3, 3,
                                /*channel_bits=*/8, /*compression=*/0, &out));
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), 16, 16, 16 * 3, 3,
                                /*channel_bits=*/8, /*compression=*/7, &out));
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), 16, 16, /*row_bytes=*/10, 3,
                                /*channel_bits=*/8, /*compression=*/3, &out));
  EXPECT_FALSE(TryFastEncodePng(pixels.data(), /*width=*/1000001, /*height=*/1,
                                /*row_bytes=*/1000001 * 3, 3,
                                /*channel_bits=*/8, /*compression=*/3, &out));
  EXPECT_EQ(out, "untouched");

  // WriteImageToBuffer must also reject row_bytes < width * num_channels *
  // (channel_bits / 8) without reading out of bounds in the libpng fallback.
  EXPECT_FALSE(WriteImageToBuffer(pixels.data(), 16, 16, /*row_bytes=*/10, 3,
                                  /*channel_bits=*/8, /*compression=*/3, &out,
                                  /*metadata=*/nullptr));
}

#if defined(TF_PNG_ENABLE_FPNGE)
TEST(FastPngTest, DirectTryFastEncodePngCoversEdgeCasesAndLargeBuffers) {
  // Both std::string and tstring overloads, overwriting non-empty output,
  // narrow widths (including 1x1), and levels 2 and 5.
  for (int width : {1, 2, 3, 7, 16}) {
    for (int height : {1, 5}) {
      for (int channels : {1, 2, 3, 4}) {
        const int row_bytes = width * channels + 5;
        std::vector<uint8_t> src(height * row_bytes, 0x5A);
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
              src[y * row_bytes + x * channels + c] =
                  static_cast<uint8_t>((x * 53 + y * 19 + c * 71) & 0xFF);
            }
          }
        }

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
  std::vector<uint8_t> large_src(kLargeH * kLargeRowBytes);
  for (size_t i = 0; i < large_src.size(); ++i) {
    large_src[i] = static_cast<uint8_t>((i * 131) & 0xFF);
  }
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
  std::vector<uint8_t> pixels(8 * 8 * 3, 128);
  std::vector<std::pair<std::string, std::string>> metadata = {
      {"Author", "TensorFlow"}};
  std::string encoded;
  ASSERT_TRUE(WriteImageToBuffer(pixels.data(), 8, 8, 8 * 3, 3,
                                 /*channel_bits=*/8, /*compression=*/3,
                                 &encoded, &metadata));
  EXPECT_NE(encoded.find("Author"), std::string::npos);
  EXPECT_NE(encoded.find("TensorFlow"), std::string::npos);
}

}  // namespace
}  // namespace png
}  // namespace tensorflow
