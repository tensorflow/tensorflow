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

#include "tensorflow/core/lib/webp/webp_io.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace webp {
namespace {

TEST(WebPImageSizeTest, RgbAndRgba) {
  EXPECT_EQ(GetWebPImageSize(7, 5, 3), 105);
  EXPECT_EQ(GetWebPImageSize(7, 5, 4), 140);
}

TEST(WebPImageSizeTest, LargeByteSizesWithoutAllocation) {
  if (sizeof(size_t) < sizeof(uint64_t)) GTEST_SKIP();
  EXPECT_EQ(GetWebPImageSize(65536, 16384, 4), uint64_t{1} << 32);
  EXPECT_EQ(GetWebPImageSize(65536, 32767, 4),
            (uint64_t{1} << 33) - (uint64_t{1} << 18));
}

TEST(WebPImageSizeTest, RejectsInvalidDimensionsAndChannels) {
  struct Dimensions {
    int64_t width;
    int64_t height;
    int channels;
  };
  for (const auto& dimensions :
       {Dimensions{0, 5, 4}, Dimensions{7, 0, 4}, Dimensions{-1, 5, 4},
        Dimensions{7, -1, 4}, Dimensions{7, 5, 0}, Dimensions{7, 5, 1},
        Dimensions{7, 5, 2}, Dimensions{7, 5, 5}}) {
    EXPECT_EQ(GetWebPImageSize(dimensions.width, dimensions.height,
                               dimensions.channels),
              0);
  }
}

TEST(WebPImageSizeTest, RejectsUnrepresentableByteSizes) {
  const int64_t max_dimension = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(GetWebPImageSize(max_dimension, 1, 4), 0);
  EXPECT_EQ(GetWebPImageSize(1, max_dimension, 4), 0);
}

class WebPIoTest : public ::testing::Test {
 protected:
  std::string ReadImage(const std::string& filename) {
    std::string encoded;
    const std::string path = GetDataDependencyFilepath(
        "tensorflow/core/lib/webp/testdata/" + filename);
    EXPECT_TRUE(ReadFileToString(Env::Default(), path, &encoded).ok());
    return encoded;
  }
};

TEST_F(WebPIoTest, StillImagesPreserveRgbAndRgbaAcrossThreadModes) {
  for (const char* filename :
       {"lossless_raw.webp", "RGB_noise_large_pixels_115x115.webp",
        "lossy_alpha1.webp"}) {
    SCOPED_TRACE(filename);
    const std::string encoded = ReadImage(filename);
    int width, height, channels;
    bool animated;
    ASSERT_TRUE(
        DecodeWebPHeader(encoded, &width, &height, &channels, &animated));
    ASSERT_FALSE(animated);
    for (int output_channels : {3, 4}) {
      const size_t size = GetWebPImageSize(width, height, output_channels);
      ASSERT_GT(size, 0);
      std::vector<uint8_t> single_thread(size);
      std::vector<uint8_t> threaded(size);
      ASSERT_TRUE(DecodeWebPImage(encoded, single_thread.data(), width, height,
                                  output_channels, false));
      ASSERT_TRUE(DecodeWebPImage(encoded, threaded.data(), width, height,
                                  output_channels, true));
      EXPECT_EQ(single_thread, threaded);
    }
  }
}

TEST_F(WebPIoTest, StillImageRejectsInvalidDimensionsChannelsAndStride) {
  const std::string encoded = ReadImage("lossless_raw.webp");
  EXPECT_FALSE(DecodeWebPImage(encoded, nullptr, 0, 1, 4, false));
  EXPECT_FALSE(DecodeWebPImage(encoded, nullptr, 1, -1, 4, false));
  EXPECT_FALSE(DecodeWebPImage(encoded, nullptr, 1, 1, 2, false));
  const int overflow_width = std::numeric_limits<int>::max() / 4 + 1;
  EXPECT_FALSE(
      DecodeWebPImage(encoded, nullptr, overflow_width, 1, 4, false));
}

TEST_F(WebPIoTest, AnimationPreservesFirstFrameAndThreadModes) {
  const std::string encoded = ReadImage("bouncy_ball.webp");
  std::vector<uint8_t> animation;
  std::vector<uint8_t> first_frame;
  size_t frame_size = 0;
  int frames = 0;
  auto decode = [&](bool expand, bool use_threads,
                    std::vector<uint8_t>* output) {
    std::string error;
    auto allocate = [&](int count, int width, int height, int channels) {
      frames = count;
      frame_size = GetWebPImageSize(width, height, channels);
      if (frame_size == 0) {
        return static_cast<uint8_t*>(nullptr);
      }
      output->resize(static_cast<size_t>(count) * frame_size);
      return output->data();
    };
    uint8_t* result =
        DecodeWebPAnimation(encoded, allocate, &error, expand, use_threads);
    EXPECT_TRUE(error.empty()) << error;
    EXPECT_EQ(result, output->data());
    return result != nullptr;
  };
  ASSERT_TRUE(decode(true, false, &animation));
  ASSERT_GT(frames, 1);
  ASSERT_EQ(animation.size(), static_cast<size_t>(frames) * frame_size);
  ASSERT_TRUE(decode(false, false, &first_frame));
  EXPECT_EQ(frames, 1);
  ASSERT_EQ(first_frame.size(), frame_size);
  EXPECT_TRUE(
      std::equal(first_frame.begin(), first_frame.end(), animation.begin()));
  std::vector<uint8_t> threaded;
  ASSERT_TRUE(decode(true, true, &threaded));
  EXPECT_EQ(animation, threaded);
}

TEST_F(WebPIoTest, AnimationAcceptsAllocationFailure) {
  const std::string encoded = ReadImage("bouncy_ball.webp");
  for (bool expand : {false, true}) {
    bool called = false;
    auto allocate = [&](int, int, int, int) -> uint8_t* {
      called = true;
      return nullptr;
    };
    std::string error;
    EXPECT_EQ(DecodeWebPAnimation(encoded, allocate, &error, expand, false),
              nullptr);
    EXPECT_TRUE(called);
    EXPECT_EQ(error, "failed to allocate output for WebP Animation");
  }
}

}  // namespace
}  // namespace webp
}  // namespace tensorflow
