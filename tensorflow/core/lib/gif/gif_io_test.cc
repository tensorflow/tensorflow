/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/gif/gif_io.h"

#include <memory>

#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace gif {
namespace {

const char kTestData[] = "tensorflow/core/lib/gif/testdata/";

struct DecodeGifTestCase {
  const string filepath;
  const int num_frames;
  const int width;
  const int height;
  const int channels;
};

void ReadFileToStringOrDie(Env* env, const string& filename, string* output) {
  TF_CHECK_OK(ReadFileToString(env, filename, output));
}

void TestDecodeGif(Env* env, DecodeGifTestCase testcase) {
  string gif;
  ReadFileToStringOrDie(env, testcase.filepath, &gif);

  // Decode gif image data.
  std::unique_ptr<uint8[]> imgdata;
  int nframes, w, h, c;
  string error_string;
  imgdata.reset(gif::Decode(
      gif.data(), gif.size(),
      [&](int frame_cnt, int width, int height, int channels) -> uint8* {
        nframes = frame_cnt;
        w = width;
        h = height;
        c = channels;
        return new uint8[static_cast<int64_t>(frame_cnt) * height * width *
                         channels];
      },
      &error_string));
  ASSERT_NE(imgdata, nullptr);
  // Make sure the decoded information matches the ground-truth image info.
  ASSERT_EQ(nframes, testcase.num_frames);
  ASSERT_EQ(w, testcase.width);
  ASSERT_EQ(h, testcase.height);
  ASSERT_EQ(c, testcase.channels);
}

TEST(GifTest, Gif) {
  Env* env = Env::Default();
  const string testdata_path = kTestData;
  std::vector<DecodeGifTestCase> testcases(
      {// file_path, num_of_channels, width, height, channels
       {testdata_path + "lena.gif", 1, 51, 26, 3},
       {testdata_path + "optimized.gif", 12, 20, 40, 3},
       {testdata_path + "red_black.gif", 1, 16, 16, 3},
       {testdata_path + "scan.gif", 12, 20, 40, 3},
       {testdata_path + "squares.gif", 2, 16, 16, 3},
       {testdata_path + "3g_multiframe.gif", 519, 1920, 1080, 3}});

  for (const auto& tc : testcases) {
    TestDecodeGif(env, tc);
  }
}

void TestDecodeAnimatedGif(Env* env, const uint8* gif_data,
                           const string& png_filepath, int frame_idx) {
  string png;  // ground-truth
  ReadFileToStringOrDie(env, png_filepath, &png);

  // Compare decoded gif to ground-truth image frames in png format.
  png::DecodeContext decode;
  png::CommonInitDecode(png, 3, 8, &decode);
  const int width = static_cast<int>(decode.width);
  const int height = static_cast<int>(decode.height);
  std::unique_ptr<uint8[]> png_imgdata(
      new uint8[height * width * decode.channels]);
  png::CommonFinishDecode(reinterpret_cast<png_bytep>(png_imgdata.get()),
                          decode.channels * width * sizeof(uint8), &decode);

  int frame_len = width * height * decode.channels;
  int gif_idx = frame_len * frame_idx;
  for (int i = 0; i < frame_len; i++) {
    ASSERT_EQ(gif_data[gif_idx + i], png_imgdata[i]);
  }
}

TEST(GifTest, AnimatedGif) {
  Env* env = Env::Default();
  const string testdata_path = kTestData;

  // Read animated gif file once.
  string gif;
  ReadFileToStringOrDie(env, testdata_path + "pendulum_sm.gif", &gif);

  std::unique_ptr<uint8[]> gif_imgdata;
  int nframes, w, h, c;
  string error_string;
  gif_imgdata.reset(gif::Decode(
      gif.data(), gif.size(),
      [&](int num_frames, int width, int height, int channels) -> uint8* {
        nframes = num_frames;
        w = width;
        h = height;
        c = channels;
        return new uint8[num_frames * height * width * channels];
      },
      &error_string));

  TestDecodeAnimatedGif(env, gif_imgdata.get(),
                        testdata_path + "pendulum_sm_frame0.png", 0);
  TestDecodeAnimatedGif(env, gif_imgdata.get(),
                        testdata_path + "pendulum_sm_frame1.png", 1);
  TestDecodeAnimatedGif(env, gif_imgdata.get(),
                        testdata_path + "pendulum_sm_frame2.png", 2);
}

void TestExpandAnimations(Env* env, const string& filepath) {
  string gif;
  ReadFileToStringOrDie(env, filepath, &gif);

  std::unique_ptr<uint8[]> imgdata;
  string error_string;
  int nframes;
  // `expand_animations` is set to true by default. Set to false.
  bool expand_animations = false;
  imgdata.reset(gif::Decode(
      gif.data(), gif.size(),
      [&](int frame_cnt, int width, int height, int channels) -> uint8* {
        nframes = frame_cnt;
        return new uint8[frame_cnt * height * width * channels];
      },
      &error_string, expand_animations));

  // Check that only 1 frame is being decoded.
  ASSERT_EQ(nframes, 1);
}

TEST(GifTest, ExpandAnimations) {
  Env* env = Env::Default();
  const string testdata_path = kTestData;

  // Test all animated gif test images.
  TestExpandAnimations(env, testdata_path + "scan.gif");
  TestExpandAnimations(env, testdata_path + "pendulum_sm.gif");
  TestExpandAnimations(env, testdata_path + "squares.gif");
}

void TestInvalidGifFormat(const string& header_bytes) {
  std::unique_ptr<uint8[]> imgdata;
  string error_string;
  int nframes;
  imgdata.reset(gif::Decode(
      header_bytes.data(), header_bytes.size(),
      [&](int frame_cnt, int width, int height, int channels) -> uint8* {
        nframes = frame_cnt;
        return new uint8[frame_cnt * height * width * channels];
      },
      &error_string));

  // Check that decoding image formats other than gif throws an error.
  string err_msg = "failed to open gif file";
  ASSERT_EQ(error_string.substr(0, 23), err_msg);
}

TEST(GifTest, BadGif) {
  // Input header bytes of other image formats to gif decoder.
  TestInvalidGifFormat("\x89\x50\x4E\x47\x0D\x0A\x1A\x0A");  // png
  TestInvalidGifFormat("\x42\x4d");                          // bmp
  TestInvalidGifFormat("\xff\xd8\xff");                      // jpeg
  TestInvalidGifFormat("\x49\x49\x2A\x00");                  // tiff
}

TEST(GifTest, TransparentIndexOutsideColorTable) {
  // Given a GIF with a transparent index outside of its color table...
  unsigned char encoded[43] = {
      'G', 'I', 'F', '8', '9', 'a',  // Header.
      3, 0, 1, 0,                    // Logical width = 3 and height = 1.
      0b1'111'0'000,                 // Global color table present, (7 + 1) bit
                                     // color, unsorted, 2^(0 + 1) palette size.
      0,                             // Background index = 0
      0,                             // Default aspect ratio.
      0x80, 0x00, 0x00,              // Palette entry 0: red.
      0xFF, 0xFF, 0xFF,              // Palette entry 1: white.
      '!', 0xF9, 0x04,               // Graphic Control Extension.
      1,                             // Transparent index is specified.
      0, 0,                          // Delay of 0 seconds.
      2,                             // Transparent index is 2.
      0,                             // End GCE block.
      ',', 0, 0, 0, 0,               // Image at logical (0, 0).
      3, 0, 1, 0,                    // Width = 3, height = 1
      0,                             // No local color table.
      2,                             // Symbols need 2 bits to cover [0, 2].
      2,                             // Two bytes of image data.
      0b01'000'100,                  // Clear (100), 0, 1 (truncated).
      0b0'101'010'0,                 // 1 (continued), 2, End (101), padding.
      0, ';'                         // End of data, end of file.
  };

  // ...decoding that image...
  std::unique_ptr<uint8[]> imgdata;
  string error_string;
  int nframes;
  auto allocate_image_data = [&](int frame_cnt, int width, int height,
                                 int channels) -> uint8* {
    nframes = frame_cnt;
    // Create the unique_ptr here, as gif::Decode does not return a pointer to
    // the allocated array in the case of an error.
    imgdata = std::make_unique<uint8[]>(frame_cnt * height * width * channels);
    return imgdata.get();
  };
  gif::Decode(encoded, sizeof(encoded), allocate_image_data, &error_string);

  // ...should be successful and treat the pixels with the transparent index as
  // transparent.
  ASSERT_EQ(nframes, 1);
  ASSERT_EQ(error_string, "");
  uint8 expected[9] = {
      0x80, 0x00, 0x00,  // Red (palette entry 0).
      0xFF, 0xFF, 0xFF,  // White (palette entry 1).
      0x00, 0x00, 0x00,  // Transparent (not in palette, specified by Graphic
                         // Control Extension), defaults to black.
  };
  for (int i = 0; i < 9; i++) {
    ASSERT_EQ(imgdata[i], expected[i]) << "i=" << i;
  }

  // However, if there is an out-of-palette pixel that is not the transparent
  // index...
  encoded[40] = 0b0'101'011'0;  // The '011' is an out-of-palette color 3.

  // ...decoding the image...
  error_string.clear();
  gif::Decode(encoded, sizeof(encoded), allocate_image_data, &error_string);

  // ...should fail with an error about a color out of range.
  ASSERT_EQ(error_string, "found color index 3 outside of color map range 2");
}

}  // namespace
}  // namespace gif
}  // namespace tensorflow
