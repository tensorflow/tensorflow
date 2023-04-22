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
        return new uint8[frame_cnt * height * width * channels];
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
       {testdata_path + "squares.gif", 2, 16, 16, 3}});

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

}  // namespace
}  // namespace gif
}  // namespace tensorflow
