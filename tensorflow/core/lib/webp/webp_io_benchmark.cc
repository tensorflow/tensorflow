/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "third_party/libwebp/src/webp/encode.h"
#include "third_party/libwebp/src/webp/mux.h"  // Needed for Animation Encoding
#include "third_party/libwebp/src/webp/types.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "tensorflow/core/lib/webp/webp_io.h"

namespace tensorflow {
namespace webp {
namespace {

// Helper to generate a large 4K image in memory and compress it to WebP.
std::string GetLargeWebPBytes(int width, int height) {
  // Create a dummy buffer
  const int stride = width * 3;
  std::vector<uint8_t> raw_data(stride * height);
  for (size_t i = 0; i < raw_data.size(); ++i) raw_data[i] = i % 255;

  uint8_t* output;
  size_t size =
      WebPEncodeRGB(raw_data.data(), width, height, stride, 80.0f, &output);
  std::string result(reinterpret_cast<char*>(output), size);
  WebPFree(output);
  return result;
}

// Helper to generate a simple WebP Animation (5 frames of 1024x1024)
std::string GetWebPAnimationBytes(int width, int height, int frames) {
  WebPAnimEncoderOptions enc_options;
  WebPAnimEncoderOptionsInit(&enc_options);
  WebPAnimEncoder* enc = WebPAnimEncoderNew(width, height, &enc_options);

  const int stride = width * 4;  // RGBA required for animation
  std::vector<uint8_t> raw_data(stride * height);

  // Create 'frames' frames
  for (int f = 0; f < frames; ++f) {
    // Modify data slightly so frames differ
    for (size_t i = 0; i < raw_data.size(); ++i) raw_data[i] = (i + f) % 255;

    WebPConfig config;
    WebPConfigInit(&config);
    WebPPicture pic;
    WebPPictureInit(&pic);
    pic.width = width;
    pic.height = height;
    pic.use_argb = 1;
    WebPPictureImportRGBA(&pic, raw_data.data(), stride);

    WebPAnimEncoderAdd(enc, &pic, f * 100, &config);
    WebPPictureFree(&pic);
  }
  WebPAnimEncoderAdd(enc, nullptr, frames * 100, nullptr);  // End marker

  WebPData webp_data;
  WebPAnimEncoderAssemble(enc, &webp_data);
  std::string result(reinterpret_cast<const char*>(webp_data.bytes),
                     webp_data.size);

  WebPAnimEncoderDelete(enc);
  WebPDataClear(&webp_data);
  return result;
}

// Benchmark 1: Single Image Decoding (4K)
// This stresses memory bandwidth and vectorization much more than tiny test
// files.
static void BM_DecodeImage(benchmark::State& state) {
  // Setup: Generate a 4K image once
  const int w_target = 3840;
  const int h_target = 2160;
  const int channels = state.range(0);  // Arg: 3 for RGB, 4 for RGBA

  static const std::string* large_webp =
      new std::string(GetLargeWebPBytes(w_target, h_target));

  // Allocation outside the loop to measure pure decode speed
  std::vector<uint8_t> output(w_target * h_target * channels);

  for (auto _ : state) {
    bool success = DecodeWebPImage(*large_webp, output.data(), w_target,
                                   h_target, channels);
    CHECK(success);
  }

  // Metric: Pixels per second (helps compare across image sizes)
  state.SetItemsProcessed(state.iterations() * w_target * h_target);
}
BENCHMARK(BM_DecodeImage)->Arg(3)->Arg(4);  // Test both RGB and RGBA

// Benchmark 2: Animation Decoding
static void BM_DecodeAnimation(benchmark::State& state) {
  const int width = 1024;
  const int height = 1024;
  const int frames = 5;
  // const int channels = 4; // Animation is always RGBA in libwebp output

  static const std::string* anim_webp =
      new std::string(GetWebPAnimationBytes(width, height, frames));

  // Allocator for the API
  auto allocator = [](int num_frames, int w, int h, int c) {
    return new uint8_t[num_frames * w * h * c];
  };

  for (auto _ : state) {
    std::string error;
    uint8_t* out = DecodeWebPAnimation(*anim_webp, allocator, &error, true);
    CHECK(out != nullptr);
    delete[] out;  // Cleanup is part of the loop since we alloc each time.
  }

  state.SetItemsProcessed(state.iterations() * width * height * frames);
}
BENCHMARK(BM_DecodeAnimation);

}  // namespace
}  // namespace webp
}  // namespace tensorflow
