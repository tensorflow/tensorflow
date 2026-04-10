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
#include "third_party/libwebp/src/webp/types.h"
#include "tensorflow/core/lib/webp/webp_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace webp {
namespace {

// Helper to generate a large 4K image in memory and compress it to WebP.
// We use synthetic 4K images because the real testdata is too small to
// demonstrate the benefits of multi-threaded decoding.
std::string GenerateSyntheticWebP(int width, int height, bool lossless) {
  const int stride = width * 3;
  std::vector<uint8_t> raw_data(stride * height);
  for (size_t i = 0; i < raw_data.size(); ++i) raw_data[i] = i % 255;

  uint8_t* output;
  size_t size;
  if (lossless) {
    size =
        WebPEncodeLosslessRGB(raw_data.data(), width, height, stride, &output);
  } else {
    size =
        WebPEncodeRGB(raw_data.data(), width, height, stride, 80.0f, &output);
  }
  std::string result(reinterpret_cast<char*>(output), size);
  WebPFree(output);
  return result;
}

// Benchmark: Single Image Decoding (4K)
// Stresses memory bandwidth and vectorization.
static void BM_DecodeLargeImage(benchmark::State& state) {
  const int w_target = 3840;
  const int h_target = 2160;
  const bool lossless = state.range(0);
  const bool use_threads = state.range(1);
  const int channels = 3;

  const std::string webp_data =
      GenerateSyntheticWebP(w_target, h_target, lossless);

  std::vector<uint8_t> output(w_target * h_target * channels);

  for (auto _ : state) {
    CHECK(DecodeWebPImage(webp_data, output.data(), w_target, h_target,
                          channels, use_threads));
  }
  state.SetItemsProcessed(state.iterations() * w_target * h_target);
}
BENCHMARK(BM_DecodeLargeImage)
    ->Args({0, 0})   // Lossy, single-threaded
    ->Args({0, 1})   // Lossy, multi-threaded
    ->Args({1, 0})   // Lossless, single-threaded
    ->Args({1, 1});  // Lossless, multi-threaded

static void BM_DecodeTestdataImage(benchmark::State& state) {
  const std::vector<std::string> filenames = {
      "lossless_raw.webp",
      "RGB_noise_large_pixels_115x115.webp",
      "lossy_alpha1.webp",
  };
  const int file_idx = state.range(0);
  const bool use_threads = state.range(1);
  const std::string filename = filenames[file_idx];
  std::string file_path = GetDataDependencyFilepath(
      io::JoinPath("tensorflow/core/lib/webp/testdata", filename));

  std::string webp_data;
  CHECK_OK(ReadFileToString(Env::Default(), file_path, &webp_data));

  int width, height, channels;
  bool has_animation;
  CHECK(
      DecodeWebPHeader(webp_data, &width, &height, &channels, &has_animation));

  std::vector<uint8_t> output(width * height * channels);

  for (auto _ : state) {
    CHECK(DecodeWebPImage(webp_data, output.data(), width, height, channels,
                          use_threads));
  }
  state.SetItemsProcessed(state.iterations() * width * height);
}
BENCHMARK(BM_DecodeTestdataImage)
    ->Args({0, 0})
    ->Args({0, 1})
    ->Args({1, 0})
    ->Args({1, 1})
    ->Args({2, 0})
    ->Args({2, 1});

// Safety/Correctness tests for Sanitizers
TEST(WebPIO, DecodeLargeImageCorrectness) {
  const int w = 2048;
  const int h = 2048;  // 4MP, exceeds 1MP heuristic
  std::string data = GenerateSyntheticWebP(w, h, false);

  int width, height, channels;
  bool has_animation;
  ASSERT_TRUE(
      DecodeWebPHeader(data, &width, &height, &channels, &has_animation));
  EXPECT_EQ(width, w);
  EXPECT_EQ(height, h);

  std::vector<uint8_t> output(width * height * channels);
  EXPECT_TRUE(DecodeWebPImage(data, output.data(), width, height, channels,
                              /*use_threads=*/true));
}

TEST(WebPIO, DecodeTestdataCorrectness) {
  std::string file_path = GetDataDependencyFilepath(
      "tensorflow/core/lib/webp/testdata/lossy_alpha1.webp");
  std::string contents;
  ASSERT_THAT(ReadFileToString(Env::Default(), file_path, &contents),
              absl_testing::IsOk());

  int width, height, channels;
  bool has_animation;
  ASSERT_TRUE(
      DecodeWebPHeader(contents, &width, &height, &channels, &has_animation));

  std::vector<uint8_t> output(width * height * channels);
  EXPECT_TRUE(DecodeWebPImage(contents, output.data(), width, height, channels,
                              /*use_threads=*/true));
}

}  // namespace
}  // namespace webp
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Allow running benchmarks if requested, otherwise just tests.
  // Use a positional argument to avoid conflict with the flag parser.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "run_benchmarks") {
      tsl::testing::RunBenchmarks();
      break;
    }
  }
  return RUN_ALL_TESTS();
}
