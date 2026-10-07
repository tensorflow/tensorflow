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
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "third_party/libwebp/src/webp/encode.h"
#include "third_party/libwebp/src/webp/types.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "tensorflow/core/lib/webp/webp_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace webp {
namespace {

// Helper to generate a large 4K image in memory and compress it to WebP.
// We use synthetic 4K images because the real testdata is too small to
// demonstrate the benefits of multi-threaded decoding.
std::string GenerateSyntheticWebP(int width, int height, bool lossless,
                                  bool high_entropy) {
  const int stride = width * 3;
  std::vector<uint8_t> raw_data(stride * height);
  if (high_entropy) {
    // Use standard minstd_rand to fill the image with high-entropy noise
    // when high_entropy is true, simulating difficult-to-compress inputs.
    std::minstd_rand prng(12345);
    for (size_t i = 0; i < raw_data.size(); ++i) {
      raw_data[i] = prng() & 0xFF;
    }
  } else {
    for (size_t i = 0; i < raw_data.size(); ++i) {
      raw_data[i] = i % 255;
    }
  }

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
static void BM_DecodeLargeImage(benchmark::State& state, bool lossless,
                                bool use_threads, bool high_entropy) {
  const int w_target = 3840;
  const int h_target = 2160;
  const int channels = 3;

  const std::string webp_data =
      GenerateSyntheticWebP(w_target, h_target, lossless, high_entropy);

  std::vector<uint8_t> output(w_target * h_target * channels);

  for (auto _ : state) {
    CHECK(DecodeWebPImage(webp_data, output.data(), w_target, h_target,
                          channels, use_threads));
  }
  state.SetItemsProcessed(state.iterations() * w_target * h_target);
  state.counters["webp_size_bytes"] = webp_data.size();
}
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossy_ST_LowEntropy, /*lossless=*/false,
                  /*use_threads=*/false, /*high_entropy=*/false);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossy_ST_HighEntropy, /*lossless=*/false,
                  /*use_threads=*/false, /*high_entropy=*/true);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossy_MT_LowEntropy, /*lossless=*/false,
                  /*use_threads=*/true, /*high_entropy=*/false);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossy_MT_HighEntropy, /*lossless=*/false,
                  /*use_threads=*/true, /*high_entropy=*/true);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossless_ST_LowEntropy,
                  /*lossless=*/true, /*use_threads=*/false,
                  /*high_entropy=*/false);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossless_ST_HighEntropy,
                  /*lossless=*/true, /*use_threads=*/false,
                  /*high_entropy=*/true);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossless_MT_LowEntropy,
                  /*lossless=*/true, /*use_threads=*/true,
                  /*high_entropy=*/false);
BENCHMARK_CAPTURE(BM_DecodeLargeImage, Lossless_MT_HighEntropy,
                  /*lossless=*/true, /*use_threads=*/true,
                  /*high_entropy=*/true);

static void BM_DecodeTestdataImage(benchmark::State& state, int file_idx,
                                   bool use_threads) {
  const std::vector<std::string> filenames = {
      "lossless_raw.webp",
      "RGB_noise_large_pixels_115x115.webp",
      "lossy_alpha1.webp",
  };
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
  state.counters["webp_size_bytes"] = webp_data.size();
}
BENCHMARK_CAPTURE(BM_DecodeTestdataImage, lossless_raw_ST, /*file_idx=*/0,
                  /*use_threads=*/false);
BENCHMARK_CAPTURE(BM_DecodeTestdataImage, lossless_raw_MT, /*file_idx=*/0,
                  /*use_threads=*/true);
BENCHMARK_CAPTURE(BM_DecodeTestdataImage, RGB_noise_ST, /*file_idx=*/1,
                  /*use_threads=*/false);
BENCHMARK_CAPTURE(BM_DecodeTestdataImage, RGB_noise_MT, /*file_idx=*/1,
                  /*use_threads=*/true);
BENCHMARK_CAPTURE(BM_DecodeTestdataImage, lossy_alpha1_ST, /*file_idx=*/2,
                  /*use_threads=*/false);
BENCHMARK_CAPTURE(BM_DecodeTestdataImage, lossy_alpha1_MT, /*file_idx=*/2,
                  /*use_threads=*/true);

static void BM_DecodeAnimation(benchmark::State& state, bool use_threads) {
  const std::string filename = "bouncy_ball.webp";
  std::string file_path = GetDataDependencyFilepath(
      io::JoinPath("tensorflow/core/lib/webp/testdata", filename));

  std::string webp_data;
  CHECK_OK(ReadFileToString(Env::Default(), file_path, &webp_data));

  int width, height, channels;
  bool has_animation;
  CHECK(
      DecodeWebPHeader(webp_data, &width, &height, &channels, &has_animation));
  CHECK(has_animation);

  std::vector<uint8_t> output_buffer;
  int num_frames = 0;
  auto allocate_output = [&output_buffer, &num_frames](int nf, int w, int h,
                                                       int c) -> uint8_t* {
    num_frames = nf;
    size_t size = nf * w * h * c;
    if (output_buffer.size() < size) {
      output_buffer.resize(size);
    }
    return output_buffer.data();
  };

  std::string error_string;
  for (auto _ : state) {
    uint8_t* output =
        DecodeWebPAnimation(webp_data, allocate_output, &error_string,
                            /*expand_animations=*/true, use_threads);
    CHECK(output != nullptr) << "Error: " << error_string;
  }
  state.SetItemsProcessed(state.iterations() * width * height * num_frames);
  state.counters["webp_size_bytes"] = webp_data.size();
}
BENCHMARK_CAPTURE(BM_DecodeAnimation, bouncy_ball_ST, /*use_threads=*/false);
BENCHMARK_CAPTURE(BM_DecodeAnimation, bouncy_ball_MT, /*use_threads=*/true);

// Safety/Correctness tests for Sanitizers
TEST(WebPIO, DecodeLargeImageCorrectness) {
  const int w = 2048;
  const int h = 2048;  // 4MP, exceeds 1MP heuristic
  std::string data = GenerateSyntheticWebP(w, h, false, /*high_entropy=*/false);

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
  tsl::testing::InitializeBenchmarks(&argc, argv);
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
