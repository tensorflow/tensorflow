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
#include <memory>
#include <optional>
#include <string>

#include "tensorflow/core/platform/tstring.h"

#if defined(TF_PNG_ENABLE_FPNGE)
#include "fpnge.h"  // from @fpnge
#endif

namespace tensorflow {
namespace png {
namespace {

#if defined(TF_PNG_ENABLE_FPNGE)
// Match PNG_USER_WIDTH_MAX / PNG_USER_HEIGHT_MAX in libpng so
// WriteImageToBuffer never emits PNGs that tensorflow::png::CommonInitDecode
// rejects.
constexpr int kMaxFastPngDim = 1000000;
// Bound total raw image bytes at 256 MiB so multi-hundred-megabyte buffers fall
// back to streaming libpng instead of overflowing fpnge's single 31-bit IDAT
// chunk length or allocating a large contiguous staging buffer.
constexpr size_t kMaxRawImageBytes = 256ULL << 20;
constexpr size_t kMaxThreadLocalScratchBytes = 8ULL << 20;

// Default-initialized (not value-initialized) so the staging buffer is not
// zero-filled; std::make_unique_for_overwrite needs C++20, which open-source
// C++17 builds (ROCm, SYCL) lack.
std::unique_ptr<uint8_t[]> AllocateUninitialized(size_t size) {
  return std::unique_ptr<uint8_t[]>(new uint8_t[size]);
}

// Shared across both std::string and tstring callers so each worker thread
// retains at most one 8 MiB scratch buffer.
uint8_t* GetThreadLocalScratch(size_t alloc_size) {
  struct ScratchBuffer {
    std::unique_ptr<uint8_t[]> ptr;
    size_t capacity = 0;
  };
  thread_local ScratchBuffer tls_scratch;
  if (tls_scratch.capacity < alloc_size) {
    tls_scratch.ptr = AllocateUninitialized(alloc_size);
    tls_scratch.capacity = alloc_size;
  }
  return tls_scratch.ptr.get();
}

// zlib levels fpnge stands in for; nullopt falls back to libpng.
// - Level 0 requests uncompressed Z_NO_COMPRESSION block
// - Levels 7+ explicitly trade ~100x CPU for ~6% smaller LZ77 output
//   (58.1 ms vs 0.20 ms on 720p RGB).
constexpr std::optional<int> ToFpngeLevel(int zlib_level) {
  switch (zlib_level) {
    case 1:
    case 2:
    case 3:
      return zlib_level;
    case -1:
    case 4:
      return FPNGE_COMPRESS_LEVEL_DEFAULT;
    case 5:
    case 6:
      return FPNGE_COMPRESS_LEVEL_BEST;
    default:
      return std::nullopt;
  }
}
static_assert(!ToFpngeLevel(0) && !ToFpngeLevel(7) && !ToFpngeLevel(-2));
static_assert(FPNGE_COMPRESS_LEVEL_DEFAULT != 0 &&
              FPNGE_COMPRESS_LEVEL_BEST != 0);

template <typename T>
bool EncodeWithFpnge(const void* image, int width, int height, int row_bytes,
                     int num_channels, int channel_bits, int compression,
                     T* png_string) {
  // 8-bit only: fpnge copies 16-bit channels in big-endian order without
  // byte-swapping host-endian uint16_t inputs.
  const std::optional<int> fpnge_level = ToFpngeLevel(compression);
  if (image == nullptr || png_string == nullptr || channel_bits != 8 ||
      num_channels < 1 || num_channels > 4 || width <= 0 || height <= 0 ||
      width > kMaxFastPngDim || height > kMaxFastPngDim || !fpnge_level ||
      row_bytes < 0) {
    return false;
  }
  const size_t w = static_cast<size_t>(width);
  const size_t h = static_cast<size_t>(height);
  const size_t c = static_cast<size_t>(num_channels);
  const size_t stride = static_cast<size_t>(row_bytes);
  if (stride < w * c || w * h * c > kMaxRawImageBytes) {
    return false;
  }

  FPNGEOptions options;
  FPNGEFillOptions(&options, *fpnge_level, FPNGE_CICP_NONE);

  const size_t alloc_size =
      FPNGEOutputAllocSize(/*bytes_per_channel=*/1, c, w, h);
  // Reuse thread-local uninitialized scratch <= 8 MiB and copy exact
  // encoded_size (~0.02 ms/MB) to avoid zero-filling alloc_size or retaining
  // ~2x raw-pixel capacity (+1.1 GiB across a 256-frame 720p batch) in output
  // tstrings.
  std::unique_ptr<uint8_t[]> large_scratch;
  uint8_t* dest = nullptr;
  if (alloc_size <= kMaxThreadLocalScratchBytes) {
    dest = GetThreadLocalScratch(alloc_size);
  } else {
    large_scratch = AllocateUninitialized(alloc_size);
    dest = large_scratch.get();
  }

  const size_t encoded_size = FPNGEEncode(
      /*bytes_per_channel=*/1, c, image, w, stride, h, dest, &options);
  if (encoded_size == 0 || encoded_size > alloc_size) {
    return false;
  }
  png_string->assign(reinterpret_cast<const char*>(dest), encoded_size);
  return true;
}
#else
template <typename T>
bool EncodeWithFpnge(const void*, int, int, int, int, int, int, T*) {
  return false;
}
#endif

}  // namespace

bool TryFastEncodePng(const void* image, int width, int height, int row_bytes,
                      int num_channels, int channel_bits, int compression,
                      std::string* png_string) {
  return EncodeWithFpnge(image, width, height, row_bytes, num_channels,
                         channel_bits, compression, png_string);
}

bool TryFastEncodePng(const void* image, int width, int height, int row_bytes,
                      int num_channels, int channel_bits, int compression,
                      tstring* png_string) {
  return EncodeWithFpnge(image, width, height, row_bytes, num_channels,
                         channel_bits, compression, png_string);
}

}  // namespace png
}  // namespace tensorflow
