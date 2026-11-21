/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/parse_ops.cc.

#include <zlib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"

// NOTE: The way zstd is packaged in TF, we cannot include it as <zstd.h>.
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"  // NOLINT(build/include)

namespace tensorflow {
namespace {

// Maximum decompressed size accepted for every compression type (1 GiB).
// Prevents attacker-controlled inputs ("decompression bombs") from causing
// unbounded memory growth / OOM.
constexpr size_t kMaxDecompressedBytes = 1024ULL * 1024 * 1024;

// zlib's SIMD fast path (`inflate_fast_chunk_`) requires at least 260 bytes of
// available output space (`INFLATE_FAST_MIN_OUTPUT` = 258 max match + 2
// literals) per iteration to avoid falling back to byte-by-byte decoding.
constexpr size_t kInflateFastHeadroom = 260;
constexpr size_t kMaxOutputAllocation =
    kMaxDecompressedBytes + kInflateFastHeadroom;

// Returns the initial output buffer capacity (in bytes) to allocate for a
// compressed input of `input_size` bytes.
size_t InitialOutputCapacity(size_t input_size) {
  if (input_size >= kMaxDecompressedBytes) {
    return kMaxOutputAllocation;
  }
  // Use 1.5x compressed size (+ SIMD headroom) rather than 2x: for 1.0x
  // incompressible inputs where `compressed_size > uncompressed_size`, 2x would
  // leave `final_size < capacity / 2` and trigger a shrink `realloc()` in
  // `tstring::resize_uninitialized(final_size)`.
  return std::min<size_t>(input_size + (input_size >> 1) + kInflateFastHeadroom,
                          kMaxOutputAllocation);
}

// NOLINTBEGIN(misc-include-cleaner): <zlib.h> gets types from <zconf.h>.

// Lazily initializes a `z_stream` on the first non-empty element and resets it
// via `inflateReset` for subsequent elements in the same `Compute()` call,
// avoiding per-element `inflateInit2`/`inflateEnd` heap churn while keeping
// `OpKernel` stateless across concurrent `Compute()` calls.
class ZlibInflateStream {
 public:
  explicit ZlibInflateStream(bool is_gzip) : is_gzip_(is_gzip) {}
  ZlibInflateStream(const ZlibInflateStream&) = delete;
  ZlibInflateStream& operator=(const ZlibInflateStream&) = delete;

  ~ZlibInflateStream() {
    if (initialized_) {
      inflateEnd(&stream_);
    }
  }

  // Decompresses `input` (ZLIB or GZIP, per `is_gzip_`) into `output`,
  // enforcing `kMaxDecompressedBytes`.
  absl::Status Decode(const tstring& input, tstring& output) {
    if (input.empty()) {
      // Preserve `ZlibInputStream` parity (relied upon by TF dataset pipelines
      // that default missing compressed features to `""`): an empty input hits
      // immediate EOF and produces `""` rather than an error.
      output.clear();
      return absl::OkStatus();
    }

    if (absl::Status s = PrepareStream(); !s.ok()) {
      return s;
    }

    // `z_stream::next_in` is non-const `Bytef*` unless `ZLIB_CONST` is set,
    // though `inflate()` never mutates the input buffer.
    stream_.next_in =
        const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    stream_.avail_in = 0;
    size_t remaining_in = input.size();

    size_t total_out = 0;
    output.resize_uninitialized(InitialOutputCapacity(input.size()));

    while (true) {
      if (stream_.avail_in == 0 && remaining_in > 0) {
        const uInt chunk_in = static_cast<uInt>(
            std::min<size_t>(remaining_in, std::numeric_limits<uInt>::max()));
        stream_.avail_in = chunk_in;
        remaining_in -= chunk_in;
      }

      // Keep at least `kInflateFastHeadroom` bytes of output space for each
      // `inflate()` call (including across concatenated gzip members). Since
      // `total_out <= kMaxDecompressedBytes` holds here (checked after every
      // `inflate()`), `output.size() < kMaxOutputAllocation` whenever this
      // branch is taken.
      if (output.size() - total_out < kInflateFastHeadroom) {
        const size_t next_cap = (output.size() > kMaxOutputAllocation / 2)
                                    ? kMaxOutputAllocation
                                    : output.size() * 2;
        output.resize_uninitialized(next_cap);
      }

      const uInt chunk_out = static_cast<uInt>(std::min<size_t>(
          output.size() - total_out, std::numeric_limits<uInt>::max()));
      stream_.next_out = reinterpret_cast<Bytef*>(output.mdata() + total_out);
      stream_.avail_out = chunk_out;

      // Pass `Z_FINISH` once all compressed bytes are queued in `avail_in`.
      // When the full output fits in the initial `avail_out`, `inflate()`
      // reaches `Z_STREAM_END` in one call and skips allocating/updating its
      // 32 KiB sliding window (`state->wsize == 0`). If more output space is
      // needed, it fills `avail_out`, saves the window, and returns
      // `Z_BUF_ERROR` so the loop grows `output`.
      const int flush = (remaining_in == 0) ? Z_FINISH : Z_NO_FLUSH;
      const int err = inflate(&stream_, flush);
      total_out += chunk_out - stream_.avail_out;

      if (total_out > kMaxDecompressedBytes) {
        output.clear();
        return absl::InvalidArgumentError(
            "Decompressed size exceeds 1GB limit");
      }

      const bool has_more_input = (stream_.avail_in > 0 || remaining_in > 0);
      if (err == Z_STREAM_END) {
        if (!is_gzip_ || !has_more_input) {
          break;
        }
        // RFC 1952 permits concatenated gzip members; reset and continue.
        if (absl::Status s = PrepareStream(); !s.ok()) {
          output.clear();
          return s;
        }
        continue;
      }
      if (err == Z_OK) {
        continue;
      }
      if (err == Z_BUF_ERROR) {
        if (!has_more_input && stream_.avail_out > 0) {
          // All compressed input has been consumed and no further output can be
          // produced (e.g. a `Z_SYNC_FLUSH` stream without a final trailer),
          // matching `ZlibInputStream` EOF semantics.
          break;
        }
        continue;
      }

      std::string error_string =
          absl::StrCat("inflate() failed with error ", err);
      if (stream_.msg != nullptr) {
        absl::StrAppend(&error_string, ": ", stream_.msg);
      }
      output.clear();
      return absl::DataLossError(error_string);
    }

    output.resize_uninitialized(total_out);
    return absl::OkStatus();
  }

 private:
  // Initializes `stream_` on first use, or resets it while preserving its
  // `next_in` / `avail_in` cursors for subsequent elements or gzip members.
  absl::Status PrepareStream() {
    if (!initialized_) {
      const int err =
          inflateInit2(&stream_, is_gzip_ ? MAX_WBITS + 16 : MAX_WBITS);
      if (err != Z_OK) {
        return absl::InternalError(
            absl::StrCat("inflateInit2() failed with error ", err));
      }
      initialized_ = true;
      return absl::OkStatus();
    }
    const int err = inflateReset(&stream_);
    if (err != Z_OK) {
      return absl::InternalError(
          absl::StrCat("inflateReset() failed with error ", err));
    }
    return absl::OkStatus();
  }

  const bool is_gzip_;
  bool initialized_ = false;
  z_stream stream_{};
};

// NOLINTEND(misc-include-cleaner)

}  // namespace

constexpr const char kSupportedArgs[] =
    "Only ZLIB, ZSTD, GZIP or NONE are supported compressions";

class DecodeCompressedOp : public OpKernel {
 public:
  explicit DecodeCompressedOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("compression_type", &compression_type_));
    OP_REQUIRES(context,
                (compression_type_.empty() || compression_type_ == "ZLIB" ||
                 compression_type_ == "GZIP" || compression_type_ == "ZSTD"),
                absl::InvalidArgumentError(kSupportedArgs));
  }

  // Do a single decompression of `input` into `output`, using the algorithm
  // specified in `compression_type_`. Return absl::OkStatus() on success, and
  // anything else on failure.
  absl::Status DecompressOne(const tstring& input, tstring& output) {
    if (compression_type_ == "ZSTD") {
      ZSTD_DCtx* decompress_ctx = ZSTD_createDCtx();
      if (decompress_ctx == nullptr) {
        return absl::InternalError("Failed to create zstd context");
      }

      const char* data = input.data();
      size_t len = input.size();

      // NOTE: Using auto here to avoid an explicit unsigned long long and
      // linter complaints.
      auto max_decompressed_size = ZSTD_getFrameContentSize(data, len);

      if (max_decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN ||
          max_decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
        ZSTD_freeDCtx(decompress_ctx);
        return absl::InvalidArgumentError(
            "Failed to determine decompressed size");
      }

      if (max_decompressed_size > kMaxDecompressedBytes) {
        ZSTD_freeDCtx(decompress_ctx);
        return absl::InvalidArgumentError(
            "Decompressed size exceeds 1GB limit");
      }

      // Allocate enough to maximally decompress into.
      output.resize_uninitialized(max_decompressed_size);

      // Do the decompression, and find out how large the result was after all.
      size_t actual_size = ZSTD_decompressDCtx(decompress_ctx, output.mdata(),
                                               output.size(), data, len);

      if (ZSTD_isError(actual_size)) {
        ZSTD_freeDCtx(decompress_ctx);
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to decompress zstd input: ",
                         ZSTD_getErrorName(actual_size)));
      }

      // Trim the output string before we're done.
      output.resize(actual_size);
      ZSTD_freeDCtx(decompress_ctx);
      return absl::OkStatus();
    }

    // We shouldn't get here, but let's just repeat our complaint about which
    // compression algorithms are allowed.
    return absl::InvalidArgumentError(kSupportedArgs);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* bytes_tensor;
    OP_REQUIRES_OK(context, context->input("bytes", &bytes_tensor));
    if (compression_type_.empty()) {
      OP_REQUIRES_OK(context, context->set_output("output", *bytes_tensor));
      return;
    }
    const auto& bytes_flat = bytes_tensor->flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", bytes_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();
    if (compression_type_ == "ZLIB" || compression_type_ == "GZIP") {
      ZlibInflateStream zlib_stream(/*is_gzip=*/compression_type_ == "GZIP");
      for (int64_t i = 0; i < bytes_flat.size(); i++) {
        OP_REQUIRES_OK(context,
                       zlib_stream.Decode(bytes_flat(i), output_flat(i)));
      }
      return;
    }
    for (int64_t i = 0; i < bytes_flat.size(); i++) {
      OP_REQUIRES_OK(context, DecompressOne(bytes_flat(i), output_flat(i)));
    }
  }

 private:
  std::string compression_type_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeCompressed").Device(DEVICE_CPU),
                        DecodeCompressedOp)

}  // namespace tensorflow
