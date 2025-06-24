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

#include <algorithm>
#include <climits>

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

// NOTE: The way zstd is packaged in TF, we cannot include it as <zstd.h>.
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"

namespace tensorflow {
namespace {
// Wrap memory buffer into InputStreamInterface
class MemoryInputStream : public io::InputStreamInterface {
 public:
  explicit MemoryInputStream(const char* buffer, size_t length)
      : buf_(buffer), len_(length), pos_(0) {}

  ~MemoryInputStream() override {}

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    result->clear();
    if (bytes_to_read < 0) {
      return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                     bytes_to_read);
    }
    int64_t bytes = bytes_to_read;
    absl::Status s = absl::OkStatus();
    if (pos_ + bytes_to_read > len_) {
      bytes = len_ - pos_;
      s = errors::OutOfRange("reached end of file");
    }
    if (bytes > 0) {
      result->resize(bytes);
      memcpy(&(*result)[0], &buf_[pos_], bytes);
      pos_ += bytes;
    }
    return s;
  }

  int64_t Tell() const override { return pos_; }

  absl::Status Reset() override {
    pos_ = 0;
    return absl::OkStatus();
  }

 private:
  const char* buf_;  // Not owned.
  int64_t len_;
  int64_t pos_ = 0;  // Tracks where we are in the file.
};

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
                errors::InvalidArgument(kSupportedArgs));
  }

  // Do a single decompression of `input` into `output`, using the algorithm
  // specified in `compression_type_`. Return absl::OkStatus() on success, and
  // anything else on failure.
  absl::Status DecompressOne(const tstring& input, tstring& output) {
    if (compression_type_.empty()) {
      output = input;
      return absl::OkStatus();
    }

    if (compression_type_ == "ZLIB" || compression_type_ == "GZIP") {
      // TODO: Don't use ZlibInputStream, since we're just going to read the
      // entire thing into memory.
      MemoryInputStream input_stream(input.data(), input.size());

      const io::ZlibCompressionOptions zlib_options =
          compression_type_ == "ZLIB" ? io::ZlibCompressionOptions::DEFAULT()
                                      : io::ZlibCompressionOptions::GZIP();
      io::ZlibInputStream zlib_stream(
          &input_stream,
          /*input_buffer_bytes=*/static_cast<size_t>(kBufferSize),
          /*output_buffer_bytes=*/static_cast<size_t>(kBufferSize),
          zlib_options);

      absl::Status result = zlib_stream.ReadNBytes(INT_MAX, &output);

      // ReadNBytes returns OutOfRange for EOF. Swallow it and return OkStatus,
      // since we're reading INT_MAX bytes anyway.
      if (absl::IsOutOfRange(result)) return absl::OkStatus();

      return result;
    }

    if (compression_type_ == "ZSTD") {
      ZSTD_DCtx* decompress_ctx = ZSTD_createDCtx();
      if (decompress_ctx == nullptr) {
        return errors::Internal("Failed to create zstd context");
      }

      const char* data = input.data();
      size_t len = input.size();

      // NOTE: Using auto here to avoid an explicit unsigned long long and
      // linter complaints.
      auto max_decompressed_size = ZSTD_getFrameContentSize(data, len);

      if (max_decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN ||
          max_decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
        ZSTD_freeDCtx(decompress_ctx);
        return errors::InvalidArgument("Failed to determine decompressed size");
      }

      // Allocate enough to maximally decompress into.
      output.resize_uninitialized(max_decompressed_size);

      // Do the decompression, and find out how large the result was after all.
      size_t actual_size = ZSTD_decompressDCtx(decompress_ctx, output.mdata(),
                                               output.size(), data, len);

      if (ZSTD_isError(actual_size)) {
        ZSTD_freeDCtx(decompress_ctx);
        return errors::InvalidArgument("Failed to decompress zstd input: ",
                                       ZSTD_getErrorName(actual_size));
      }

      // Trim the output string before we're done.
      output.resize(actual_size);
      ZSTD_freeDCtx(decompress_ctx);
      return absl::OkStatus();
    }

    // We shouldn't get here, but let's just repeat our complaint about which
    // compression algorithms are allowed.
    return errors::InvalidArgument(kSupportedArgs);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* bytes_tensor;
    OP_REQUIRES_OK(context, context->input("bytes", &bytes_tensor));
    const auto& bytes_flat = bytes_tensor->flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", bytes_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();
    for (int64_t i = 0; i < bytes_flat.size(); i++) {
      OP_REQUIRES_OK(context, DecompressOne(bytes_flat(i), output_flat(i)));
    }
  }

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  string compression_type_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeCompressed").Device(DEVICE_CPU),
                        DecodeCompressedOp)

}  // namespace tensorflow
