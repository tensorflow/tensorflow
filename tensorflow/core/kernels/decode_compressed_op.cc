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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace {
// Wrap memory buffer into InputStreamInterface
class MemoryInputStream : public io::InputStreamInterface {
 public:
  explicit MemoryInputStream(const char* buffer, size_t length)
      : buf_(buffer), len_(length), pos_(0) {}

  ~MemoryInputStream() override {}

  Status ReadNBytes(int64_t bytes_to_read, tstring* result) override {
    result->clear();
    if (bytes_to_read < 0) {
      return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                     bytes_to_read);
    }
    int64_t bytes = bytes_to_read;
    Status s = absl::OkStatus();
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

  Status Reset() override {
    pos_ = 0;
    return absl::OkStatus();
  }

 private:
  const char* buf_;  // Not owned.
  int64_t len_;
  int64_t pos_ = 0;  // Tracks where we are in the file.
};
}  // namespace

class DecodeCompressedOp : public OpKernel {
 public:
  explicit DecodeCompressedOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("compression_type", &compression_type_));
    OP_REQUIRES(context,
                (compression_type_.empty() || compression_type_ == "ZLIB" ||
                 compression_type_ == "GZIP"),
                errors::InvalidArgument(
                    "Only ZLIB, GZIP or NONE are supported compressions"));
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
    if (compression_type_.empty()) {
      for (int64_t i = 0; i < bytes_flat.size(); i++) {
        output_flat(i) = bytes_flat(i);
      }
    } else {
      const io::ZlibCompressionOptions zlib_options =
          compression_type_ == "ZLIB" ? io::ZlibCompressionOptions::DEFAULT()
                                      : io::ZlibCompressionOptions::GZIP();
      for (int64_t i = 0; i < bytes_flat.size(); i++) {
        std::unique_ptr<MemoryInputStream> input_stream(
            new MemoryInputStream(bytes_flat(i).data(), bytes_flat(i).size()));
        std::unique_ptr<io::ZlibInputStream> zlib_stream(
            new io::ZlibInputStream(
                input_stream.get(), static_cast<size_t>(kBufferSize),
                static_cast<size_t>(kBufferSize), zlib_options));
        tstring output_string;
        Status s = zlib_stream->ReadNBytes(INT_MAX, &output_string);
        OP_REQUIRES(context, (s.ok() || errors::IsOutOfRange(s)), s);
        output_flat(i) = std::move(output_string);
      }
    }
  }

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  string compression_type_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeCompressed").Device(DEVICE_CPU),
                        DecodeCompressedOp)

}  // namespace tensorflow
