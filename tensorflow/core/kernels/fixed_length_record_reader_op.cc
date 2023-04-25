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

// See docs in ../ops/io_ops.cc.

#include <memory>
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// In the constructor hop_bytes_ is set to record_bytes_ if it was 0,
// so that we will always "hop" after each read (except first).
class FixedLengthRecordReader : public ReaderBase {
 public:
  FixedLengthRecordReader(const string& node_name, int64_t header_bytes,
                          int64_t record_bytes, int64_t footer_bytes,
                          int64_t hop_bytes, const string& encoding, Env* env)
      : ReaderBase(
            strings::StrCat("FixedLengthRecordReader '", node_name, "'")),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        hop_bytes_(hop_bytes == 0 ? record_bytes : hop_bytes),
        env_(env),
        record_number_(0),
        encoding_(encoding) {}

  // On success:
  // * buffered_inputstream_ != nullptr,
  // * buffered_inputstream_->Tell() == header_bytes_
  Status OnWorkStartedLocked() override {
    record_number_ = 0;

    lookahead_cache_.clear();

    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file_));
    if (encoding_ == "ZLIB" || encoding_ == "GZIP") {
      const io::ZlibCompressionOptions zlib_options =
          encoding_ == "ZLIB" ? io::ZlibCompressionOptions::DEFAULT()
                              : io::ZlibCompressionOptions::GZIP();
      file_stream_.reset(new io::RandomAccessInputStream(file_.get()));
      buffered_inputstream_.reset(new io::ZlibInputStream(
          file_stream_.get(), static_cast<size_t>(kBufferSize),
          static_cast<size_t>(kBufferSize), zlib_options));
    } else {
      buffered_inputstream_.reset(
          new io::BufferedInputStream(file_.get(), kBufferSize));
    }
    // header_bytes_ is always skipped.
    TF_RETURN_IF_ERROR(buffered_inputstream_->SkipNBytes(header_bytes_));

    return OkStatus();
  }

  Status OnWorkFinishedLocked() override {
    buffered_inputstream_.reset(nullptr);
    return OkStatus();
  }

  Status ReadLocked(tstring* key, tstring* value, bool* produced,
                    bool* at_end) override {
    // We will always "hop" the hop_bytes_ except the first record
    // where record_number_ == 0
    if (record_number_ != 0) {
      if (hop_bytes_ <= lookahead_cache_.size()) {
        // If hop_bytes_ is smaller than the cached data we skip the
        // hop_bytes_ from the cache.
        lookahead_cache_ = lookahead_cache_.substr(hop_bytes_);
      } else {
        // If hop_bytes_ is larger than the cached data, we clean up
        // the cache, then skip hop_bytes_ - cache_size from the file
        // as the cache_size has been skipped through cache.
        int64_t cache_size = lookahead_cache_.size();
        lookahead_cache_.clear();
        Status s = buffered_inputstream_->SkipNBytes(hop_bytes_ - cache_size);
        if (!s.ok()) {
          if (!errors::IsOutOfRange(s)) {
            return s;
          }
          *at_end = true;
          return OkStatus();
        }
      }
    }

    // Fill up lookahead_cache_ to record_bytes_ + footer_bytes_
    int bytes_to_read = record_bytes_ + footer_bytes_ - lookahead_cache_.size();
    Status s = buffered_inputstream_->ReadNBytes(bytes_to_read, value);
    if (!s.ok()) {
      value->clear();
      if (!errors::IsOutOfRange(s)) {
        return s;
      }
      *at_end = true;
      return OkStatus();
    }
    lookahead_cache_.append(*value, 0, bytes_to_read);
    value->clear();

    // Copy first record_bytes_ from cache to value
    *value = lookahead_cache_.substr(0, record_bytes_);

    *key = strings::StrCat(current_work(), ":", record_number_);
    *produced = true;
    ++record_number_;

    return OkStatus();
  }

  Status ResetLocked() override {
    record_number_ = 0;
    buffered_inputstream_.reset(nullptr);
    lookahead_cache_.clear();
    return ReaderBase::ResetLocked();
  }

  // TODO(joshl): Implement serializing and restoring the state.

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  const int64_t header_bytes_;
  const int64_t record_bytes_;
  const int64_t footer_bytes_;
  const int64_t hop_bytes_;
  // The purpose of lookahead_cache_ is to allows "one-pass" processing
  // without revisit previous processed data of the stream. This is needed
  // because certain compression like zlib does not allow random access
  // or even obtain the uncompressed stream size before hand.
  // The max size of the lookahead_cache_ could be
  // record_bytes_ + footer_bytes_
  string lookahead_cache_;
  Env* const env_;
  int64_t record_number_;
  string encoding_;
  // must outlive buffered_inputstream_
  std::unique_ptr<RandomAccessFile> file_;
  // must outlive buffered_inputstream_
  std::unique_ptr<io::RandomAccessInputStream> file_stream_;
  std::unique_ptr<io::InputStreamInterface> buffered_inputstream_;
};

class FixedLengthRecordReaderOp : public ReaderOpKernel {
 public:
  explicit FixedLengthRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int64_t header_bytes = -1, record_bytes = -1, footer_bytes = -1,
            hop_bytes = -1;
    OP_REQUIRES_OK(context, context->GetAttr("header_bytes", &header_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("record_bytes", &record_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("footer_bytes", &footer_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("hop_bytes", &hop_bytes));
    OP_REQUIRES(context, header_bytes >= 0,
                errors::InvalidArgument("header_bytes must be >= 0 not ",
                                        header_bytes));
    OP_REQUIRES(context, record_bytes >= 0,
                errors::InvalidArgument("record_bytes must be >= 0 not ",
                                        record_bytes));
    OP_REQUIRES(context, footer_bytes >= 0,
                errors::InvalidArgument("footer_bytes must be >= 0 not ",
                                        footer_bytes));
    OP_REQUIRES(
        context, hop_bytes >= 0,
        errors::InvalidArgument("hop_bytes must be >= 0 not ", hop_bytes));
    Env* env = context->env();
    string encoding;
    OP_REQUIRES_OK(context, context->GetAttr("encoding", &encoding));
    SetReaderFactory([this, header_bytes, record_bytes, footer_bytes, hop_bytes,
                      encoding, env]() {
      return new FixedLengthRecordReader(name(), header_bytes, record_bytes,
                                         footer_bytes, hop_bytes, encoding,
                                         env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReader").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);
REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReaderV2").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);

}  // namespace tensorflow
