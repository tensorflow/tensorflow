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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class FixedLengthRecordReader : public ReaderBase {
 public:
  FixedLengthRecordReader(const string& node_name, int64 header_bytes,
                          int64 record_bytes, int64 footer_bytes,
                          int64 hop_bytes, Env* env)
      : ReaderBase(
            strings::StrCat("FixedLengthRecordReader '", node_name, "'")),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        hop_bytes_(hop_bytes),
        env_(env),
        file_pos_limit_(-1),
        record_number_(0) {}

  // On success:
  // * buffered_inputstream_ != nullptr,
  // * buffered_inputstream_->Tell() == header_bytes_
  // * file_pos_limit_ == file size - footer_bytes_
  Status OnWorkStartedLocked() override {
    record_number_ = 0;
    uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env_->GetFileSize(current_work(), &file_size));
    file_pos_limit_ = file_size - footer_bytes_;

    lookahead_cache_.clear();

    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file_));

    buffered_inputstream_.reset(
        new io::BufferedInputStream(file_.get(), kBufferSize));
    TF_RETURN_IF_ERROR(buffered_inputstream_->SkipNBytes(header_bytes_));
    // In case hop_bytes_ is in between 0 and record_bytes_,
    // we will need to hold a cache so that later on we could prefix the cache
    // with the remaining data to read:
    // For example, assume record_bytes is 2 and hop_bytes is 1,
    // For file "H123",
    // We will process the data in the following way:
    // 1. Read hope_bytes of 1 ("1") and save it in hop cache.
    // 2. Read record_bytes - hop_bytes = 1 ("2")
    // 3. Prefix data in 2. with hop cache in one, we have "12",
    //    this is used for the record.
    // 4. Shift hop_bytes of 1 from the record, and we put "2" to hop cache.
    // 5. Continue step 2.
    // In order to acheive the above, we will peek in "record_bytes_ -
    // hop_bytes_"
    // before we read the first record.
    if (0 < hop_bytes_ && hop_bytes_ < record_bytes_) {
      TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(
          record_bytes_ - hop_bytes_, &lookahead_cache_));
    }
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    buffered_inputstream_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    // In case hop_bytes_ is in between 0 and record_bytes_,
    // we will need to hold a cache so that later on we could prefix the cache
    // with the remaining data to read:
    // For example, assume record_bytes is 2 and hop_bytes is 1,
    // For file "H123",
    // We will process the data in the following way:
    // 1. Read hope_bytes of 1 ("1") and save it in hop cache.
    // 2. Read record_bytes - hop_bytes = 1 ("2")
    // 3. Prefix data in 2. with hop cache in one, we have "12",
    //    this is used for the record.
    // 4. Shift hop_bytes of 1 from the record, and we put "2" to hop cache.
    // 5. Continue step 2.
    // In order to acheive the above, in the following only
    // 'record_bytes_ - lookahead_cache_.size()' needs to be read. The
    // lookahead_cache_
    // is then prefixed to piece together the whole record.

    int bytes_to_read = record_bytes_;
    if (0 < hop_bytes_ && hop_bytes_ < record_bytes_) {
      bytes_to_read = hop_bytes_;
    }
    if (buffered_inputstream_->Tell() >= file_pos_limit_ ||
        buffered_inputstream_->Tell() + bytes_to_read > file_pos_limit_) {
      *at_end = true;
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(bytes_to_read, value));
    if (0 < hop_bytes_ && hop_bytes_ < record_bytes_) {
      lookahead_cache_.append(*value, 0, bytes_to_read);
      *value = lookahead_cache_;
      lookahead_cache_ = lookahead_cache_.substr(bytes_to_read);
    }
    *key = strings::StrCat(current_work(), ":", record_number_);
    *produced = true;
    ++record_number_;

    if (hop_bytes_ > record_bytes_) {
      buffered_inputstream_->SkipNBytes(hop_bytes_ - record_bytes_);
    }

    return Status::OK();
  }

  Status ResetLocked() override {
    file_pos_limit_ = -1;
    record_number_ = 0;
    buffered_inputstream_.reset(nullptr);
    lookahead_cache_.clear();
    return ReaderBase::ResetLocked();
  }

  // TODO(josh11b): Implement serializing and restoring the state.

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  const int64 header_bytes_;
  const int64 record_bytes_;
  const int64 footer_bytes_;
  const int64 hop_bytes_;
  string lookahead_cache_;
  Env* const env_;
  int64 file_pos_limit_;
  int64 record_number_;
  // must outlive buffered_inputstream_
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::InputStreamInterface> buffered_inputstream_;
};

class FixedLengthRecordReaderOp : public ReaderOpKernel {
 public:
  explicit FixedLengthRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int64 header_bytes = -1, record_bytes = -1, footer_bytes = -1,
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
    SetReaderFactory(
        [this, header_bytes, record_bytes, footer_bytes, hop_bytes, env]() {
          return new FixedLengthRecordReader(name(), header_bytes, record_bytes,
                                             footer_bytes, hop_bytes, env);
        });
  }
};

REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReader").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);
REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReaderV2").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);

}  // namespace tensorflow
