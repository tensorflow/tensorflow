/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class FixedLengthRecordReader : public ReaderBase {
 public:
  FixedLengthRecordReader(const string& node_name, int64 header_bytes,
                          int64 record_bytes, int64 footer_bytes, Env* env)
      : ReaderBase(
            strings::StrCat("FixedLengthRecordReader '", node_name, "'")),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        env_(env),
        file_pos_limit_(-1),
        record_number_(0) {}

  // On success:
  // * input_buffer_ != nullptr,
  // * input_buffer_->Tell() == footer_bytes_
  // * file_pos_limit_ == file size - header_bytes_
  Status OnWorkStartedLocked() override {
    record_number_ = 0;
    uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env_->GetFileSize(current_work(), &file_size));
    file_pos_limit_ = file_size - footer_bytes_;

    RandomAccessFile* file = nullptr;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file));
    input_buffer_.reset(new io::InputBuffer(file, kBufferSize));
    TF_RETURN_IF_ERROR(input_buffer_->SkipNBytes(header_bytes_));
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    input_buffer_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    if (input_buffer_->Tell() >= file_pos_limit_) {
      *at_end = true;
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(input_buffer_->ReadNBytes(record_bytes_, value));
    *key = strings::StrCat(current_work(), ":", record_number_);
    *produced = true;
    ++record_number_;
    return Status::OK();
  }

  Status ResetLocked() override {
    file_pos_limit_ = -1;
    record_number_ = 0;
    input_buffer_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

  // TODO(josh11b): Implement serializing and restoring the state.

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  const int64 header_bytes_;
  const int64 record_bytes_;
  const int64 footer_bytes_;
  Env* const env_;
  int64 file_pos_limit_;
  int64 record_number_;
  std::unique_ptr<io::InputBuffer> input_buffer_;
};

class FixedLengthRecordReaderOp : public ReaderOpKernel {
 public:
  explicit FixedLengthRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int64 header_bytes = -1, record_bytes = -1, footer_bytes = -1;
    OP_REQUIRES_OK(context, context->GetAttr("header_bytes", &header_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("record_bytes", &record_bytes));
    OP_REQUIRES_OK(context, context->GetAttr("footer_bytes", &footer_bytes));
    OP_REQUIRES(context, header_bytes >= 0,
                errors::InvalidArgument("header_bytes must be >= 0 not ",
                                        header_bytes));
    OP_REQUIRES(context, record_bytes >= 0,
                errors::InvalidArgument("record_bytes must be >= 0 not ",
                                        record_bytes));
    OP_REQUIRES(context, footer_bytes >= 0,
                errors::InvalidArgument("footer_bytes must be >= 0 not ",
                                        footer_bytes));
    Env* env = context->env();
    SetReaderFactory([this, header_bytes, record_bytes, footer_bytes, env]() {
      return new FixedLengthRecordReader(name(), header_bytes, record_bytes,
                                         footer_bytes, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordReader").Device(DEVICE_CPU),
                        FixedLengthRecordReaderOp);

}  // namespace tensorflow
