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

#include "absl/status/status.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class TextLineReader : public ReaderBase {
 public:
  TextLineReader(const string& node_name, int skip_header_lines, Env* env)
      : ReaderBase(strings::StrCat("TextLineReader '", node_name, "'")),
        skip_header_lines_(skip_header_lines),
        env_(env),
        line_number_(0) {}

  Status OnWorkStartedLocked() override {
    line_number_ = 0;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file_));

    input_buffer_.reset(new io::InputBuffer(file_.get(), kBufferSize));
    for (; line_number_ < skip_header_lines_; ++line_number_) {
      string line_contents;
      Status status = input_buffer_->ReadLine(&line_contents);
      if (absl::IsOutOfRange(status)) {
        // We ignore an end of file error when skipping header lines.
        // We will end up skipping this file.
        return OkStatus();
      }
      TF_RETURN_IF_ERROR(status);
    }
    return OkStatus();
  }

  Status OnWorkFinishedLocked() override {
    input_buffer_.reset(nullptr);
    return OkStatus();
  }

  Status ReadLocked(tstring* key, tstring* value, bool* produced,
                    bool* at_end) override {
    Status status = input_buffer_->ReadLine(value);
    ++line_number_;
    if (status.ok()) {
      *key = strings::StrCat(current_work(), ":", line_number_);
      *produced = true;
      return status;
    }
    if (absl::IsOutOfRange(status)) {  // End of file, advance to the next.
      *at_end = true;
      return OkStatus();
    } else {  // Some other reading error
      return status;
    }
  }

  Status ResetLocked() override {
    line_number_ = 0;
    input_buffer_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

  // TODO(josh11b): Implement serializing and restoring the state.  Need
  // to create TextLineReaderState proto to store ReaderBaseState,
  // line_number_, and input_buffer_->Tell().

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  const int skip_header_lines_;
  Env* const env_;
  int64_t line_number_;
  std::unique_ptr<RandomAccessFile> file_;  // must outlive input_buffer_
  std::unique_ptr<io::InputBuffer> input_buffer_;
};

class TextLineReaderOp : public ReaderOpKernel {
 public:
  explicit TextLineReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int skip_header_lines = -1;
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_header_lines", &skip_header_lines));
    OP_REQUIRES(context, skip_header_lines >= 0,
                errors::InvalidArgument("skip_header_lines must be >= 0 not ",
                                        skip_header_lines));
    Env* env = context->env();
    SetReaderFactory([this, skip_header_lines, env]() {
      return new TextLineReader(name(), skip_header_lines, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TextLineReader").Device(DEVICE_CPU),
                        TextLineReaderOp);
REGISTER_KERNEL_BUILDER(Name("TextLineReaderV2").Device(DEVICE_CPU),
                        TextLineReaderOp);

}  // namespace tensorflow
