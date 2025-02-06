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
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class TFRecordReader : public ReaderBase {
 public:
  TFRecordReader(const string& node_name, const string& compression_type,
                 Env* env)
      : ReaderBase(strings::StrCat("TFRecordReader '", node_name, "'")),
        env_(env),
        offset_(0),
        compression_type_(compression_type) {}

  absl::Status OnWorkStartedLocked() override {
    offset_ = 0;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file_));

    io::RecordReaderOptions options =
        io::RecordReaderOptions::CreateRecordReaderOptions(compression_type_);
    reader_.reset(new io::RecordReader(file_.get(), options));
    return absl::OkStatus();
  }

  absl::Status OnWorkFinishedLocked() override {
    reader_.reset(nullptr);
    file_.reset(nullptr);
    return absl::OkStatus();
  }

  absl::Status ReadLocked(tstring* key, tstring* value, bool* produced,
                          bool* at_end) override {
    *key = strings::StrCat(current_work(), ":", offset_);
    absl::Status status = reader_->ReadRecord(&offset_, value);
    if (absl::IsOutOfRange(status)) {
      *at_end = true;
      return absl::OkStatus();
    }
    if (!status.ok()) return status;
    *produced = true;
    return absl::OkStatus();
  }

  absl::Status ResetLocked() override {
    offset_ = 0;
    reader_.reset(nullptr);
    file_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

  // TODO(josh11b): Implement serializing and restoring the state.

 private:
  Env* const env_;
  uint64 offset_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::RecordReader> reader_;
  string compression_type_ = "";
};

class TFRecordReaderOp : public ReaderOpKernel {
 public:
  explicit TFRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();

    string compression_type;
    OP_REQUIRES_OK(context,
                   context->GetAttr("compression_type", &compression_type));

    SetReaderFactory([this, compression_type, env]() {
      return new TFRecordReader(name(), compression_type, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TFRecordReader").Device(DEVICE_CPU),
                        TFRecordReaderOp);
REGISTER_KERNEL_BUILDER(Name("TFRecordReaderV2").Device(DEVICE_CPU),
                        TFRecordReaderOp);

}  // namespace tensorflow
