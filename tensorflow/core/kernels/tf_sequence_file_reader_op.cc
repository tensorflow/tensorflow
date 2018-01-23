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

#include <memory>

#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/hadoop_sequence_file/reader.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class TFSequenceFileReader : public ReaderBase {
 public:
  TFSequenceFileReader(const string& node_name, Env* env)
      : ReaderBase(strings::StrCat("TFSequenceFileReader '", node_name, "'")),
        env_(env), index_(0) {}

  Status OnWorkStartedLocked() override {
    index_ = 0;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file_));
    reader_.reset(new io::SequenceFileReader(file_.get(),
        io::SequenceFileReaderOptions::Defaults()));
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    reader_.reset(nullptr);
    file_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    *key = strings::StrCat(current_work(), ":", index_);
    const Status status = reader_->ReadRecord(value);
    if (errors::IsOutOfRange(status)) {
      *at_end = true;
      return Status::OK();
    }
    if (!status.ok()) {
      return status;
    }
    *produced = true;
    return Status::OK();
  }

  Status ResetLocked() override {
    index_ = 0;
    reader_.reset(nullptr);
    file_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

 private:
  Env* const env_;
  int64 index_;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::SequenceFileReader> reader_;
};

class TFSequenceFileReaderOp : public ReaderOpKernel {
 public:
  explicit TFSequenceFileReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory([this, env]() {
      return new TFSequenceFileReader(name(), env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TFSequenceFileReader").Device(DEVICE_CPU),
                        TFSequenceFileReaderOp);

}  // namespace tensorflow
