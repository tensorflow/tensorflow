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

#include "tensorflow/core/util/tensor_slice_writer.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

namespace {

class TableBuilder : public TensorSliceWriter::Builder {
 public:
  TableBuilder(const string& name, WritableFile* f) : name_(name), file_(f) {
    table::Options option;
    option.compression = table::kNoCompression;
    builder_.reset(new table::TableBuilder(option, f));
  }
  void Add(StringPiece key, StringPiece val) override {
    builder_->Add(key, val);
  }
  Status Finish(int64* file_size) override {
    *file_size = -1;
    Status s = builder_->Finish();
    if (s.ok()) {
      s = file_->Close();
      if (s.ok()) {
        *file_size = builder_->FileSize();
      }
    }
    if (!s.ok()) {
      s = errors::Internal("Error writing (tmp) checkpoint file: ", name_, ": ",
                           s.ToString());
    }
    builder_.reset();
    file_.reset();
    return s;
  }

 private:
  string name_;
  std::unique_ptr<WritableFile> file_;
  std::unique_ptr<table::TableBuilder> builder_;
};
}  // anonymous namespace

Status CreateTableTensorSliceBuilder(const string& name,
                                     TensorSliceWriter::Builder** builder) {
  *builder = nullptr;
  WritableFile* f;
  Status s = Env::Default()->NewWritableFile(name, &f);
  if (s.ok()) {
    *builder = new TableBuilder(name, f);
    return Status::OK();
  } else {
    return s;
  }
}

TensorSliceWriter::TensorSliceWriter(const string& filename,
                                     CreateBuilderFunction create_builder)
    : filename_(filename),
      create_builder_(create_builder),
      tmpname_(strings::StrCat(filename, ".tempstate", random::New64())),
      slices_(0) {
  VersionDef* versions = sts_.mutable_meta()->mutable_versions();
  versions->set_producer(TF_CHECKPOINT_VERSION);
  versions->set_min_consumer(TF_CHECKPOINT_VERSION_MIN_CONSUMER);
}

Status TensorSliceWriter::Finish() {
  Builder* b;
  Status s = create_builder_(tmpname_, &b);
  if (!s.ok()) {
    delete b;
    return s;
  }
  std::unique_ptr<Builder> builder(b);

  // We save the saved tensor slice metadata as the first element.
  string meta;
  sts_.AppendToString(&meta);
  builder->Add(kSavedTensorSlicesKey, meta);

  // Go through all the data and add them
  for (const auto& x : data_) {
    builder->Add(x.first, x.second);
  }

  int64 file_size;
  s = builder->Finish(&file_size);
  // We need to rename the file to the proper name
  if (s.ok()) {
    s = Env::Default()->RenameFile(tmpname_, filename_);
    if (s.ok()) {
      VLOG(1) << "Written " << slices_ << " slices for "
              << sts_.meta().tensor_size() << " tensors (" << file_size
              << " bytes) to " << filename_;
    } else {
      LOG(ERROR) << "Failed to rename file " << tmpname_ << " to " << filename_;
    }
  } else {
    Env::Default()->DeleteFile(tmpname_);
  }
  return s;
}

}  // namespace checkpoint

}  // namespace tensorflow
