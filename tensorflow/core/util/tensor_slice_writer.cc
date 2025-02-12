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

#include "tensorflow/core/util/tensor_slice_writer.h"

#include <memory>
#include <utility>

#include "tensorflow/core/framework/versions.pb.h"
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
    builder_ = std::make_unique<table::TableBuilder>(option, f);
  }
  void Add(absl::string_view key, absl::string_view val) override {
    builder_->Add(key, val);
  }
  absl::Status Finish(int64_t* file_size) override {
    *file_size = -1;
    absl::Status s = builder_->Finish();
    if (s.ok()) {
      s = file_->Close();
      if (s.ok()) {
        *file_size = builder_->FileSize();
      }
    }
    if (!s.ok()) {
      s = errors::Internal("Error writing (tmp) checkpoint file: ", name_, ": ",
                           s.message());
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

absl::Status CreateTableTensorSliceBuilder(
    const string& name, TensorSliceWriter::Builder** builder) {
  *builder = nullptr;
  std::unique_ptr<WritableFile> f;
  absl::Status s = Env::Default()->NewWritableFile(name, &f);
  if (s.ok()) {
    *builder = new TableBuilder(name, f.release());
    return absl::OkStatus();
  } else {
    return s;
  }
}

TensorSliceWriter::TensorSliceWriter(const string& filename,
                                     CreateBuilderFunction create_builder)
    : filename_(filename),
      create_builder_(std::move(create_builder)),
      slices_(0) {
  Env* env = Env::Default();
  absl::Status status = env->CanCreateTempFile(filename_, &use_temp_file_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to get CanCreateTempFile attribute: " << filename_;
    use_temp_file_ = true;
  }

  data_filename_ = filename_;
  if (use_temp_file_) {
    data_filename_ = strings::StrCat(filename_, ".tempstate", random::New64());
  }
  VersionDef* versions = sts_.mutable_meta()->mutable_versions();
  versions->set_producer(TF_CHECKPOINT_VERSION);
  versions->set_min_consumer(TF_CHECKPOINT_VERSION_MIN_CONSUMER);
}

absl::Status TensorSliceWriter::Finish() {
  Builder* b;
  absl::Status s = create_builder_(data_filename_, &b);
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

  int64_t file_size;
  s = builder->Finish(&file_size);
  // If use temp file, we need to rename the file to the proper name.
  if (use_temp_file_) {
    if (s.ok()) {
      s = Env::Default()->RenameFile(data_filename_, filename_);
      if (s.ok()) {
        VLOG(1) << "Written " << slices_ << " slices for "
                << sts_.meta().tensor_size() << " tensors (" << file_size
                << " bytes) to " << filename_;
      } else {
        LOG(ERROR) << "Failed to rename file " << data_filename_ << " to "
                   << filename_;
      }
    } else {
      Env::Default()->DeleteFile(data_filename_).IgnoreError();
    }
  }
  return s;
}

/* static */
size_t TensorSliceWriter::MaxBytesPerElement(DataType dt) {
  size_t max_bytes_per_element =
      TensorSliceWriter::MaxBytesPerElementOrZero(dt);
  if (max_bytes_per_element == 0) {
    LOG(FATAL) << "MaxBytesPerElement not implemented for dtype: " << dt;
  }
  return max_bytes_per_element;
}

/* static */
size_t TensorSliceWriter::MaxBytesPerElementOrZero(DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return 4;
    case DT_DOUBLE:
      return 8;
    case DT_INT32:
      return 10;
    case DT_UINT8:
      return 2;
    case DT_INT16:
      return 10;
    case DT_INT8:
      return 10;
    case DT_COMPLEX64:
      return 8;
    case DT_INT64:
      return 10;
    case DT_BOOL:
      return 1;
    case DT_QINT8:
      return 10;
    case DT_QUINT8:
      return 2;
    case DT_QINT32:
      return 10;
    case DT_QINT16:
      return 10;
    case DT_QUINT16:
      return 3;
    case DT_UINT16:
      return 3;
    case DT_COMPLEX128:
      return 16;
    case DT_HALF:
      return 3;
    case DT_INVALID:
    case DT_STRING:
    case DT_BFLOAT16:
    default:
      return 0;
  }
}

template <>
absl::Status TensorSliceWriter::SaveData(const tstring* data,
                                         int64_t num_elements, SavedSlice* ss) {
  size_t size_bound = ss->ByteSize() + kTensorProtoHeaderBytes +
                      (num_elements * MaxBytesPerElement(DT_INT32));
  for (int64_t i = 0; i < num_elements; ++i) {
    size_bound += data[i].size();
  }
  if (size_bound > kMaxMessageBytes) {
    return errors::InvalidArgument(
        "Tensor slice is too large to serialize (conservative estimate: ",
        size_bound, " bytes)");
  }
  Fill(data, num_elements, ss->mutable_data());
  DCHECK_GE(ss->ByteSize(), 0);
  DCHECK_LE(ss->ByteSize(), size_bound);
  return absl::OkStatus();
}

}  // namespace checkpoint

}  // namespace tensorflow
