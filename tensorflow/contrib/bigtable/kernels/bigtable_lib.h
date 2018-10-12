/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_BIGTABLE_LIB_H_
#define TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_BIGTABLE_LIB_H_

// Note: we use bigtable/client/internal/table.h as this is the no-exception API

#include "google/cloud/bigtable/data_client.h"
#include "google/cloud/bigtable/internal/table.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

Status GrpcStatusToTfStatus(const ::grpc::Status& status);

string RegexFromStringSet(const std::vector<string>& strs);

class BigtableClientResource : public ResourceBase {
 public:
  BigtableClientResource(
      string project_id, string instance_id,
      std::shared_ptr<google::cloud::bigtable::DataClient> client)
      : project_id_(std::move(project_id)),
        instance_id_(std::move(instance_id)),
        client_(std::move(client)) {}

  std::shared_ptr<google::cloud::bigtable::DataClient> get_client() {
    return client_;
  }

  string DebugString() override {
    return strings::StrCat("BigtableClientResource(project_id: ", project_id_,
                           ", instance_id: ", instance_id_, ")");
  }

 private:
  const string project_id_;
  const string instance_id_;
  std::shared_ptr<google::cloud::bigtable::DataClient> client_;
};

class BigtableTableResource : public ResourceBase {
 public:
  BigtableTableResource(BigtableClientResource* client, string table_name)
      : client_(client),
        table_name_(std::move(table_name)),
        table_(client->get_client(), table_name_,
               google::cloud::bigtable::AlwaysRetryMutationPolicy()) {
    client_->Ref();
  }

  ~BigtableTableResource() override { client_->Unref(); }

  ::google::cloud::bigtable::noex::Table& table() { return table_; }

  string DebugString() override {
    return strings::StrCat(
        "BigtableTableResource(client: ", client_->DebugString(),
        ", table: ", table_name_, ")");
  }

 private:
  BigtableClientResource* client_;  // Ownes one ref.
  const string table_name_;
  ::google::cloud::bigtable::noex::Table table_;
};

namespace data {

// BigtableReaderDatasetIterator is an abstract class for iterators from
// datasets that are "readers" (source datasets, not transformation datasets)
// that read from Bigtable.
template <typename Dataset>
class BigtableReaderDatasetIterator : public DatasetIterator<Dataset> {
 public:
  explicit BigtableReaderDatasetIterator(
      const typename DatasetIterator<Dataset>::Params& params)
      : DatasetIterator<Dataset>(params), iterator_(nullptr, false) {}

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(EnsureIteratorInitialized());
    if (iterator_ == reader_->end()) {
      grpc::Status status = reader_->Finish();
      if (status.ok()) {
        *end_of_sequence = true;
        return Status::OK();
      }
      return GrpcStatusToTfStatus(status);
    }
    *end_of_sequence = false;
    google::cloud::bigtable::Row& row = *iterator_;
    Status s = ParseRow(ctx, row, out_tensors);
    // Ensure we always advance.
    ++iterator_;
    return s;
  }

 protected:
  virtual ::google::cloud::bigtable::RowRange MakeRowRange() = 0;
  virtual ::google::cloud::bigtable::Filter MakeFilter() = 0;
  virtual Status ParseRow(IteratorContext* ctx,
                          const ::google::cloud::bigtable::Row& row,
                          std::vector<Tensor>* out_tensors) = 0;

 private:
  Status EnsureIteratorInitialized() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (reader_) {
      return Status::OK();
    }

    auto rows = MakeRowRange();
    auto filter = MakeFilter();

    // Note: the this in `this->dataset()` below is necessary due to namespace
    // name conflicts.
    reader_.reset(new ::google::cloud::bigtable::RowReader(
        this->dataset()->table()->table().ReadRows(rows, filter)));
    iterator_ = reader_->begin();
    return Status::OK();
  }

  mutex mu_;
  std::unique_ptr<::google::cloud::bigtable::RowReader> reader_ GUARDED_BY(mu_);
  ::google::cloud::bigtable::RowReader::iterator iterator_ GUARDED_BY(mu_);
};

}  // namespace data

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_BIGTABLE_LIB_H_
