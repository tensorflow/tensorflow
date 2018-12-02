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

#ifndef TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_TEST_KERNELS_BIGTABLE_TEST_CLIENT_H_
#define TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_TEST_KERNELS_BIGTABLE_TEST_CLIENT_H_

#include "google/cloud/bigtable/data_client.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class BigtableTestClient : public ::google::cloud::bigtable::DataClient {
 public:
  std::string const& project_id() const override { return project_id_; }
  std::string const& instance_id() const override { return instance_id_; }
  void reset() override {
    mutex_lock l(mu_);
    table_ = Table();
  }

  grpc::Status MutateRow(
      grpc::ClientContext* context,
      google::bigtable::v2::MutateRowRequest const& request,
      google::bigtable::v2::MutateRowResponse* response) override;

  grpc::Status CheckAndMutateRow(
      grpc::ClientContext* context,
      google::bigtable::v2::CheckAndMutateRowRequest const& request,
      google::bigtable::v2::CheckAndMutateRowResponse* response) override;

  grpc::Status ReadModifyWriteRow(
      grpc::ClientContext* context,
      google::bigtable::v2::ReadModifyWriteRowRequest const& request,
      google::bigtable::v2::ReadModifyWriteRowResponse* response) override;

  std::unique_ptr<
      grpc::ClientReaderInterface<google::bigtable::v2::ReadRowsResponse>>
  ReadRows(grpc::ClientContext* context,
           google::bigtable::v2::ReadRowsRequest const& request) override;
  std::unique_ptr<
      grpc::ClientReaderInterface<google::bigtable::v2::SampleRowKeysResponse>>
  SampleRowKeys(
      grpc::ClientContext* context,
      google::bigtable::v2::SampleRowKeysRequest const& request) override;

  std::unique_ptr<
      grpc::ClientReaderInterface<google::bigtable::v2::MutateRowsResponse>>
  MutateRows(grpc::ClientContext* context,
             google::bigtable::v2::MutateRowsRequest const& request) override;

  std::unique_ptr<grpc::ClientAsyncResponseReaderInterface<
      google::bigtable::v2::MutateRowResponse>>
  AsyncMutateRow(grpc::ClientContext* context,
                 google::bigtable::v2::MutateRowRequest const& request,
                 grpc::CompletionQueue* cq) override;

  std::unique_ptr<::grpc::ClientAsyncReaderInterface<
      ::google::bigtable::v2::SampleRowKeysResponse>>
  AsyncSampleRowKeys(
      ::grpc::ClientContext* context,
      const ::google::bigtable::v2::SampleRowKeysRequest& request,
      ::grpc::CompletionQueue* cq, void* tag) override;

  std::unique_ptr<::grpc::ClientAsyncReaderInterface<
      ::google::bigtable::v2::MutateRowsResponse>>
  AsyncMutateRows(::grpc::ClientContext* context,
                  const ::google::bigtable::v2::MutateRowsRequest& request,
                  ::grpc::CompletionQueue* cq, void* tag) override;

  std::shared_ptr<grpc::Channel> Channel() override;

 private:
  friend class SampleRowKeysResponse;
  friend class ReadRowsResponse;
  friend class MutateRowsResponse;

  struct Row {
    string row_key;
    std::map<string, string> columns;
  };
  struct Table {
    std::map<string, Row> rows;
  };

  mutex mu_;
  const std::string project_id_ = "testproject";
  const std::string instance_id_ = "testinstance";
  Table table_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BIGTABLE_KERNELS_TEST_KERNELS_BIGTABLE_TEST_CLIENT_H_
