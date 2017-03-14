/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <memory>
#include <set>

#include "tensorflow/contrib/cloud/kernels/bigquery_table_accessor.h"
#include "tensorflow/contrib/cloud/kernels/bigquery_table_partition.pb.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {
namespace {

constexpr int64 kDefaultRowBufferSize = 1000;  // Number of rows to buffer.

// This is a helper function for reading table attributes from context.
Status GetTableAttrs(OpKernelConstruction* context, string* project_id,
                     string* dataset_id, string* table_id,
                     int64* timestamp_millis, std::vector<string>* columns,
                     string* test_end_point) {
  TF_RETURN_IF_ERROR(context->GetAttr("project_id", project_id));
  TF_RETURN_IF_ERROR(context->GetAttr("dataset_id", dataset_id));
  TF_RETURN_IF_ERROR(context->GetAttr("table_id", table_id));
  TF_RETURN_IF_ERROR(context->GetAttr("timestamp_millis", timestamp_millis));
  TF_RETURN_IF_ERROR(context->GetAttr("columns", columns));
  TF_RETURN_IF_ERROR(context->GetAttr("test_end_point", test_end_point));
  return Status::OK();
}

}  // namespace

// Note that overriden methods with names ending in "Locked" are called by
// ReaderBase while a mutex is held.
// See comments for ReaderBase.
class BigQueryReader : public ReaderBase {
 public:
  explicit BigQueryReader(BigQueryTableAccessor* bigquery_table_accessor,
                          const string& node_name)
      : ReaderBase(strings::StrCat("BigQueryReader '", node_name, "'")),
        bigquery_table_accessor_(CHECK_NOTNULL(bigquery_table_accessor)) {}

  Status OnWorkStartedLocked() override {
    BigQueryTablePartition partition;
    if (!partition.ParseFromString(current_work())) {
      return errors::InvalidArgument(
          "Could not parse work as as valid partition.");
    }
    TF_RETURN_IF_ERROR(bigquery_table_accessor_->SetPartition(partition));
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    *at_end = false;
    *produced = false;
    if (bigquery_table_accessor_->Done()) {
      *at_end = true;
      return Status::OK();
    }

    Example example;
    int64 row_id;
    TF_RETURN_IF_ERROR(bigquery_table_accessor_->ReadRow(&row_id, &example));

    *key = std::to_string(row_id);
    *value = example.SerializeAsString();
    *produced = true;
    return Status::OK();
  }

 private:
  // Not owned.
  BigQueryTableAccessor* bigquery_table_accessor_;
};

class BigQueryReaderOp : public ReaderOpKernel {
 public:
  explicit BigQueryReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    string table_id;
    string project_id;
    string dataset_id;
    int64 timestamp_millis;
    std::vector<string> columns;
    string test_end_point;

    OP_REQUIRES_OK(context,
                   GetTableAttrs(context, &project_id, &dataset_id, &table_id,
                                 &timestamp_millis, &columns, &test_end_point));
    OP_REQUIRES_OK(context,
                   BigQueryTableAccessor::New(
                       project_id, dataset_id, table_id, timestamp_millis,
                       kDefaultRowBufferSize, test_end_point, columns,
                       BigQueryTablePartition(), &bigquery_table_accessor_));

    SetReaderFactory([this]() {
      return new BigQueryReader(bigquery_table_accessor_.get(), name());
    });
  }

 private:
  std::unique_ptr<BigQueryTableAccessor> bigquery_table_accessor_;
};

REGISTER_KERNEL_BUILDER(Name("BigQueryReader").Device(DEVICE_CPU),
                        BigQueryReaderOp);

class GenerateBigQueryReaderPartitionsOp : public OpKernel {
 public:
  explicit GenerateBigQueryReaderPartitionsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string project_id;
    string dataset_id;
    string table_id;
    int64 timestamp_millis;
    std::vector<string> columns;
    string test_end_point;

    OP_REQUIRES_OK(context,
                   GetTableAttrs(context, &project_id, &dataset_id, &table_id,
                                 &timestamp_millis, &columns, &test_end_point));
    OP_REQUIRES_OK(context,
                   BigQueryTableAccessor::New(
                       project_id, dataset_id, table_id, timestamp_millis,
                       kDefaultRowBufferSize, test_end_point, columns,
                       BigQueryTablePartition(), &bigquery_table_accessor_));
    OP_REQUIRES_OK(context, InitializeNumberOfPartitions(context));
    OP_REQUIRES_OK(context, InitializeTotalNumberOfRows());
  }

  void Compute(OpKernelContext* context) override {
    const int64 partition_size = tensorflow::MathUtil::CeilOfRatio<int64>(
        total_num_rows_, num_partitions_);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_partitions_}),
                                            &output_tensor));

    auto output = output_tensor->template flat<string>();
    for (int64 i = 0; i < num_partitions_; ++i) {
      BigQueryTablePartition partition;
      partition.set_start_index(i * partition_size);
      partition.set_end_index(
          std::min(total_num_rows_, (i + 1) * partition_size) - 1);
      output(i) = partition.SerializeAsString();
    }
  }

 private:
  Status InitializeTotalNumberOfRows() {
    total_num_rows_ = bigquery_table_accessor_->total_num_rows();
    if (total_num_rows_ <= 0) {
      return errors::FailedPrecondition("Invalid total number of rows.");
    }
    return Status::OK();
  }

  Status InitializeNumberOfPartitions(OpKernelConstruction* context) {
    TF_RETURN_IF_ERROR(context->GetAttr("num_partitions", &num_partitions_));
    if (num_partitions_ <= 0) {
      return errors::FailedPrecondition("Invalid number of partitions.");
    }
    return Status::OK();
  }

  int64 num_partitions_;
  int64 total_num_rows_;
  std::unique_ptr<BigQueryTableAccessor> bigquery_table_accessor_;
};

REGISTER_KERNEL_BUILDER(
    Name("GenerateBigQueryReaderPartitions").Device(DEVICE_CPU),
    GenerateBigQueryReaderPartitionsOp);

}  // namespace tensorflow
