/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/sql/driver_manager.h"
#include "tensorflow/core/kernels/data/sql/query_connection.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following ops.

class SqlDatasetOp : public DatasetOpKernel {
 public:
  explicit SqlDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    for (const DataType& dt : output_types_) {
      OP_REQUIRES(ctx,
                  dt == DT_STRING || dt == DT_INT8 || dt == DT_INT16 ||
                      dt == DT_INT32 || dt == DT_INT64 || dt == DT_UINT8 ||
                      dt == DT_UINT16 || dt == DT_BOOL || dt == DT_DOUBLE,
                  errors::InvalidArgument(
                      "Each element of `output_types_` must be one of: "
                      "DT_STRING, DT_INT8, DT_INT16, DT_INT32, DT_INT64, "
                      "DT_UINT8, DT_UINT16, DT_BOOL, DT_DOUBLE "));
    }
    for (const PartialTensorShape& pts : output_shapes_) {
      OP_REQUIRES(ctx, pts.dims() == 0,
                  errors::InvalidArgument(
                      "Each element of `output_shapes_` must be a scalar."));
    }
  }
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    string driver_name;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<string>(ctx, "driver_name", &driver_name));

    string data_source_name;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "data_source_name",
                                                    &data_source_name));

    string query;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "query", &query));

    // TODO(b/64276826) Change this check when we add support for other
    // databases.
    OP_REQUIRES(ctx, driver_name == "sqlite",
                errors::InvalidArgument(tensorflow::strings::Printf(
                    "The database type, %s, is not supported by SqlDataset. "
                    "The set of supported databases is: {'sqlite'}.",
                    driver_name.c_str())));

    *output = new Dataset(ctx, driver_name, data_source_name, query,
                          output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const string& driver_name,
            const string& data_source_name, const string& query,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          driver_name_(driver_name),
          data_source_name_(data_source_name),
          query_(query),
          output_types_(output_types),
          output_shapes_(output_shapes) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Sql")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "SqlDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* driver_name_node;
      TF_RETURN_IF_ERROR(b->AddScalar(driver_name_, &driver_name_node));
      Node* data_source_name_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(data_source_name_, &data_source_name_node));
      Node* query_node;
      TF_RETURN_IF_ERROR(b->AddScalar(query_, &query_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {driver_name_node, data_source_name_node, query_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}
      ~Iterator() override {
        if (query_connection_initialized_) {
          Status s = query_connection_->Close();
          if (!s.ok()) {
            LOG(WARNING) << "Failed to close query connection: " << s;
          }
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!query_connection_initialized_) {
          TF_RETURN_IF_ERROR(InitializeQueryConnection());
        }
        next_calls_++;
        return query_connection_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (query_connection_initialized_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("next_calls"), next_calls_));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("next_calls"))) {
          TF_RETURN_IF_ERROR(InitializeQueryConnection());
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("next_calls"), &next_calls_));
          int64 rem_next_calls = next_calls_;
          std::vector<Tensor> out_tensors;
          bool end_of_sequence = false;
          while (rem_next_calls--) {
            TF_RETURN_IF_ERROR(query_connection_->GetNext(ctx, &out_tensors,
                                                          &end_of_sequence));
            out_tensors.clear();
          }
        } else {
          query_connection_initialized_ = false;
        }
        return Status::OK();
      }

     private:
      Status InitializeQueryConnection() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        query_connection_initialized_ = true;
        query_connection_ =
            sql::DriverManager::CreateQueryConnection(dataset()->driver_name_);
        Status s = query_connection_->Open(dataset()->data_source_name_,
                                           dataset()->query_,
                                           dataset()->output_types_);
        next_calls_ = 0;
        if (!s.ok()) {
          LOG(WARNING) << "Failed to connect to database: " << s;
          return s;
        }
        return Status::OK();
      }

      mutex mu_;
      // TODO(shivaniagrawal): explore ways to seek into a SQLite databases.
      int64 next_calls_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<sql::QueryConnection> query_connection_ GUARDED_BY(mu_);
      bool query_connection_initialized_ GUARDED_BY(mu_) = false;
    };
    const string driver_name_;
    const string data_source_name_;
    const string query_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("SqlDataset").Device(DEVICE_CPU), SqlDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
