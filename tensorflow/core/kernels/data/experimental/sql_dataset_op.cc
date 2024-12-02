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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/experimental/sql/driver_manager.h"
#include "tensorflow/core/kernels/data/experimental/sql/query_connection.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

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
    tstring driver_name;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<tstring>(ctx, "driver_name", &driver_name));

    tstring data_source_name;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "data_source_name",
                                                     &data_source_name));

    tstring query;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, "query", &query));

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
      return std::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Sql")});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "SqlDatasetOp::Dataset"; }

    absl::Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      return absl::OkStatus();
    }

    absl::Status CheckExternalState() const override {
      return absl::OkStatus();
    }

   protected:
    absl::Status AsGraphDefInternal(SerializationContext* ctx,
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
      return absl::OkStatus();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}
      ~Iterator() override {
        if (query_connection_initialized_) {
          absl::Status s = query_connection_->Close();
          if (!s.ok()) {
            LOG(WARNING) << "Failed to close query connection: " << s;
          }
        }
      }

      absl::Status GetNextInternal(IteratorContext* ctx,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!query_connection_initialized_) {
          TF_RETURN_IF_ERROR(InitializeQueryConnection());
        }
        absl::Status status = absl::OkStatus();
        if (!end_of_sequence_) {
          next_calls_++;
          status =
              query_connection_->GetNext(ctx, out_tensors, &end_of_sequence_);
        }
        *end_of_sequence = end_of_sequence_;
        return status;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      absl::Status SaveInternal(SerializationContext* ctx,
                                IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (query_connection_initialized_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("next_calls"), next_calls_));
        }
        return absl::OkStatus();
      }

      absl::Status RestoreInternal(IteratorContext* ctx,
                                   IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("next_calls"))) {
          TF_RETURN_IF_ERROR(InitializeQueryConnection());
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("next_calls"), &next_calls_));
          int64_t rem_next_calls = next_calls_;
          std::vector<Tensor> out_tensors;
          end_of_sequence_ = false;
          while (rem_next_calls--) {
            TF_RETURN_IF_ERROR(query_connection_->GetNext(ctx, &out_tensors,
                                                          &end_of_sequence_));
            out_tensors.clear();
          }
        } else {
          query_connection_initialized_ = false;
          end_of_sequence_ = false;
        }
        return absl::OkStatus();
      }

     private:
      absl::Status InitializeQueryConnection()
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        query_connection_initialized_ = true;
        end_of_sequence_ = false;
        query_connection_ =
            sql::DriverManager::CreateQueryConnection(dataset()->driver_name_);
        absl::Status s = query_connection_->Open(dataset()->data_source_name_,
                                                 dataset()->query_,
                                                 dataset()->output_types_);
        next_calls_ = 0;
        if (!s.ok()) {
          LOG(WARNING) << "Failed to connect to database: " << s;
          return s;
        }
        return absl::OkStatus();
      }

      mutex mu_;
      // TODO(b/129062371): explore ways to seek into a SQLite databases.
      int64_t next_calls_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<sql::QueryConnection> query_connection_
          TF_GUARDED_BY(mu_);
      bool query_connection_initialized_ TF_GUARDED_BY(mu_) = false;
      bool end_of_sequence_ TF_GUARDED_BY(mu_) = false;
    };
    const tstring driver_name_;
    const tstring data_source_name_;
    const tstring query_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("SqlDataset").Device(DEVICE_CPU), SqlDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalSqlDataset").Device(DEVICE_CPU),
                        SqlDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
