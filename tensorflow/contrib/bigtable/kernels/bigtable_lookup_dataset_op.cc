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

#include "tensorflow/contrib/bigtable/kernels/bigtable_lib.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {
namespace {

class BigtableLookupDatasetOp : public UnaryDatasetOpKernel {
 public:
  using UnaryDatasetOpKernel::UnaryDatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    BigtableTableResource* table;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &table));
    core::ScopedUnref scoped_unref(table);

    std::vector<string> column_families;
    std::vector<string> columns;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<string>(ctx, "column_families",
                                                    &column_families));
    OP_REQUIRES_OK(ctx, ParseVectorArgument<string>(ctx, "columns", &columns));
    OP_REQUIRES(
        ctx, column_families.size() == columns.size(),
        errors::InvalidArgument("len(columns) != len(column_families)"));

    const uint64 num_outputs = columns.size() + 1;
    std::vector<PartialTensorShape> output_shapes;
    output_shapes.reserve(num_outputs);
    DataTypeVector output_types;
    output_types.reserve(num_outputs);
    for (uint64 i = 0; i < num_outputs; ++i) {
      output_shapes.push_back({});
      output_types.push_back(DT_STRING);
    }

    *output =
        new Dataset(ctx, input, table, std::move(column_families),
                    std::move(columns), output_types, std::move(output_shapes));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input,
                     BigtableTableResource* table,
                     std::vector<string> column_families,
                     std::vector<string> columns,
                     const DataTypeVector& output_types,
                     std::vector<PartialTensorShape> output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          table_(table),
          column_families_(std::move(column_families)),
          columns_(std::move(columns)),
          output_types_(output_types),
          output_shapes_(std::move(output_shapes)),
          filter_(MakeFilter(column_families_, columns_)) {
      table_->Ref();
      input_->Ref();
    }

    ~Dataset() override {
      table_->Unref();
      input_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::BigtableLookup")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "BigtableLookupDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return errors::Unimplemented("%s does not support serialization",
                                   DebugString());
    }

   private:
    static ::google::cloud::bigtable::Filter MakeFilter(
        const std::vector<string>& column_families,
        const std::vector<string>& columns) {
      string column_family_regex = RegexFromStringSet(column_families);
      string column_regex = RegexFromStringSet(columns);

      return ::google::cloud::bigtable::Filter::Chain(
          ::google::cloud::bigtable::Filter::Latest(1),
          ::google::cloud::bigtable::Filter::FamilyRegex(column_family_regex),
          ::google::cloud::bigtable::Filter::ColumnRegex(column_regex));
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);  // Sequence requests.
        std::vector<Tensor> input_tensors;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &input_tensors, end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }
        if (input_tensors.size() != 1) {
          return errors::InvalidArgument(
              "Upstream iterator (", dataset()->input_->DebugString(),
              ") did not produce a single `tf.string` `tf.Tensor`. It "
              "produced ",
              input_tensors.size(), " tensors.");
        }
        if (input_tensors[0].NumElements() == 0) {
          return errors::InvalidArgument("Upstream iterator (",
                                         dataset()->input_->DebugString(),
                                         ") return an empty set of keys.");
        }
        if (input_tensors[0].NumElements() == 1) {
          // Single key lookup.
          ::grpc::Status status;
          auto pair = dataset()->table_->table().ReadRow(
              input_tensors[0].scalar<string>()(), dataset()->filter_, status);
          if (!status.ok()) {
            return GrpcStatusToTfStatus(status);
          }
          if (!pair.first) {
            return errors::DataLoss("Row key '",
                                    input_tensors[0].scalar<string>()(),
                                    "' not found.");
          }
          TF_RETURN_IF_ERROR(ParseRow(ctx, pair.second, out_tensors));
        } else {
          // Batched get.
          return errors::Unimplemented(
              "BigtableLookupDataset doesn't yet support batched retrieval.");
        }
        return Status::OK();
      }

     private:
      Status ParseRow(IteratorContext* ctx,
                      const ::google::cloud::bigtable::Row& row,
                      std::vector<Tensor>* out_tensors) {
        out_tensors->reserve(dataset()->columns_.size() + 1);
        Tensor row_key_tensor(ctx->allocator({}), DT_STRING, {});
        row_key_tensor.scalar<string>()() = string(row.row_key());
        out_tensors->emplace_back(std::move(row_key_tensor));

        if (row.cells().size() > 2 * dataset()->columns_.size()) {
          LOG(WARNING) << "An excessive number of columns ("
                       << row.cells().size()
                       << ") were retrieved when reading row: "
                       << row.row_key();
        }

        for (uint64 i = 0; i < dataset()->columns_.size(); ++i) {
          Tensor col_tensor(ctx->allocator({}), DT_STRING, {});
          bool found_column = false;
          for (auto cell_itr = row.cells().begin();
               !found_column && cell_itr != row.cells().end(); ++cell_itr) {
            if (cell_itr->family_name() == dataset()->column_families_[i] &&
                string(cell_itr->column_qualifier()) ==
                    dataset()->columns_[i]) {
              col_tensor.scalar<string>()() = string(cell_itr->value());
              found_column = true;
            }
          }
          if (!found_column) {
            return errors::DataLoss("Column ", dataset()->column_families_[i],
                                    ":", dataset()->columns_[i],
                                    " not found in row: ", row.row_key());
          }
          out_tensors->emplace_back(std::move(col_tensor));
        }
        return Status::OK();
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    BigtableTableResource* table_;
    const std::vector<string> column_families_;
    const std::vector<string> columns_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const ::google::cloud::bigtable::Filter filter_;
  };
};

REGISTER_KERNEL_BUILDER(Name("BigtableLookupDataset").Device(DEVICE_CPU),
                        BigtableLookupDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
