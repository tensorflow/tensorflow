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

class BigtableScanDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    string prefix;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "prefix", &prefix));
    string start_key;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "start_key", &start_key));
    string end_key;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "end_key", &end_key));

    OP_REQUIRES(ctx, !(prefix.empty() && start_key.empty()),
                errors::InvalidArgument(
                    "Either prefix or start_key must be specified"));
    OP_REQUIRES(ctx, prefix.empty() || start_key.empty(),
                errors::InvalidArgument(
                    "Only one of prefix and start_key can be provided"));
    if (!prefix.empty()) {
      OP_REQUIRES(ctx, end_key.empty(),
                  errors::InvalidArgument(
                      "If prefix is specified, end_key must be empty."));
    }

    std::vector<string> column_families;
    std::vector<string> columns;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<string>(ctx, "column_families",
                                                    &column_families));
    OP_REQUIRES_OK(ctx, ParseVectorArgument<string>(ctx, "columns", &columns));
    OP_REQUIRES(
        ctx, column_families.size() == columns.size(),
        errors::InvalidArgument("len(columns) != len(column_families)"));
    OP_REQUIRES(ctx, !column_families.empty(),
                errors::InvalidArgument("`column_families` is empty"));

    float probability = 0;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<float>(ctx, "probability", &probability));
    OP_REQUIRES(
        ctx, probability > 0 && probability <= 1,
        errors::InvalidArgument(
            "Probability outside the range of (0, 1]. Got: ", probability));

    BigtableTableResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    core::ScopedUnref scoped_unref(resource);

    const uint64 num_outputs = columns.size() + 1;
    std::vector<PartialTensorShape> output_shapes;
    output_shapes.reserve(num_outputs);
    DataTypeVector output_types;
    output_types.reserve(num_outputs);
    for (uint64 i = 0; i < num_outputs; ++i) {
      output_shapes.push_back({});
      output_types.push_back(DT_STRING);
    }

    *output = new Dataset(ctx, resource, std::move(prefix),
                          std::move(start_key), std::move(end_key),
                          std::move(column_families), std::move(columns),
                          probability, output_types, std::move(output_shapes));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, BigtableTableResource* table,
                     string prefix, string start_key, string end_key,
                     std::vector<string> column_families,
                     std::vector<string> columns, float probability,
                     const DataTypeVector& output_types,
                     std::vector<PartialTensorShape> output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          table_(table),
          prefix_(std::move(prefix)),
          start_key_(std::move(start_key)),
          end_key_(std::move(end_key)),
          column_families_(std::move(column_families)),
          columns_(std::move(columns)),
          column_family_regex_(RegexFromStringSet(column_families_)),
          column_regex_(RegexFromStringSet(columns_)),
          probability_(probability),
          output_types_(output_types),
          output_shapes_(std::move(output_shapes)) {
      table_->Ref();
    }

    ~Dataset() override { table_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::BigtableScan")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "BigtableScanDatasetOp::Dataset";
    }

    BigtableTableResource* table() const { return table_; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return errors::Unimplemented("%s does not support serialization",
                                   DebugString());
    }

   private:
    class Iterator : public BigtableReaderDatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : BigtableReaderDatasetIterator<Dataset>(params) {}

      ::google::cloud::bigtable::RowRange MakeRowRange() override {
        if (!dataset()->prefix_.empty()) {
          DCHECK(dataset()->start_key_.empty());
          return ::google::cloud::bigtable::RowRange::Prefix(
              dataset()->prefix_);
        } else {
          DCHECK(!dataset()->start_key_.empty())
              << "Both prefix and start_key were empty!";
          return ::google::cloud::bigtable::RowRange::Range(
              dataset()->start_key_, dataset()->end_key_);
        }
      }
      ::google::cloud::bigtable::Filter MakeFilter() override {
        // TODO(saeta): Investigate optimal ordering here.
        return ::google::cloud::bigtable::Filter::Chain(
            ::google::cloud::bigtable::Filter::Latest(1),
            ::google::cloud::bigtable::Filter::FamilyRegex(
                dataset()->column_family_regex_),
            ::google::cloud::bigtable::Filter::ColumnRegex(
                dataset()->column_regex_),
            dataset()->probability_ != 1.0
                ? ::google::cloud::bigtable::Filter::RowSample(
                      dataset()->probability_)
                : ::google::cloud::bigtable::Filter::PassAllFilter());
      }
      Status ParseRow(IteratorContext* ctx,
                      const ::google::cloud::bigtable::Row& row,
                      std::vector<Tensor>* out_tensors) override {
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
            return errors::InvalidArgument(
                "Column ", dataset()->column_families_[i], ":",
                dataset()->columns_[i], " not found in row: ", row.row_key());
          }
          out_tensors->emplace_back(std::move(col_tensor));
        }
        return Status::OK();
      }
    };

    BigtableTableResource* table_;
    const string prefix_;
    const string start_key_;
    const string end_key_;
    const std::vector<string> column_families_;
    const std::vector<string> columns_;
    const string column_family_regex_;
    const string column_regex_;
    const float probability_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("BigtableScanDataset").Device(DEVICE_CPU),
                        BigtableScanDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
