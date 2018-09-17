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

class BigtablePrefixKeyDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    string prefix;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<string>(ctx, "prefix", &prefix));

    BigtableTableResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    *output = new Dataset(ctx, resource, std::move(prefix));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, BigtableTableResource* table,
                     string prefix)
        : DatasetBase(DatasetContext(ctx)),
          table_(table),
          prefix_(std::move(prefix)) {
      table_->Ref();
    }

    ~Dataset() override { table_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::BigtablePrefixKey")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override {
      return "BigtablePrefixKeyDatasetOp::Dataset";
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
        return ::google::cloud::bigtable::RowRange::Prefix(dataset()->prefix_);
      }
      ::google::cloud::bigtable::Filter MakeFilter() override {
        return ::google::cloud::bigtable::Filter::Chain(
            ::google::cloud::bigtable::Filter::CellsRowLimit(1),
            ::google::cloud::bigtable::Filter::StripValueTransformer());
      }
      Status ParseRow(IteratorContext* ctx,
                      const ::google::cloud::bigtable::Row& row,
                      std::vector<Tensor>* out_tensors) override {
        Tensor output_tensor(ctx->allocator({}), DT_STRING, {});
        output_tensor.scalar<string>()() = string(row.row_key());
        out_tensors->emplace_back(std::move(output_tensor));
        return Status::OK();
      }
    };

    BigtableTableResource* const table_;
    const string prefix_;
  };
};

REGISTER_KERNEL_BUILDER(Name("BigtablePrefixKeyDataset").Device(DEVICE_CPU),
                        BigtablePrefixKeyDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
