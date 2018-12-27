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
#include "tensorflow/contrib/bigtable/kernels/bigtable_range_helpers.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {
namespace {

class BigtableSampleKeyPairsDatasetOp : public DatasetOpKernel {
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

    BigtableTableResource* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    core::ScopedUnref scoped_unref(resource);

    OP_REQUIRES(ctx, prefix.empty() || start_key.empty(),
                errors::InvalidArgument(
                    "Only one of prefix and start_key can be provided"));
    if (!prefix.empty()) {
      OP_REQUIRES(ctx, end_key.empty(),
                  errors::InvalidArgument(
                      "If prefix is specified, end_key must be empty."));
    }

    *output = new Dataset(ctx, resource, std::move(prefix),
                          std::move(start_key), std::move(end_key));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, BigtableTableResource* table,
                     string prefix, string start_key, string end_key)
        : DatasetBase(DatasetContext(ctx)),
          table_(table),
          key_range_(MakeMultiModeKeyRange(
              std::move(prefix), std::move(start_key), std::move(end_key))) {
      table_->Ref();
    }

    ~Dataset() override { table_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::BigtableSampleKeyPairs")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes =
          new DataTypeVector({DT_STRING, DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}, {}});
      return *shapes;
    }

    string DebugString() const override {
      return "BigtableSampleKeyPairsDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      return errors::Unimplemented("%s does not support serialization",
                                   DebugString());
    }

   private:
    static MultiModeKeyRange MakeMultiModeKeyRange(string prefix,
                                                   string start_key,
                                                   string end_key) {
      if (!start_key.empty()) {
        return MultiModeKeyRange::FromRange(std::move(start_key),
                                            std::move(end_key));
      }
      return MultiModeKeyRange::FromPrefix(std::move(prefix));
    }

    BigtableTableResource& table() const { return *table_; }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      // Computes split points (`keys_`) to use when scanning the table.
      //
      // Initialize first retrieves the sample keys from the table (`row_keys`),
      // as these often form good split points within the table. We then iterate
      // over them, and copy them to `keys_` if they fall within the requested
      // range to scan (`dataset()->key_range_`). Because the requested range
      // might start between elements of the sampled keys list, care is taken to
      // ensure we don't accidentally miss any subsets of the requested range by
      // including `begin_key()` and `end_key()` as appropriate.
      Status Initialize(IteratorContext* ctx) override {
        grpc::Status status;
        std::vector<google::cloud::bigtable::RowKeySample> row_keys =
            dataset()->table().table().SampleRows(status);
        if (!status.ok()) {
          return GrpcStatusToTfStatus(status);
        }

        for (size_t i = 0; i < row_keys.size(); ++i) {
          string row_key(row_keys[i].row_key);
          if (dataset()->key_range_.contains_key(row_key)) {
            // First key: check to see if we need to add the begin_key.
            if (keys_.empty() && dataset()->key_range_.begin_key() != row_key) {
              keys_.push_back(dataset()->key_range_.begin_key());
            }
            keys_.push_back(std::move(row_key));
          } else if (!keys_.empty()) {
            // If !keys_.empty(), then we have found at least one element of
            // `row_keys` that is within our requested range
            // (`dataset()->key_range_`). Because `row_keys` is sorted, if we
            // have found an element that's not within our key range, then we
            // are after our requested range (ranges are contiguous) and can end
            // iteration early.
            break;
          }
        }

        // Handle the case where we skip over the selected range entirely.
        if (keys_.empty()) {
          keys_.push_back(dataset()->key_range_.begin_key());
        }

        // Last key: check to see if we need to add the end_key.
        if (keys_.back() != dataset()->key_range_.end_key()) {
          keys_.push_back(dataset()->key_range_.end_key());
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (index_ + 2 > keys_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        *end_of_sequence = false;
        out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                  TensorShape({}));
        out_tensors->back().scalar<string>()() = keys_[index_];

        out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                  TensorShape({}));
        out_tensors->back().scalar<string>()() = keys_[index_ + 1];
        ++index_;

        return Status::OK();
      }

     private:
      mutex mu_;
      size_t index_ GUARDED_BY(mu_) = 0;
      // Note: we store the keys_ on the iterator instead of the dataset
      // because we want to re-sample the row keys in case there have been
      // tablet rebalancing operations since the dataset was created.
      //
      // Note: keys_ is readonly after Initialize, and thus does not need a
      // guarding lock.
      std::vector<string> keys_;
    };

    BigtableTableResource* const table_;
    const MultiModeKeyRange key_range_;
  };
};

REGISTER_KERNEL_BUILDER(
    Name("BigtableSampleKeyPairsDataset").Device(DEVICE_CPU),
    BigtableSampleKeyPairsDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
