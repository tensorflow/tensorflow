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

#include "tensorflow/contrib/data/kernels/indexed_dataset.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace data {
namespace {

class IdentityIndexedDatasetOp : public IndexedDatasetOpKernel {
 public:
  using IndexedDatasetOpKernel::IndexedDatasetOpKernel;

  void MakeIndexedDataset(OpKernelContext* ctx,
                          IndexedDataset** output) override {
    uint64 size = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<uint64>(ctx, "size", &size));
    OP_REQUIRES(ctx, size > 0, errors::InvalidArgument("`size` must be > 0"));
    *output = new Dataset(ctx, size);
  }

  class Dataset : public IndexedDataset {
   public:
    Dataset(OpKernelContext* ctx, uint64 size)
        : IndexedDataset(DatasetContext(ctx)), size_(size) {}

    Status MaterializeDataset(
        std::shared_ptr<MaterializedIndexedDataset>* materialized) override {
      materialized->reset(new Materialized(this));
      return Status::OK();
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_UINT64});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::IdentityIndexedDataset")}));
    }

    string DebugString() const override {
      return "IdentityIndexedDataset::Dataset";
    }

    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** node) const override {
      return errors::Unimplemented(
          "identity_indexed_dataset.AsGraphDefInternal");
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (cur_ < dataset()->size_) {
          Tensor result_tensor(ctx->allocator({}), DT_UINT64, {});
          result_tensor.scalar<uint64>()() = cur_++;
          out_tensors->emplace_back(std::move(result_tensor));
          *end_of_sequence = false;
          return Status::OK();
        }
        *end_of_sequence = true;
        return Status::OK();
      }

     private:
      mutex mu_;
      uint64 cur_ GUARDED_BY(mu_);
    };

    class Materialized : public MaterializedIndexedDataset {
     public:
      explicit Materialized(Dataset* dataset) : dataset_(dataset) {
        dataset->Ref();
      }

      ~Materialized() override {
        // TODO(saeta): Pull this into MaterializedIndexedDataset
        dataset_->Unref();
      }

      const DataTypeVector& output_dtypes() const override {
        return dataset_->output_dtypes();
      }

      const std::vector<PartialTensorShape>& output_shapes() const override {
        return dataset_->output_shapes();
      }

      Status Get(IteratorContext&& ctx, uint64 index,
                 std::vector<Tensor>* out_tensors) const override {
        LOG(INFO) << "Materialized(" << dataset_->size_ << ")::Get(" << index
                  << ")";
        if (index >= dataset_->size_) {
          // Note: use InvalidArgument instead of OutOfRange error because many
          // things consider OutOfRange to be a "clean termination" error.
          return errors::InvalidArgument(
              "Index ", index,
              " is out of range for this dataset. (Size is: ", dataset_->size_,
              ".)");
        }
        Tensor result_tensor(ctx.allocator({}), DT_UINT64, {});
        result_tensor.scalar<uint64>()() = index;
        out_tensors->emplace_back(std::move(result_tensor));
        return Status::OK();
      }

      Status Size(uint64* size) const override {
        *size = dataset_->size_;
        return Status::OK();
      }

     private:
      const Dataset* const dataset_;  // Not owned.
    };

    const uint64 size_;
    std::shared_ptr<Materialized> materialized_;
  };
};

REGISTER_KERNEL_BUILDER(Name("IdentityIndexedDataset").Device(DEVICE_CPU),
                        IdentityIndexedDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
