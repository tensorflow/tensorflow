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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class RandomDatasetOp : public DatasetOpKernel {
 public:
  explicit RandomDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    int64 seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));

    int64 seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

    // By TensorFlow convention, passing 0 for both seeds indicates
    // that the shuffling should be seeded non-deterministically.
    if (seed == 0 && seed2 == 0) {
      seed = random::New64();
      seed2 = random::New64();
    }

    *output = new Dataset(ctx, seed, seed2);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 seed, int64 seed2)
        : DatasetBase(DatasetContext(ctx)), seed_(seed), seed2_(seed2) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Random")});
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_INT64});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override {
      return strings::StrCat("RandomDatasetOp(", seed_, ", ", seed2_,
                             ")::Dataset");
    }

    int64 Cardinality() const override { return kInfiniteCardinality; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* seed = nullptr;
      Node* seed2 = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
      TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {seed, seed2}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            parent_generator_(dataset()->seed_, dataset()->seed2_),
            generator_(&parent_generator_) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        out_tensors->emplace_back(ctx->allocator({}), DT_INT64,
                                  TensorShape({}));
        out_tensors->back().scalar<int64>()() = Random();
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("num_random_samples"),
                                               num_random_samples_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("num_random_samples"),
                                              &num_random_samples_));
        parent_generator_ =
            random::PhiloxRandom(dataset()->seed_, dataset()->seed2_);
        generator_ = random::SingleSampleAdapter<random::PhiloxRandom>(
            &parent_generator_);
        generator_.Skip(num_random_samples_);
        return Status::OK();
      }

     private:
      random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        num_random_samples_++;
        auto out = generator_();
        return out;
      }
      mutex mu_;
      random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
      random::SingleSampleAdapter<random::PhiloxRandom> generator_
          GUARDED_BY(mu_);
      int64 num_random_samples_ GUARDED_BY(mu_) = 0;
    };

    const int64 seed_;
    const int64 seed2_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalRandomDataset").Device(DEVICE_CPU),
                        RandomDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
