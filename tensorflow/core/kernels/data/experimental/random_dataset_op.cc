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
#include "tensorflow/core/kernels/data/experimental/random_dataset_op.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {
namespace experimental {

// Constants declared in random_dataset_op.h and used both here and in test
// cases.
/* static */ constexpr const char* const RandomDatasetOp::kDatasetType;
/* static */ constexpr const char* const RandomDatasetOp::kSeed;
/* static */ constexpr const char* const RandomDatasetOp::kSeed2;
/* static */ constexpr const char* const RandomDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RandomDatasetOp::kOutputShapes;

class RandomDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64 seed, int64 seed2)
      : DatasetBase(DatasetContext(ctx)), seeds_(seed, seed2) {}

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
    return strings::StrCat("RandomDatasetOp(", seeds_.first, ", ",
                           seeds_.second, ")::Dataset");
  }

  int64 Cardinality() const override { return kInfiniteCardinality; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return Status::OK();
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* seed = nullptr;
    Node* seed2 = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.first, &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.second, &seed2));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {seed, seed2}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          seeds_(MaybeOverrideSeeds(dataset()->seeds_)),
          parent_generator_(seeds_.first, seeds_.second),
          generator_(&parent_generator_) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      out_tensors->reserve(1);
      mutex_lock l(mu_);
      out_tensors->emplace_back(ctx->allocator({}), DT_INT64, TensorShape({}));
      out_tensors->back().scalar<int64>()() = Random();
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
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
      parent_generator_ = random::PhiloxRandom(seeds_.first, seeds_.second);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
      return Status::OK();
    }

   private:
    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      num_random_samples_++;
      auto out = generator_();
      return out;
    }
    const std::pair<int64, int64> seeds_;
    mutex mu_;
    random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        TF_GUARDED_BY(mu_);
    int64 num_random_samples_ TF_GUARDED_BY(mu_) = 0;
  };

  const std::pair<int64, int64> seeds_;
};  // RandomDatasetOp::Dataset

RandomDatasetOp::RandomDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {}

void RandomDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  int64 seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));

  int64 seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

  *output = new Dataset(ctx, seed, seed2);
}
namespace {

REGISTER_KERNEL_BUILDER(Name("RandomDataset").Device(DEVICE_CPU),
                        RandomDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalRandomDataset").Device(DEVICE_CPU),
                        RandomDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
