/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class SamplingDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SamplingDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  // Create a new SamplingDatasetOp::Dataset, and return it as the output.
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    float rate;
    int64 seed;
    int64 seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<float>(ctx, "rate", &rate));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed", &seed));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "seed2", &seed2));

    if (seed == 0 && seed2 == 0) {
      seed = random::New64();
      seed2 = random::New64();
    }
    *output = new Dataset(ctx, rate, seed, seed2, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, float rate, int64 seed, int64 seed2,
            const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)),
          rate_(rate),
          seed_(seed),
          seed2_(seed2),
          input_(input) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::Sampling")}, seed_, seed2_));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SamplingDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* rate = nullptr;
      Node* seed = nullptr;
      Node* seed2 = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(rate_, &rate));
      TF_RETURN_IF_ERROR(b->AddScalar(seed_, &seed));
      TF_RETURN_IF_ERROR(b->AddScalar(seed2_, &seed2));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, rate, seed, seed2}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params, int64 seed, int64 seed2)
          : DatasetIterator<Dataset>(params),
            seed_(seed),
            seed2_(seed2),
            parent_generator_(seed, seed2),
            generator_(&parent_generator_) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        bool rand_val_hit;
        do {
          {
            tf_shared_lock l(mu_);
            if (!input_impl_) {
              *end_of_sequence = true;
              return Status::OK();
            }
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
          }
          if (*end_of_sequence) {
            mutex_lock l(mu_);
            input_impl_.reset();
            return Status::OK();
          }

          // generate a number from random uniform [0, 1)
          float rand_val = Random();
          rand_val_hit = rand_val < dataset()->rate_;
          if (!rand_val_hit) {
            // Clear the output tensor list since it doesn't match.
            out_tensors->clear();
          }
        } while (!rand_val_hit);
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      void ResetRngs() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // Reset the generators based on the current iterator seeds.
        parent_generator_ = random::PhiloxRandom(seed_, seed2_);
        generator_ = random::SimplePhilox(&parent_generator_);

        parent_generator_.Skip(num_random_samples_);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        // Save state needed to restore the random number generators.
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            this->full_name("num_random_samples"), num_random_samples_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name("seed"), seed_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name("seed2"), seed2_));

        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        // Restore the random number generators.
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            this->full_name("num_random_samples"), &num_random_samples_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name("seed"), &seed_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name("seed2"), &seed2_));
        ResetRngs();

        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        return Status::OK();
      }

      mutex mu_;
      int64 seed_ GUARDED_BY(mu_);
      int64 seed2_ GUARDED_BY(mu_);

     private:
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);

      float Random() {
        mutex_lock l(mu_);
        num_random_samples_++;
        auto out = generator_.RandFloat();
        return out;
      }

      // random util
      random::PhiloxRandom parent_generator_ GUARDED_BY(mu_);
      random::SimplePhilox generator_ GUARDED_BY(mu_);
      int64 num_random_samples_ GUARDED_BY(mu_) = 0;
    };

    const float rate_;
    const int64 seed_, seed2_;
    const DatasetBase* const input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SamplingDataset").Device(DEVICE_CPU),
                        SamplingDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
