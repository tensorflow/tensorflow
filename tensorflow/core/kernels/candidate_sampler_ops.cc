/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/candidate_sampling_ops.cc.

#define EIGEN_USE_THREADS

#include <cfloat>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/range_sampler.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

class BaseCandidateSamplerOp : public OpKernel {
 public:
  explicit BaseCandidateSamplerOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_sampled", &num_sampled_));
    OP_REQUIRES_OK(context, context->GetAttr("num_true", &num_true_));
    OP_REQUIRES_OK(context, context->GetAttr("unique", &unique_));
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& true_classes = context->input(0);
    OP_REQUIRES(context, true_classes.dims() == 2,
                errors::InvalidArgument("true_classes must be a matrix"));
    const int32 batch_size = true_classes.dim_size(0);
    OP_REQUIRES(
        context, true_classes.dim_size(1) == num_true_,
        errors::InvalidArgument("true_classes must have "
                                "num_true columns, expected: ",
                                true_classes.dim_size(1), " was: ", num_true_));
    CHECK(sampler_) << "CandidateSamplerOp did not set sampler_";

    if (unique_) {
      OP_REQUIRES(context, num_sampled_ <= sampler_->range(),
                  errors::InvalidArgument("Sampler's range is too small."));
    }

    // Output candidates and expected_count.
    Tensor* out_sampled_candidates = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_sampled_}),
                                            &out_sampled_candidates));

    Tensor* out_true_expected_count = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({batch_size, num_true_}),
                                &out_true_expected_count));
    Tensor* out_sampled_expected_count = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, TensorShape({num_sampled_}),
                                            &out_sampled_expected_count));

    gtl::ArraySlice<int64> true_candidate(true_classes.matrix<int64>().data(),
                                          batch_size * num_true_);
    gtl::MutableArraySlice<int64> sampled_candidate(
        out_sampled_candidates->vec<int64>().data(), num_sampled_);
    gtl::MutableArraySlice<float> true_expected_count(
        out_true_expected_count->matrix<float>().data(),
        batch_size * num_true_);
    gtl::MutableArraySlice<float> sampled_expected_count(
        out_sampled_expected_count->vec<float>().data(), num_sampled_);

    // Approximately conservatively estimate the number of samples required.
    // In cases where rejection sampling is used we may occasionally use more
    // samples than expected, which will result in reused random bits.
    const int64 samples32 = 2048 * num_sampled_;

    // Pick sampled candidates.
    auto local_gen = generator_.ReserveSamples32(samples32);
    random::SimplePhilox random(&local_gen);
    sampler_->SampleBatchGetExpectedCount(&random, unique_, &sampled_candidate,
                                          &sampled_expected_count,
                                          true_candidate, &true_expected_count);

    if (sampler_->NeedsUpdates()) {
      sampler_->Update(true_candidate);
    }
  }

 protected:
  void set_sampler(RangeSampler* sampler) { sampler_.reset(sampler); }

 private:
  int32 num_true_;
  int32 num_sampled_;
  bool unique_;
  std::unique_ptr<RangeSampler> sampler_;
  GuardedPhiloxRandom generator_;
};

template <class RangeSamplerType>
class SimpleCandidateSamplerOp : public BaseCandidateSamplerOp {
 public:
  explicit SimpleCandidateSamplerOp(OpKernelConstruction* context)
      : BaseCandidateSamplerOp(context) {
    int64 range_max;
    OP_REQUIRES_OK(context, context->GetAttr("range_max", &range_max));
    set_sampler(new RangeSamplerType(range_max));
  }
};

REGISTER_KERNEL_BUILDER(Name("UniformCandidateSampler").Device(DEVICE_CPU),
                        SimpleCandidateSamplerOp<UniformSampler>);

REGISTER_KERNEL_BUILDER(Name("LogUniformCandidateSampler").Device(DEVICE_CPU),
                        SimpleCandidateSamplerOp<LogUniformSampler>);

REGISTER_KERNEL_BUILDER(Name("LearnedUnigramCandidateSampler")
                            .Device(DEVICE_CPU),
                        SimpleCandidateSamplerOp<UnigramSampler>);

REGISTER_KERNEL_BUILDER(Name("ThreadUnsafeUnigramCandidateSampler")
                            .Device(DEVICE_CPU),
                        SimpleCandidateSamplerOp<ThreadUnsafeUnigramSampler>);

class AllCandidateSamplerOp : public BaseCandidateSamplerOp {
 public:
  explicit AllCandidateSamplerOp(OpKernelConstruction* context)
      : BaseCandidateSamplerOp(context) {
    int64 range_max;
    OP_REQUIRES_OK(context, context->GetAttr("num_sampled", &range_max));
    set_sampler(new AllSampler(range_max));
  }
};

REGISTER_KERNEL_BUILDER(Name("AllCandidateSampler").Device(DEVICE_CPU),
                        AllCandidateSamplerOp);

class FixedUnigramCandidateSamplerOp : public BaseCandidateSamplerOp {
 public:
  explicit FixedUnigramCandidateSamplerOp(OpKernelConstruction* context)
      : BaseCandidateSamplerOp(context) {
    int64 range_max;
    OP_REQUIRES_OK(context, context->GetAttr("range_max", &range_max));
    string vocab_file;
    OP_REQUIRES_OK(context, context->GetAttr("vocab_file", &vocab_file));
    std::vector<float> unigrams;
    OP_REQUIRES_OK(context, context->GetAttr("unigrams", &unigrams));
    OP_REQUIRES(
        context, !vocab_file.empty() || !unigrams.empty(),
        errors::InvalidArgument("Must provide either vocab_file or unigrams."));
    OP_REQUIRES(context, vocab_file.empty() || unigrams.empty(),
                errors::InvalidArgument(
                    "Must only provide one of vocab_file and unigrams."));
    float distortion;
    OP_REQUIRES_OK(context, context->GetAttr("distortion", &distortion));
    int64 num_reserved_ids;
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_reserved_ids", &num_reserved_ids));
    int64 num_shards;
    OP_REQUIRES_OK(context, context->GetAttr("num_shards", &num_shards));
    int64 shard;
    OP_REQUIRES_OK(context, context->GetAttr("shard", &shard));

    if (!vocab_file.empty()) {
      set_sampler(new FixedUnigramSampler(context->env(), range_max, vocab_file,
                                          distortion, num_reserved_ids,
                                          num_shards, shard));
    } else {
      set_sampler(new FixedUnigramSampler(range_max, unigrams, distortion,
                                          num_reserved_ids, num_shards, shard));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FixedUnigramCandidateSampler").Device(DEVICE_CPU),
                        FixedUnigramCandidateSamplerOp);

class ComputeAccidentalHitsOp : public OpKernel {
 public:
  explicit ComputeAccidentalHitsOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_true", &num_true_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& in_true_candidates = context->input(0);
    const TensorShape& in_true_candidates_shape = in_true_candidates.shape();
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(in_true_candidates_shape) &&
                             in_true_candidates_shape.dim_size(1) == num_true_,
                errors::InvalidArgument(
                    "true_candidates must be a batch_size * num_true matrix"));

    const int64 batch_size = in_true_candidates_shape.dim_size(0);

    const Tensor& in_sampled_candidates = context->input(1);
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(in_sampled_candidates.shape()),
                errors::InvalidArgument(
                    "sampled_candidates must be a vector, which is typically "
                    "an output from CandidateSampler"));

    std::unordered_map<int64, int> sampled_candidate_to_pos;
    for (int64 i = 0; i < in_sampled_candidates.dim_size(0); ++i) {
      sampled_candidate_to_pos[in_sampled_candidates.vec<int64>()(i)] = i;
    }

    // Produce output in the same format as UnpackSparseFeatures.
    std::vector<int> indices;
    std::vector<int64> ids;
    std::vector<float> weights;

    for (int64 i = 0; i < batch_size; ++i) {
      for (int64 j = 0; j < num_true_; ++j) {
        const int64 true_candidate = in_true_candidates.matrix<int64>()(i, j);
        const auto look = sampled_candidate_to_pos.find(true_candidate);
        if (look != sampled_candidate_to_pos.end()) {
          indices.push_back(i);
          ids.push_back(look->second);
          weights.push_back(-FLT_MAX);
        }
      }
    }

    Tensor* out_indices = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({static_cast<int>(indices.size())}), &out_indices));
    Tensor* out_ids = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     1, TensorShape({static_cast<int>(ids.size())}), &out_ids));
    Tensor* out_weights = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            2, TensorShape({static_cast<int>(weights.size())}), &out_weights));

    for (size_t i = 0; i < indices.size(); ++i) {
      out_indices->vec<int32>()(i) = indices[i];
      out_ids->vec<int64>()(i) = ids[i];
      out_weights->vec<float>()(i) = weights[i];
    }
  }

 private:
  int64 num_true_;
};

REGISTER_KERNEL_BUILDER(Name("ComputeAccidentalHits").Device(DEVICE_CPU),
                        ComputeAccidentalHitsOp);

}  // namespace tensorflow
