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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status CandidateSamplerShapeFn(InferenceContext* c) {
  int64 num_sampled;
  TF_RETURN_IF_ERROR(c->GetAttr("num_sampled", &num_sampled));
  int64 num_true;
  TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));

  ShapeHandle true_classes_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &true_classes_shape));
  DimensionHandle batch_size = c->Dim(true_classes_shape, 0);

  ShapeHandle num_sampled_v = c->Vector(num_sampled);
  c->set_output(0, num_sampled_v);
  c->set_output(1, c->Matrix(batch_size, num_true));
  c->set_output(2, num_sampled_v);
  return Status::OK();
}

}  // namespace

REGISTER_OP("UniformCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful()
    .Doc(R"doc(
Generates labels for candidate sampling with a uniform distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

true_classes: A batch_size * num_true matrix, in which each row contains the
  IDs of the num_true target_classes in the corresponding original label.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_true: Number of true labels per context.
num_sampled: Number of candidates to randomly sample.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
range_max: The sampler will sample integers from the interval [0, range_max).
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("LogUniformCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful()
    .Doc(R"doc(
Generates labels for candidate sampling with a log-uniform distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.


true_classes: A batch_size * num_true matrix, in which each row contains the
  IDs of the num_true target_classes in the corresponding original label.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_true: Number of true labels per context.
num_sampled: Number of candidates to randomly sample.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
range_max: The sampler will sample integers from the interval [0, range_max).
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("LearnedUnigramCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful()
    .Doc(R"doc(
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

true_classes: A batch_size * num_true matrix, in which each row contains the
  IDs of the num_true target_classes in the corresponding original label.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_true: Number of true labels per context.
num_sampled: Number of candidates to randomly sample.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
range_max: The sampler will sample integers from the interval [0, range_max).
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("ThreadUnsafeUnigramCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful()
    .Doc(R"doc(
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

true_classes: A batch_size * num_true matrix, in which each row contains the
  IDs of the num_true target_classes in the corresponding original label.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_true: Number of true labels per context.
num_sampled: Number of candidates to randomly sample.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
range_max: The sampler will sample integers from the interval [0, range_max).
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("FixedUnigramCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("vocab_file: string = ''")
    .Attr("distortion: float = 1.0")
    .Attr("num_reserved_ids: int = 0")
    .Attr("num_shards: int >= 1 = 1")
    .Attr("shard: int >= 0 = 0")
    .Attr("unigrams: list(float) = []")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful()
    .Doc(R"doc(
Generates labels for candidate sampling with a learned unigram distribution.

A unigram sampler could use a fixed unigram distribution read from a
file or passed in as an in-memory array instead of building up the distribution
from data on the fly. There is also an option to skew the distribution by
applying a distortion power to the weights.

The vocabulary file should be in CSV-like format, with the last field
being the weight associated with the word.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

true_classes: A batch_size * num_true matrix, in which each row contains the
  IDs of the num_true target_classes in the corresponding original label.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_true: Number of true labels per context.
num_sampled: Number of candidates to randomly sample.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
range_max: The sampler will sample integers from the interval [0, range_max).
vocab_file: Each valid line in this file (which should have a CSV-like format)
  corresponds to a valid word ID. IDs are in sequential order, starting from
  num_reserved_ids. The last entry in each line is expected to be a value
  corresponding to the count or relative probability. Exactly one of vocab_file
  and unigrams needs to be passed to this op.
distortion: The distortion is used to skew the unigram probability distribution.
  Each weight is first raised to the distortion's power before adding to the
  internal unigram distribution. As a result, distortion = 1.0 gives regular
  unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
  a uniform distribution.
num_reserved_ids: Optionally some reserved IDs can be added in the range [0,
  ..., num_reserved_ids) by the users. One use case is that a special unknown
  word token is used as ID 0. These IDs will have a sampling probability of 0.
num_shards: A sampler can be used to sample from a subset of the original range
  in order to speed up the whole computation through parallelism. This parameter
  (together with 'shard') indicates the number of partitions that are being
  used in the overall computation.
shard: A sampler can be used to sample from a subset of the original range
  in order to speed up the whole computation through parallelism. This parameter
  (together with 'num_shards') indicates the particular partition number of a
  sampler op, when partitioning is being used.
unigrams: A list of unigram counts or probabilities, one per ID in sequential
  order. Exactly one of vocab_file and unigrams should be passed to this op.
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("AllCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful()
    .Doc(R"doc(
Generates labels for candidate sampling with a learned unigram distribution.

See explanations of candidate sampling and the data formats at
go/candidate-sampling.

For each batch, this op picks a single set of sampled candidate labels.

The advantages of sampling candidates per-batch are simplicity and the
possibility of efficient dense matrix multiplication. The disadvantage is that
the sampled candidates must be chosen independently of the context and of the
true labels.

true_classes: A batch_size * num_true matrix, in which each row contains the
  IDs of the num_true target_classes in the corresponding original label.
sampled_candidates: A vector of length num_sampled, in which each element is
  the ID of a sampled candidate.
true_expected_count: A batch_size * num_true matrix, representing
  the number of times each candidate is expected to occur in a batch
  of sampled candidates. If unique=true, then this is a probability.
sampled_expected_count: A vector of length num_sampled, for each sampled
  candidate representing the number of times the candidate is expected
  to occur in a batch of sampled candidates.  If unique=true, then this is a
  probability.
num_true: Number of true labels per context.
num_sampled: Number of candidates to produce.
unique: If unique is true, we sample with rejection, so that all sampled
  candidates in a batch are unique. This requires some approximation to
  estimate the post-rejection sampling probabilities.
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

REGISTER_OP("ComputeAccidentalHits")
    .Input("true_classes: int64")
    .Input("sampled_candidates: int64")
    .Output("indices: int32")
    .Output("ids: int64")
    .Output("weights: float")
    .Attr("num_true: int")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      int64 num_true;
      TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));

      // Validate true_classes.
      ShapeHandle true_classes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &true_classes));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(true_classes, 1), num_true, &unused));

      // All three outputs are the same shape.
      ShapeHandle v = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, v);
      c->set_output(1, v);
      c->set_output(2, v);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the ids of the positions in sampled_candidates that match true_labels.

When doing log-odds NCE, the result of this op should be passed through a
SparseToDense op, then added to the logits of the sampled candidates. This has
the effect of 'removing' the sampled labels that match the true labels by
making the classifier sure that they are sampled labels.

true_classes: The true_classes output of UnpackSparseLabels.
sampled_candidates: The sampled_candidates output of CandidateSampler.
indices: A vector of indices corresponding to rows of true_candidates.
ids: A vector of IDs of positions in sampled_candidates that match a true_label
  for the row with the corresponding index in indices.
weights: A vector of the same length as indices and ids, in which each element
  is -FLOAT_MAX.
num_true: Number of true labels per context.
seed: If either seed or seed2 are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: An second seed to avoid seed collision.
)doc");

}  // namespace tensorflow
