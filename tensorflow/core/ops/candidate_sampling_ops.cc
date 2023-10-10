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
  int64_t num_sampled;
  TF_RETURN_IF_ERROR(c->GetAttr("num_sampled", &num_sampled));
  int64_t num_true;
  TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));

  ShapeHandle true_classes_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &true_classes_shape));
  DimensionHandle batch_size = c->Dim(true_classes_shape, 0);

  ShapeHandle num_sampled_v = c->Vector(num_sampled);
  c->set_output(0, num_sampled_v);
  c->set_output(1, c->Matrix(batch_size, num_true));
  c->set_output(2, num_sampled_v);
  return OkStatus();
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
    .SetIsStateful();

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
    .SetIsStateful();

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
    .SetIsStateful();

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
    .SetIsStateful();

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
    .SetIsStateful();

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
    .SetIsStateful();

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
      int64_t num_true;
      TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));

      // Validate true_classes, must be a matrix.
      ShapeHandle true_classes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &true_classes));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(true_classes, 1), num_true, &unused));
      // Validate sampled_candidates, must be a vector.
      ShapeHandle sampled_candidates;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sampled_candidates));

      // All three outputs are the same shape.
      ShapeHandle v = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, v);
      c->set_output(1, v);
      c->set_output(2, v);
      return OkStatus();
    });

}  // namespace tensorflow
