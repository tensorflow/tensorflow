/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CandidateSamplerOpsTest, CandidateSampler_ShapeFn) {
  for (const char* op_name : {
           "AllCandidateSampler", "FixedUnigramCandidateSampler",
           "LearnedUnigramCandidateSampler", "LogUniformCandidateSampler",
           "ThreadUnsafeUnigramCandidateSampler", "UniformCandidateSampler",
       }) {
    ShapeInferenceTestOp op(op_name);
    TF_CHECK_OK(NodeDefBuilder("test", op.name)
                    .Input({"a", 0, DT_INT64})
                    .Attr("num_sampled", 5)
                    .Attr("num_true", 10)
                    .Finalize(&op.node_def));

    // num_sampled = 5, num_true = 10.
    INFER_OK(op, "?", "[5];[?,10];[5]");

    // input.dim(0) becomes output[2].dim(0).
    INFER_OK(op, "[?,?]", "[5];[d0_0,10];[5]");
    INFER_OK(op, "[8,9]", "[5];[d0_0,10];[5]");

    // Rank check.
    INFER_ERROR("must be rank 2", op, "[1]");
  }
}

TEST(CandidateSamplerOpsTest, ComputeAccidentalHits_ShapeFn) {
  ShapeInferenceTestOp op("ComputeAccidentalHits");
  TF_CHECK_OK(NodeDefBuilder("test", op.name)
                  .Input({"a", 0, DT_INT64})
                  .Input({"b", 0, DT_INT64})
                  .Attr("num_true", 10)
                  .Finalize(&op.node_def));

  // output is always 3 [?] vectors.
  INFER_OK(op, "?;?", "[?];[?];[?]");
  INFER_OK(op, "[?,?];?", "[?];[?];[?]");
  INFER_OK(op, "[?,10];?", "[?];[?];[?]");
  INFER_OK(op, "[5,?];?", "[?];[?];[?]");

  // Error checks on first input -> must be rank 2, and input[0].dim(1) ==
  // num_true.
  INFER_ERROR("must be rank 2", op, "[1];?");
  INFER_ERROR("must be 10", op, "[?,11];?");
}

}  // end namespace tensorflow
