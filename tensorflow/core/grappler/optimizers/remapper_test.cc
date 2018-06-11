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

#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class RemapperTest : public GrapplerTest {};

TEST_F(RemapperTest, FusedBatchNorm) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(s.WithOpName("dflt"), {3.14f, 2.7f}, {2, 1, 1, 1});
  Output x = ops::PlaceholderWithDefault(s.WithOpName("x"), dflt, {2, 1, 1, 1});
  Output scale = ops::Const(s.WithOpName("scale"), {0.3f}, {1});
  Output offset = ops::Const(s.WithOpName("offset"), {0.123f}, {1});
  Output mean = ops::Const(s.WithOpName("mean"), {7.3f}, {1});
  Output variance = ops::Const(s.WithOpName("variance"), {0.57f}, {1});
  ops::FusedBatchNorm::Attrs attr;
  attr = attr.IsTraining(false);
  ops::FusedBatchNorm bn(s.WithOpName("batch_norm"), x, scale, offset, mean,
                         variance, attr);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(RemapperTest, FusedBatchNormNCHW) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output dflt =
      ops::Const(s.WithOpName("dflt"), {3.14f, 2.7f, 1.0f, 2.0f, 3.0f, 100.0f},
                 {1, 3, 1, 2});
  Output x = ops::PlaceholderWithDefault(s.WithOpName("x"), dflt, {1, 3, 1, 2});
  Output scale = ops::Const(s.WithOpName("scale"), {0.3f, 7.0f, 123.0f}, {3});
  Output offset =
      ops::Const(s.WithOpName("offset"), {0.123f, 2.1f, 0.55f}, {3});
  Output mean = ops::Const(s.WithOpName("mean"), {7.3f, 8.3f, 3.1f}, {3});
  Output variance =
      ops::Const(s.WithOpName("variance"), {0.57f, 1.0f, 2.0f}, {3});
  ops::FusedBatchNorm::Attrs attr;
  attr = attr.IsTraining(false);
  attr = attr.DataFormat("NCHW");
  ops::FusedBatchNorm bn(s.WithOpName("batch_norm").WithDevice("/device:GPU:0"),
                         x, scale, offset, mean, variance, attr);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"batch_norm"};

  Remapper optimizer(RewriterConfig::ON);
  GraphDef output;
  TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

  if (GetNumAvailableGPUs() > 0) {
    // NCHW batch norm is only supported on GPU.
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
    EXPECT_EQ(1, tensors_expected.size());
    auto tensors = EvaluateNodes(output, item.fetch);
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  }
}

}  // namespace grappler
}  // namespace tensorflow
