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

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include "tensorflow/core/util/mkl_heuristics.h"

#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

TEST(MklHeuristicsTest, MklCalculateMFlops) {
  int batch = 8;
  int width = 32;
  int height = 32;
  int in_depth = 3;

  int filter_h = 3;
  int filter_w = 3;
  int out_depth = 64;

  // Test calculation for number of MFLOPs for convolution
  AttrValue attr_input_shape;
  TensorShapeProto* proto = attr_input_shape.mutable_list()->add_shape();
  proto->add_dim()->set_size(batch);
  proto->add_dim()->set_size(width);
  proto->add_dim()->set_size(height);
  proto->add_dim()->set_size(in_depth);
  proto = attr_input_shape.mutable_list()->add_shape();
  proto->add_dim()->set_size(filter_h);
  proto->add_dim()->set_size(filter_w);
  proto->add_dim()->set_size(in_depth);
  proto->add_dim()->set_size(out_depth);

  NodeDef ndef;

  // If node doesn't have any _input_shapes it should return -1
  double calculated_empty_mflops =
      CalculateNodeMFlops(AttrSlice(ndef), "Conv2D");
  EXPECT_EQ(calculated_empty_mflops, -1);

  (*ndef.mutable_attr())["_input_shapes"] = attr_input_shape;

  double conv_calculated_mflops =
      CalculateNodeMFlops(AttrSlice(ndef), "Conv2D");
  double expected_conv_mflops = batch * width * height * in_depth * filter_h *
                                filter_w * out_depth / static_cast<double>(1e6);
  EXPECT_EQ(conv_calculated_mflops, expected_conv_mflops);

  // We should get the same calculation for fused convolution too
  double fused_calculated_mflops =
      CalculateNodeMFlops(AttrSlice(ndef), "_FusedConv2D");
  EXPECT_EQ(conv_calculated_mflops, expected_conv_mflops);

  // Finally calculate for sigmoid number of MFLOPS
  double sigmoid_calculated_mflops =
      CalculateNodeMFlops(AttrSlice(ndef), "Sigmoid");
  double expected_sigmoid_mflops =
      batch * width * height * in_depth / static_cast<double>(1e6);
  EXPECT_EQ(sigmoid_calculated_mflops, expected_sigmoid_mflops);
}

#ifdef DNNL_AARCH64_USE_ACL
TEST(MklHeuristicsTest, MklThresholds) {
  int cpu_family = tsl::port::CPUFamily();
  int cpu_model_num = tsl::port::CPUModelNum();

  int neoverse_v1_family = 0x41;
  int neoverse_v1_model = 0xd40;

  string op_type = "Conv2D";

  if (neoverse_v1_family == cpu_family && neoverse_v1_model == cpu_model_num) {
    double thread_sync_cost = -1;
    double framework_cost = -1;
    for (const RewriteThreshold* i = rewrite_thresholds; i->op != ""; i++) {
      if (i->op == op_type) {
        thread_sync_cost = i->params.thread_sync_cost;
        framework_cost = i->params.framework_cost;
        break;
      }
    }

    EXPECT_NE(thread_sync_cost, -1);
    EXPECT_NE(thread_sync_cost, -1);

    int no_threads = 0;
    double calculated_threshold_zero_threads =
        FindRewriteThreshold(op_type, no_threads);
    EXPECT_EQ(calculated_threshold_zero_threads, 0);

    int threads = 8;
    double calculated_threshold = FindRewriteThreshold(op_type, threads);
    double expected_threshold = threads * thread_sync_cost + framework_cost;
    EXPECT_EQ(expected_threshold, calculated_threshold);
  }
}
#endif  // DNNL_AARCG64_USE_ACL

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
