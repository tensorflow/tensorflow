/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* BM_CombinedNonMaxSuppression(int batches, int box_num,
                                           int class_num, int q) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor boxes(DT_FLOAT, TensorShape({batches, box_num, q, 4}));
  boxes.flat<float>().setRandom();
  Tensor scores(DT_FLOAT, TensorShape({batches, box_num, class_num}));
  scores.flat<float>().setRandom();

  Tensor max_output_size_per_class(100);
  Tensor max_total_size(9000);
  Tensor iou_threshold(float(0.3));
  Tensor score_threshold(float(0.25));

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "CombinedNonMaxSuppression")
                  .Input(test::graph::Constant(g, boxes))
                  .Input(test::graph::Constant(g, scores))
                  .Input(test::graph::Constant(g, max_output_size_per_class))
                  .Input(test::graph::Constant(g, max_total_size))
                  .Input(test::graph::Constant(g, iou_threshold))
                  .Input(test::graph::Constant(g, score_threshold))
                  .Attr("pad_per_class", false)
                  .Attr("clip_boxes", true)
                  .Finalize(g, &ret));
  return g;
}

#define BM_CombinedNonMaxSuppressionDev(DEVICE, B, BN, CN, Q)                \
  static void BM_CombinedNMS_##DEVICE##_##B##_##BN##_##CN##_##Q(int iters) { \
    testing::ItemsProcessed(iters* B);                                       \
    test::Benchmark(#DEVICE, BM_CombinedNonMaxSuppression(B, BN, CN, Q))     \
        .Run(iters);                                                         \
  }                                                                          \
  BENCHMARK(BM_CombinedNMS_##DEVICE##_##B##_##BN##_##CN##_##Q);

#define BM_Batch(BN, CN, Q)                            \
  BM_CombinedNonMaxSuppressionDev(cpu, 1, BN, CN, Q);  \
  BM_CombinedNonMaxSuppressionDev(cpu, 28, BN, CN, Q); \
  BM_CombinedNonMaxSuppressionDev(cpu, 32, BN, CN, Q); \
  BM_CombinedNonMaxSuppressionDev(cpu, 64, BN, CN, Q);

#define BN_Boxes_Number(CN, Q) \
  BM_Batch(500, CN, Q);        \
  BM_Batch(1000, CN, Q);       \
  BM_Batch(1917, CN, Q);       \
  BM_Batch(2500, CN, Q);

BN_Boxes_Number(25, 1);
BN_Boxes_Number(25, 25);
BN_Boxes_Number(90, 1);
BN_Boxes_Number(90, 90);
BN_Boxes_Number(200, 1);
BN_Boxes_Number(200, 200);

}  // namespace tensorflow
