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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* CropAndResize(int batches, int width, int height, int depth,
                            int crop_height, int crop_width) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in(DT_FLOAT, TensorShape({batches, height, width, depth}));
  in.flat<float>().setRandom();
  Tensor boxes(DT_FLOAT, TensorShape({batches, 4}));
  auto boxes_tensor = boxes.matrix<float>();
  Tensor box_ind(DT_INT32, TensorShape({batches}));
  auto box_ind_flat = box_ind.flat<int32>();
  for (int i = 0; i < batches; ++i) {
    boxes_tensor(i, 0) = 0.2;
    boxes_tensor(i, 1) = 0.2;
    boxes_tensor(i, 2) = 0.8;
    boxes_tensor(i, 3) = 0.7;
    box_ind_flat(i) = i;
  }
  Tensor crop_size(DT_INT32, TensorShape({2}));
  auto crop_size_flat = crop_size.flat<int32>();
  crop_size_flat(0) = crop_height;
  crop_size_flat(1) = crop_width;
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "CropAndResize")
                  .Input(test::graph::Constant(g, in))
                  .Input(test::graph::Constant(g, boxes))
                  .Input(test::graph::Constant(g, box_ind))
                  .Input(test::graph::Constant(g, crop_size))
                  .Finalize(g, &ret));
  return g;
}

#define BM_CropAndResizeDev(DEVICE, B, W, H, D, CH, CW)                        \
  static void BM_CropAndResize_##DEVICE##_##B##_##W##_##H##_##D##_##CH##_##CW( \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, CropAndResize(B, W, H, D, CH, CW),                \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(state.iterations() * B * W * H * D);               \
  }                                                                            \
  BENCHMARK(BM_CropAndResize_##DEVICE##_##B##_##W##_##H##_##D##_##CH##_##CW);

// Benchmark results using CPU:Intel Haswell with HyperThreading (6 cores)
// Benchmark                                Time(ns) CPU(ns)  Iterations
// BM_CropAndResize_cpu_1_640_640_3_512_512 7078765 7173520 100 163.361M items/s
// BM_CropAndResize_cpu_1_640_640_1_512_512 3801232 3914692 185  99.784M items/s
// BM_CropAndResize_cpu_1_80_80_512_7_7      182470  241767 2941  1.372G items/s

BM_CropAndResizeDev(cpu, 1, 640, 640, 3, 512, 512);
BM_CropAndResizeDev(cpu, 1, 640, 640, 1, 512, 512);
BM_CropAndResizeDev(cpu, 1, 80, 80, 512, 7, 7);

}  // namespace tensorflow
