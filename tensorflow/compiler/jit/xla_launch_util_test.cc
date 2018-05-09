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

// Contains microbenchmarks for performance critical functions in
// xla_launch_util.cc.

#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// Test ExtractSubBuffer with different depths (depth of ShapeTree) and fan-outs
// (cardinality of each non-leaf node's children).
void BM_ExtractSubBuffer(int iters, int depth, int fan_out) {
  tensorflow::testing::StopTiming();
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {32, 64, 128});
  for (int i = 0; i < depth; ++i) {
    std::vector<xla::Shape> shapes(fan_out, shape);
    shape = xla::ShapeUtil::MakeTupleShape(shapes);
  }
  xla::ShapedBuffer shaped_buffer(shape, shape, /*platform=*/nullptr,
                                  /*device_ordinal=*/0);
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    // Extract a buffer from approximately the middle of the first level of the
    // tree.
    tensorflow::internal::ExtractSubShapedBuffer(&shaped_buffer,
                                                 /*index=*/fan_out / 2,
                                                 /*allocator=*/nullptr)
        .release();
  }
}

BENCHMARK(BM_ExtractSubBuffer)
    ->ArgPair(1, 4)
    ->ArgPair(1, 8)
    ->ArgPair(1, 32)
    ->ArgPair(1, 64)
    ->ArgPair(1, 128)
    ->ArgPair(1, 256)
    ->ArgPair(1, 512)
    ->ArgPair(2, 4)
    ->ArgPair(2, 8)
    ->ArgPair(2, 32)
    ->ArgPair(2, 64)
    ->ArgPair(2, 128);

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  tensorflow::testing::RunBenchmarks();
  return RUN_ALL_TESTS();
}
