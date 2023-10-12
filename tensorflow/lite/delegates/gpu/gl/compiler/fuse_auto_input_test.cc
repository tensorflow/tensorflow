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

#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(FuseAutoInputTest, SkipsDiamond) {
  //     v0
  //   /    \
  // n1      n2
  // v1      v2
  //   \    /
  //     n3
  //     v3
  GraphFloat32 graph;
  auto* v0 = graph.NewValue();
  auto* v1 = graph.NewValue();
  auto* v2 = graph.NewValue();
  auto* v3 = graph.NewValue();
  auto* n1 = graph.NewNode();
  CompiledNodeAttributes a1;
  a1.code.output = IOStructure::AUTO;
  n1->operation.attributes = std::move(a1);
  ASSERT_OK(graph.AddConsumer(n1->id, v0->id));
  ASSERT_OK(graph.SetProducer(n1->id, v1->id));
  auto* n2 = graph.NewNode();
  CompiledNodeAttributes a2;
  a2.code.output = IOStructure::AUTO;
  n2->operation.attributes = std::move(a2);
  ASSERT_OK(graph.AddConsumer(n2->id, v0->id));
  ASSERT_OK(graph.SetProducer(n2->id, v2->id));
  auto* n3 = graph.NewNode();
  CompiledNodeAttributes a3;
  a3.code.input = IOStructure::AUTO;
  n3->operation.attributes = std::move(a3);
  ASSERT_OK(graph.AddConsumer(n3->id, v1->id));
  ASSERT_OK(graph.AddConsumer(n3->id, v2->id));
  ASSERT_OK(graph.SetProducer(n3->id, v3->id));

  FuseAutoInput fuse_auto_input;
  EXPECT_EQ(fuse_auto_input.ApplyToNode(n3, &graph).status,
            TransformStatus::SKIPPED);
}

TEST(FuseAutoInputTest, SkipsTriangle) {
  // v0
  // |  \
  // |   n1
  // |   v1
  // |  /
  // n2
  // v2
  GraphFloat32 graph;
  auto* v0 = graph.NewValue();
  auto* v1 = graph.NewValue();
  auto* v2 = graph.NewValue();
  auto* n1 = graph.NewNode();
  CompiledNodeAttributes a1;
  a1.code.output = IOStructure::AUTO;
  n1->operation.attributes = std::move(a1);
  ASSERT_OK(graph.AddConsumer(n1->id, v0->id));
  ASSERT_OK(graph.SetProducer(n1->id, v1->id));
  auto* n2 = graph.NewNode();
  CompiledNodeAttributes a2;
  a2.code.input = IOStructure::AUTO;
  n2->operation.attributes = std::move(a2);
  ASSERT_OK(graph.AddConsumer(n2->id, v0->id));
  ASSERT_OK(graph.AddConsumer(n2->id, v1->id));
  ASSERT_OK(graph.SetProducer(n2->id, v2->id));

  FuseAutoInput fuse_auto_input;
  EXPECT_EQ(fuse_auto_input.ApplyToNode(n2, &graph).status,
            TransformStatus::SKIPPED);
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
