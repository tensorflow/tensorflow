/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <dirent.h>
#include <string.h>
#include <fstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

using ::testing::ElementsAre;

class TRTEngineOpTest : public OpsTestBase {};

TEST_F(TRTEngineOpTest, Basic) {
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));

  // Create simple TF graph.
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("TensorRTInputPH_0"), DT_FLOAT, ops::Placeholder::Shape({1, 2}));
  auto const_1 = ops::Const(s.WithOpName("const_1"), 1.0f, TensorShape({1, 2}));
  auto add = ops::Add(s.WithOpName("add"), feed, const_1);
  ops::Identity(s.WithOpName("TensorRTOutputPH_0"), add);

  // Serialize the graph. TRTEngineOp will convert it using dynamic mode.
  GraphDef graph_def;
  s.ToGraphDef(&graph_def);
  const string segment_string = graph_def.SerializeAsString();
  std::vector<TensorShapeProto> input_shape_protos(1);
  TensorShape({1, 2}).AsProto(&input_shape_protos[0]);
  std::vector<TensorShapeProto> output_shape_protos(1);
  TensorShape({1, 2}).AsProto(&output_shape_protos[0]);

  // Create the op.
  SetDevice(DEVICE_GPU, std::move(device));
  TF_ASSERT_OK(NodeDefBuilder("op", "TRTEngineOp")
                   .Input(FakeInput(1, DT_FLOAT))
                   .Attr("input_shapes", input_shape_protos)
          .Attr("output_shapes", output_shape_protos)
          .Attr("static_engine", false)
          .Attr("segment_funcdef_name", "") // no native fallback
          .Attr("serialized_segment", segment_string)
          .Attr("calibration_data", "")
          .Attr("max_cached_engines_count", 1)
          .Attr("workspace_size_bytes", 1 << 20)
          .Attr("precision_mode", "FP32")
          .Attr("use_calibration", false)
          .Attr("OutT", {DT_FLOAT})
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Execute the op.
  AddInputFromArray<float>(TensorShape({1, 2}), {0.0f, 1.0f});
  TF_ASSERT_OK(RunOpKernel());

  // Verify the result.
  // TODO(laigd): OpsTestBase::GetOutput() doesn't work.
  Tensor* output = context_->mutable_output(0);
  // Tensor* output = GetOutput(0);
  const auto& tensor_map = output->flat<float>();
  const auto span = absl::Span<const float>(tensor_map.data(), tensor_map.size());
  EXPECT_THAT(span, ElementsAre(1.0f, 2.0f));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
