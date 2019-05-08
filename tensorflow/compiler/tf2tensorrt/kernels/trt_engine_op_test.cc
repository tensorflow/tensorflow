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
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {
using ::testing::ElementsAre;

template <typename T>
class TRTEngineOpTest : public OpsTestBase {};

using TypeList = ::testing::Types<float, Eigen::half>;
TYPED_TEST_SUITE(TRTEngineOpTest, TypeList);

TYPED_TEST(TRTEngineOpTest, Basic) {
  DataType dtype = DataTypeToEnum<TypeParam>::v();
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));

  // Create simple TF graph.
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("TensorRTInputPH_0"), dtype,
                               ops::Placeholder::Shape({1, 2}));
  auto add = ops::Add(s.WithOpName("add"), feed, feed);
  ops::Identity(s.WithOpName("TensorRTOutputPH_0"), add);

  // Serialize the graph. TRTEngineOp will convert it using dynamic mode.
  GraphDef graph_def;
  TF_ASSERT_OK(s.ToGraphDef(&graph_def));
  TensorShapeProto shape;
  TensorShape({1, 2}).AsProto(&shape);

  // Create the op.
  OpsTestBase::SetDevice(DEVICE_GPU, std::move(device));
  TF_ASSERT_OK(NodeDefBuilder("op", "TRTEngineOp")
                   .Input(FakeInput(1, dtype))
                   .Attr("input_shapes", {shape})
                   .Attr("output_shapes", {shape})
                   .Attr("static_engine", false)
                   .Attr("segment_funcdef_name", "")  // no native fallback
                   .Attr("serialized_segment", graph_def.SerializeAsString())
                   .Attr("calibration_data", "")
                   .Attr("max_cached_engines_count", 1)
                   .Attr("workspace_size_bytes", 1 << 20)
                   .Attr("precision_mode", "FP32")
                   .Attr("use_calibration", false)
                   .Attr("OutT", {dtype})
                   .Finalize(OpsTestBase::node_def()));
  TF_ASSERT_OK(OpsTestBase::InitOp());

  // Execute the op.
  OpsTestBase::AddInputFromArray<TypeParam>(TensorShape({1, 2}),
                                            {TypeParam(0.0f), TypeParam(1.0f)});
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Verify the result.
  // TODO(laigd): OpsTestBase::GetOutput() doesn't work.
  Tensor* output = OpsTestBase::context_->mutable_output(0);
  const auto& tensor_map = output->flat<TypeParam>();
  std::vector<TypeParam> output_data(tensor_map.size());
  ASSERT_EQ(0, cudaDeviceSynchronize());
  ASSERT_EQ(0, cudaMemcpy(output_data.data(), tensor_map.data(),
                          sizeof(TypeParam) * tensor_map.size(),
                          cudaMemcpyDeviceToHost));
  EXPECT_THAT(absl::Span<const TypeParam>(output_data),
              ElementsAre(TypeParam(0.0f), TypeParam(2.0f)));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
