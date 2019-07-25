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
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {
using ::testing::ElementsAre;

class TRTEngineOpTestBase : public OpsTestBase {
 public:
  void AddSimpleTrtOp(DataType dtype, int max_cached_engines_count = 1) {
    // Create the GPU device.
    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));

    // Create simple TF graph.
    Scope s = Scope::NewRootScope();
    auto feed = ops::Placeholder(s.WithOpName("TensorRTInputPH_0"), dtype,
                                 ops::Placeholder::Shape({-1, -1}));
    auto add = ops::Add(s.WithOpName("add"), feed, feed);
    ops::Identity(s.WithOpName("TensorRTOutputPH_0"), add);

    // Serialize the graph. TRTEngineOp will convert it using dynamic mode.
    GraphDef graph_def;
    TF_ASSERT_OK(s.ToGraphDef(&graph_def));
    PartialTensorShape shape({-1, -1});

    // Create the op.
    OpsTestBase::SetDevice(DEVICE_GPU, std::move(device));
    TF_ASSERT_OK(NodeDefBuilder("myop", "TRTEngineOp")
                     .Input(FakeInput(1, dtype))
                     .Attr("input_shapes", {shape})
                     .Attr("output_shapes", {shape})
                     .Attr("static_engine", false)
                     .Attr("segment_funcdef_name", "")  // no native fallback
                     .Attr("serialized_segment", graph_def.SerializeAsString())
                     .Attr("calibration_data", "")
                     .Attr("max_cached_engines_count", max_cached_engines_count)
                     .Attr("workspace_size_bytes", 1 << 20)
                     .Attr("precision_mode", "FP32")
                     .Attr("use_calibration", false)
                     .Attr("OutT", {dtype})
                     .Finalize(OpsTestBase::node_def()));
    TF_ASSERT_OK(OpsTestBase::InitOp());
  }

  template <typename T>
  void AddSimpleInput(const TensorShape& shape) {
    std::vector<T> input(shape.num_elements());
    std::iota(input.begin(), input.end(), T(0));
    OpsTestBase::AddInputFromArray<T>(shape, input);
  }

  void ResetInputs() {
    inputs_.clear();
    gtl::STLDeleteElements(&tensors_);
  }
};

TEST_F(TRTEngineOpTestBase, dynamic_shapes) {
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/4);

  // Execute the op with batch size > 1.
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({2, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(device_->resource_manager()->Lookup("TF-TRT-Engine-Cache",
                                                   "myop", &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // It should contain only one engine.
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  EXPECT_THAT(cache->begin()->first, ElementsAre(TensorShape({2, 2})));

  // Execute the op with batch size 1. It should reuse existing engine to
  // execute.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({1, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(1, cache->size());
  EXPECT_THAT(cache->begin()->first, ElementsAre(TensorShape({2, 2})));

  // Execute the op with a larger batch size.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({3, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(2, cache->size());
  EXPECT_THAT(cache->begin()->first, ElementsAre(TensorShape({3, 2})));
  EXPECT_THAT((++cache->begin())->first, ElementsAre(TensorShape({2, 2})));

  // Execute the op with an input that has different non-batch dimension.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({10, 10}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  // Execute it again with an input that has the same non-batch dimension but
  // smallest batch size. It should find the correct engine to use.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({1, 10}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(3, cache->size());  // Should only create 3 engines in total.
  auto iter = cache->begin();
  EXPECT_THAT(iter->first, ElementsAre(TensorShape({10, 10})));
  EXPECT_THAT((++iter)->first, ElementsAre(TensorShape({3, 2})));
  EXPECT_THAT((++iter)->first, ElementsAre(TensorShape({2, 2})));
}

template <typename T>
class TRTEngineOpTest : public TRTEngineOpTestBase {};

using TypeList = ::testing::Types<float, Eigen::half>;
TYPED_TEST_SUITE(TRTEngineOpTest, TypeList);

TYPED_TEST(TRTEngineOpTest, Basic) {
  TRTEngineOpTestBase::AddSimpleTrtOp(DataTypeToEnum<TypeParam>::v());

  // Execute the op.
  OpsTestBase::AddInputFromArray<TypeParam>(TensorShape({1, 2}),
                                            {TypeParam(0.0f), TypeParam(1.0f)});
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Verify the result.
  Tensor* output = OpsTestBase::GetOutput(0);
  EXPECT_THAT(
      absl::Span<const TypeParam>(output->template flat<TypeParam>().data(),
                                  output->NumElements()),
      ElementsAre(TypeParam(0.0f), TypeParam(2.0f)));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
