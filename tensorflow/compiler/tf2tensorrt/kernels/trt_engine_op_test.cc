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
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {
using ::absl::StrCat;
using ::testing::ElementsAre;

class TRTEngineOpTestBase : public OpsTestBase {
 public:
  void AddSimpleTrtOp(DataType dtype, int max_cached_engines_count = 1) {
    // Create the GPU device.
    std::unique_ptr<Device> device(
        DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));

    // Create simple TF graph.
    Scope s = Scope::NewRootScope();
    auto feed = ops::_Arg(s.WithOpName("TensorRTInputPH_0"), dtype, 0);
    auto add = ops::Add(s.WithOpName("add"), feed, feed);
    ops::_Retval(s.WithOpName("TensorRTOutputPH_0"), add, 0);

    // Serialize the graph. TRTEngineOp will convert it using dynamic mode.
    GraphDef graph_def;
    TF_ASSERT_OK(s.ToGraphDef(&graph_def));
    Graph* graph = s.graph();
    const char* op_name = "myop";
    TF_ASSERT_OK(
        convert::RegisterGraphToFunctionLibrary(graph_def, graph, op_name));
    TF_ASSERT_OK(flib_def_->AddLibrary(graph->flib_def()));

    PartialTensorShape shape({-1, -1});

    // Create the op.
    OpsTestBase::SetDevice(DEVICE_GPU, std::move(device));
    NameAttrList function;
    function.set_name(StrCat(op_name, "_native_segment"));
    TF_ASSERT_OK(NodeDefBuilder(op_name, "TRTEngineOp")
                     .Input(FakeInput(1, dtype))
                     .Attr("input_shapes", {shape})
                     .Attr("output_shapes", {shape})
                     .Attr("static_engine", false)
                     .Attr("segment_func", function)
                     .Attr("serialized_segment", "")
                     .Attr("calibration_data", "")
                     .Attr("max_cached_engines_count", max_cached_engines_count)
                     .Attr("workspace_size_bytes", 1 << 20)
                     .Attr("precision_mode", "FP32")
                     .Attr("use_calibration", false)
                     .Attr("use_implicit_batch", true)
                     .Attr("OutT", {dtype})
                     .Finalize(OpsTestBase::node_def()));
    TF_ASSERT_OK(InitOpWithFunctionLibrary());
  }

  template <typename T>
  void AddSimpleInput(const TensorShape& shape) {
    std::vector<T> input(shape.num_elements());
    std::iota(input.begin(), input.end(), T(0));
    OpsTestBase::AddInputFromArray<T>(shape, input);
  }

  void ResetInputs() {
    inputs_.clear();
    for (auto& temp : tensors_) {
      delete temp;
    }
    tensors_.clear();
  }

 private:
  Status InitOpWithFunctionLibrary() {
    OpKernel* kernel = nullptr;
    Status status = CreateOpKernel(device_type_, device_, allocator(),
                                   pflr_->GetFLR(device_->name()), node_def_,
                                   TF_GRAPH_DEF_VERSION, &kernel);
    kernel_ = std::unique_ptr<OpKernel>(kernel);
    if (kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }
};

TEST_F(TRTEngineOpTestBase, DynamicShapes) {
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/4);

  // Execute the op with batch size > 1.
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({2, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(
      device_->resource_manager()->Lookup("TF-TRT", "myop", &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // It should contain only one engine.
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));

  // Execute the op with batch size 1. It should reuse existing engine to
  // execute.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({1, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(1, cache->size());
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));

  // Execute the op with a larger batch size.
  ResetInputs();
  TRTEngineOpTestBase::AddSimpleInput<float>(TensorShape({3, 2}));
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());
  EXPECT_EQ(2, cache->size());
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));
  EXPECT_EQ(1, cache->count({TensorShape({3, 2})}));

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
  EXPECT_EQ(1, cache->count({TensorShape({2, 2})}));
  EXPECT_EQ(1, cache->count({TensorShape({3, 2})}));
  EXPECT_EQ(1, cache->count({TensorShape({10, 10})}));
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
