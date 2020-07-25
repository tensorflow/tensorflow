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

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/version.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
using ::absl::StrCat;
using ::testing::ElementsAre;

class TRTEngineOpTestBase : public OpsTestBase {
 public:
  void AddSimpleTrtOp(DataType dtype, int max_cached_engines_count = 1,
                      PartialTensorShape shape = PartialTensorShape({-1, -1}),
                      bool use_implicit_batch = true,
                      bool allow_build_at_runtime = true) {
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

    // Create the op.
    // In implicit batch mode, the input shapes that we specify here are not
    // used for engine creation, we use the concrete shapes during inference
    // time for creating the engine.
    // In explicit batch mode, the input shapes attribute is used to define
    // the network for the TensorRT engine.
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
                     .Attr("_use_implicit_batch", use_implicit_batch)
                     .Attr("_allow_build_at_runtime", allow_build_at_runtime)
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
    auto flr = pflr_->GetFLR(device_->name());
    std::shared_ptr<const NodeProperties> props;
    Status status = NodeProperties::CreateFromNodeDef(
        node_def_, flr->GetFunctionLibraryDefinition(), &props);
    if (status.ok()) {
      status.Update(CreateOpKernel(device_type_, device_, allocator(), flr,
                                   props, TF_GRAPH_DEF_VERSION, &kernel));
    }
    kernel_ = std::unique_ptr<OpKernel>(kernel);
    if (kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }
};

TEST_F(TRTEngineOpTestBase, DynamicEngines) {
  // Test dynamic engine creation during inference time
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

TEST_F(TRTEngineOpTestBase, AllowBuildAtRuntime) {
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/1,
                                      PartialTensorShape({-1, -1}),
                                      /*use_implicit_batch=*/true,
                                      /*allow_build_at_runtime=*/false);

  // Execute the op
  TensorShape input_shape({2, 2});
  TRTEngineOpTestBase::AddSimpleInput<float>(input_shape);
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(
      device_->resource_manager()->Lookup("TF-TRT", "myop", &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // It should contain a placeholder with an empty cuda_engine (to mark that
  // engine creation was not successful for the given input shape).
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(1, cache->size());
  ASSERT_EQ(1, cache->count({input_shape}));
  EngineContext* ectx = cache->at({input_shape}).get();
  EXPECT_EQ(ectx->cuda_engine, nullptr);
}

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
TEST_F(TRTEngineOpTestBase, ExplicitBatch) {
  // Test inference in explicit batch mode with static input shapes. Static
  // shapes in this context means that the TensorRT knows all the input shapes
  // during engine creation time.
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/1,
                                      /*shape=*/PartialTensorShape({1, 2}),
                                      /*use_implicit_batch=*/false);

  TensorShape input_shape({1, 2});
  TRTEngineOpTestBase::AddSimpleInput<float>(input_shape);
  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(
      device_->resource_manager()->Lookup("TF-TRT", "myop", &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // Due to the way the engine lookup is implemented, explicit batch mode
  // requires profile generation. Currently profile generaton is not enabled in
  // this test therfore engine creation fails.
  //
  // TODO(Tamas) find a way to enable profile generation mode and test it
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(0, cache->size());
  // ASSERT_EQ(1, cache->count({input_shape}));
  // EngineContext* ectx = cache->at({input_shape}).get();
  // EXPECT_NE(ectx->cuda_engine, nullptr);
}

TEST_F(TRTEngineOpTestBase, DynamicShapes) {
  // Test inference in explicit batch mode with dynamic input shapes. Dynamic
  // shapes in this context means that some input shapes for TensorRT are
  // unknown during engine creation time. When we create the network, the
  // unknow shapes are repsesented as -1. Before we run inference, these shapes
  // have to be specified by calling setBindingDimensions.
  TRTEngineOpTestBase::AddSimpleTrtOp(DT_FLOAT, /*max_cached_engines_count=*/1,
                                      /*shape=*/PartialTensorShape({-1, -1}),
                                      /*use_implicit_batch=*/false);

  TensorShape input_shape({1, 2});
  TRTEngineOpTestBase::AddSimpleInput<float>(input_shape);

  TF_ASSERT_OK(OpsTestBase::RunOpKernel());

  // Get the engine cache.
  TRTEngineCacheResource* cache_resource = nullptr;
  TF_ASSERT_OK(
      device_->resource_manager()->Lookup("TF-TRT", "myop", &cache_resource));
  core::ScopedUnref sc(cache_resource);

  // We did not have profile generation mode therfore engine creation failed.
  // TODO(Tamas) find a way to enable profile generation mode and test it
  auto cache = &cache_resource->cache_;
  EXPECT_EQ(0, cache->size());
  // ASSERT_EQ(1, cache->count({input_shape}));
  // EngineContext* ectx = cache->at({input_shape}).get();
  // EXPECT_NE(ectx->cuda_engine, nullptr);
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
#endif

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
