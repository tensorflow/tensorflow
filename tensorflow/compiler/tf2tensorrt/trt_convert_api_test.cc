/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace tensorrt {

struct TestParam {
  TfTrtConversionParams conv_params;
  std::vector<std::vector<int64>> input_shapes;
};

class TrtConverterTest : public ::testing::TestWithParam<TestParam> {
 protected:
  TrtConverterTest() { param_ = GetParam(); }

  // Returns the following graph: output = input * [42, 137] + input
  MetaGraphDef GetModel() {
    PartialTensorShape shape({-1, 2});

    Scope root = Scope::NewRootScope();
    auto c = ops::Const(root.WithOpName("my_const"), {{42.0f, 137.0f}});
    const auto attrs = ops::Placeholder::Shape(shape);
    auto x = ops::Placeholder(root.WithOpName("input"), DT_FLOAT, attrs);
    auto y = ops::Mul(root.WithOpName("my_mul"), x, c);
    auto z = ops::Add(root.WithOpName("my_add"), x, y);
    auto q = ops::Identity(root.WithOpName("output"), z);

    MetaGraphDef out;
    TF_CHECK_OK(root.ToGraphDef(out.mutable_graph_def()));

    TensorShapeProto shape_proto;
    shape.AsProto(&shape_proto);
    SignatureDef signature_def;
    (*signature_def.mutable_inputs())["input"].set_name("input:0");
    (*signature_def.mutable_inputs())["input"].set_dtype(DT_FLOAT);
    (*signature_def.mutable_inputs())["input"].mutable_tensor_shape()->CopyFrom(
        shape_proto);
    (*signature_def.mutable_outputs())["output"].set_name("output:0");
    (*signature_def.mutable_outputs())["output"].set_dtype(DT_FLOAT);
    (*signature_def.mutable_outputs())["output"]
        .mutable_tensor_shape()
        ->CopyFrom(shape_proto);
    (*out.mutable_signature_def())["serving_default"] = signature_def;

    VLOG(2) << signature_def.DebugString();
    return out;
  }

  // Confirms that we have a TRT node with the correct attributes.
  void CheckTrtNode(const GraphDef& converted_graph_def) {
    int n_trt_ops = 0;
    string op_name{"TRTEngineOp"};
    for (const auto& node : converted_graph_def.node()) {
      if (!op_name.compare(node.op())) {
        n_trt_ops++;
        const auto& attr = node.attr();
        EXPECT_EQ(attr.at("static_engine").b(),
                  param_.conv_params.convert_to_static_engine);
        if (param_.conv_params.convert_to_static_engine) {
          VLOG(2) << "Found serialized segment with size "
                  << attr.at("serialized_segment").s().size();
          EXPECT_GT(attr.at("serialized_segment").s().size(), 0);
        }
      }
    }
    EXPECT_EQ(n_trt_ops, 1);
  }

  void ConvertAndRun() {
    MetaGraphDef meta_graph_def = GetModel();

    // Create a list of input tensors, they will be used to build the engines.
    std::vector<std::vector<Tensor>> input_tensors;
    for (const std::vector<int64>& shape : param_.input_shapes) {
      Tensor tensor(DT_FLOAT, TensorShape(shape));
      test::FillIota(&tensor, 1.0f);
      input_tensors.push_back({tensor});
    }

    StatusOr<GraphDef> result = tensorrt::ConvertAndBuild(
        meta_graph_def.graph_def(), {"input"}, {"output"}, input_tensors,
        param_.conv_params);
    TF_ASSERT_OK(result.status());
    const GraphDef& converted_graph_def = result.ValueOrDie();
    CheckTrtNode(converted_graph_def);

    // Create sessions to execute the original and the converted graphs.
    Session* p_session = nullptr;
    TF_EXPECT_OK(NewSession(SessionOptions(), &p_session));
    std::unique_ptr<tensorflow::Session> session(p_session);
    TF_EXPECT_OK(session->Create(meta_graph_def.graph_def()));

    p_session = nullptr;
    TF_EXPECT_OK(NewSession(SessionOptions(), &p_session));
    std::unique_ptr<tensorflow::Session> trt_session(p_session);
    TF_EXPECT_OK(trt_session->Create(converted_graph_def));

    // Run models and compare the output.
    for (const std::vector<Tensor>& input : input_tensors) {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(
          session->Run({{"input", input.at(0)}}, {"output"}, {}, &outputs));
      std::cout << outputs.at(0).DebugString() << std::endl;

      std::vector<Tensor> trt_outputs;
      TF_EXPECT_OK(trt_session->Run({{"input", input.at(0)}}, {"output"}, {},
                                    &trt_outputs));
      std::cout << trt_outputs.at(0).DebugString() << std::endl;
      ASSERT_EQ(outputs.size(), 1);
      ASSERT_EQ(trt_outputs.size(), 1);
      tensorflow::test::ExpectEqual(outputs[0], trt_outputs[0]);
    }
  }
  TestParam param_;
};

INSTANTIATE_TEST_CASE_P(
    TrtConverterTestInstantiation, TrtConverterTest,
    ::testing::Values(
        // Dynamic shape mode test with conver_to_static_engine=true.
        TestParam{TfTrtConversionParams{
                      1 << 20,  // max workspace size
                      TrtPrecisionMode::FP32,
                      3,      // minimum_segment_size
                      1,      // max_cached_engines
                      false,  // use_calibration
                      true,   // use_dynamic_shape
                      ProfileStrategy::kOptimal,
                      true,  // allow_build_at_runtime
                      true   // convert_to_static_engine
                  },
                  {{1, 2}, {4, 2}}},
        // Implicit batch mode test with conver_to_static_engine=true.
        TestParam{TfTrtConversionParams{
                      1 << 20,  // max workspace size
                      TrtPrecisionMode::FP16,
                      3,      // minimum_segment_size
                      1,      // max_cached_engines
                      false,  // use_calibration
                      false,  // use_dynamic_shape
                      ProfileStrategy::kRange,
                      true,  // allow_build_at_runtime
                      true   // convert_to_static_engine
                  },
                  {{1, 2}}},
        // Dynamic shape mode test convert_to_static_engine=false: we cannot
        // save the engines, therefore we do not generate profiles. A single
        // engine will be built during runtime, with profile that matches the
        // first shape ({1,2}). The second shape will run as native segment.
        TestParam{TfTrtConversionParams{
                      1 << 20,  // max workspace size
                      TrtPrecisionMode::FP32,
                      3,      // minimum_segment_size
                      1,      // max_cached_engines
                      false,  // use_calibration
                      true,   // use_dynamic_shape
                      ProfileStrategy::kOptimal,
                      true,  // allow_build_at_runtime
                      false  // convert_to_static_engine
                  },
                  {{1, 2}, {4, 2}}},
        // Implicit batch mode test with convert_to_static_engine=false. We
        // will have two engines in the cache to handle the two shapes.
        TestParam{TfTrtConversionParams{
                      1 << 20,  // max workspace size
                      TrtPrecisionMode::FP16,
                      3,      // minimum_segment_size
                      2,      // max_cached_engines
                      false,  // use_calibration
                      false,  // use_dynamic_shape
                      ProfileStrategy::kRange,
                      true,  // allow_build_at_runtime
                      false  // convert_to_static_engine
                  },
                  {{1, 2}, {4, 2}}}));

TEST_P(TrtConverterTest, Basic) { ConvertAndRun(); }

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
