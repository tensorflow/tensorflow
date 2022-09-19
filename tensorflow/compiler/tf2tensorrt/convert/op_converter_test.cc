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
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"

#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

class ExampleOpConverter : public OpConverterBase<ExampleOpConverter> {
 public:
  explicit ExampleOpConverter(const OpConverterParams* params)
      : OpConverterBase<ExampleOpConverter>(params, {DataType::DT_FLOAT}) {}

  static constexpr const char* NodeDefDataTypeAttributeName() {
    return "data_type";
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("input_tensor", TrtInputArg::kTensor),
        InputArgSpec::Create("weight", TrtInputArg::kWeight)};
  }

  Status Validate() { return Status::OK(); }

  Status Convert() {
    AddOutput(TRT_TensorOrWeights(nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims{1, {1, 1, 1}}, 1));
    return Status::OK();
  }
};

TEST(TestOpConverterBase, TestOpConverterBase) {
  // Register a converter which uses the base converter class.
  GetOpConverterRegistry()->Register(
      "FakeFunc", 1, MakeConverterFunction<ExampleOpConverter>());

  NodeDef def;
  def.set_op("FakeFunc");
  auto converter = Converter::Create(TrtPrecisionMode::FP32, false,
                                     Logger::GetLogger(), false, "test_engine");
  EXPECT_THAT(converter, IsOk());

  // Base class should check attribute with key given by
  // Impl::NodeDefDataTypeAttributeName().
  Status conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(conversion_status,
              StatusIs(error::INVALID_ARGUMENT,
                       HasSubstr("Attribute with name data_type not found")));

  // Add partial inputs to the node and make the converter aware.
  def.mutable_input()->Add("input1");
  conversion_status = (*converter)
                          ->AddInputTensor("input1", nvinfer1::DataType::kFLOAT,
                                           nvinfer1::Dims{4, {1, 1, 1, 1}}, 1);
  EXPECT_THAT(conversion_status, IsOk());

  // Base class method should check number of inputs.
  AddNodeAttr("data_type", DT_FLOAT, &def);
  conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(conversion_status, StatusIs(error::INTERNAL));

  // Add second input to the node and make the converter aware.
  def.mutable_input()->Add("input2");
  conversion_status = (*converter)
                          ->AddInputTensor("input2", nvinfer1::DataType::kFLOAT,
                                           nvinfer1::Dims{4, {1, 1, 1, 1}}, 1);
  EXPECT_THAT(conversion_status, IsOk());

  // Base class validation should check the type (Constant or Tensor) of the
  // inputs.
  conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(
      conversion_status,
      StatusIs(error::UNIMPLEMENTED,
               HasSubstr("input \"weight\" for FakeFunc must be a constant")));

  // Correct input2 so that it is a weight.
  (*converter)->TensorsMap().erase("input2");
  (*converter)
      ->TensorsMap()
      .insert(std::make_pair("input2", TRT_TensorOrWeights(TRT_ShapedWeights(
                                           nvinfer1::DataType::kFLOAT))));

  // With the correct input types, check that the converter is called and sets
  // one output tensor.
  conversion_status = (*converter)->ConvertNode(def);
  EXPECT_THAT(conversion_status, IsOk());
  EXPECT_EQ((*converter)->TensorsMap().size(), 3U);

  GetOpConverterRegistry()->Clear("FakeFunc");
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif
