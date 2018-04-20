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

#include "tensorflow/contrib/lite/toco/toco_saved_model.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/contrib/lite/toco/model_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace toco {
namespace {

using tensorflow::ops::Add;
using tensorflow::ops::Const;
using tensorflow::ops::FakeQuantWithMinMaxArgs;
using tensorflow::ops::Placeholder;

class TocoSavedModelTest : public ::testing::Test {
 protected:
  // Calls functions to process cmdline arguments and calls ParseMetaData.
  // ParseMetaData parses input_arrays, output_arrays, and gets metadata from
  // SavedModel it is not defined in the cmdline arguments.
  void ProcessGraphDefMetadata(const std::unordered_set<string>& inputs,
                               const std::unordered_set<string>& outputs,
                               const tensorflow::GraphDef& graph_def) {
    ReadTocoFlagsFromCommandLineFlags(parsed_toco_flags_, &toco_flags_);
    ReadModelFlagsFromCommandLineFlags(parsed_model_flags_, &model_flags_);
    ParseMetaData(graph_def, inputs, outputs, parsed_toco_flags_,
                  parsed_model_flags_, &toco_flags_, &model_flags_);
  }

  // Gets the GraphDef from the SavedModelBundle and processes metadata.
  void ProcessSavedModelMetadata(const std::unordered_set<string>& inputs,
                                 const std::unordered_set<string>& outputs) {
    const tensorflow::GraphDef graph_def = bundle_.meta_graph_def.graph_def();
    ProcessGraphDefMetadata(inputs, outputs, graph_def);
  }

  // Returns a GraphDef representing a simple float model with a single input.
  tensorflow::GraphDef GetFloatGraphDef(const std::vector<int64>& shape) {
    tensorflow::GraphDef graph_def;
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    tensorflow::Output input =
        Placeholder(scope.WithOpName("input"), tensorflow::DT_FLOAT,
                    Placeholder::Shape(tensorflow::PartialTensorShape(shape)));
    tensorflow::Output zero = Const(scope.WithOpName("zero"), 0.0f, {});
    tensorflow::Output add = Add(scope.WithOpName("add"), input, zero);

    TF_EXPECT_OK(scope.ToGraphDef(&graph_def));
    return graph_def;
  }

  // Returns a GraphDef representing a simple float model with two inputs.
  tensorflow::GraphDef GetComplexFloatGraphDef() {
    tensorflow::GraphDef graph_def;
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    tensorflow::Output inputA =
        Placeholder(scope.WithOpName("inputA"), tensorflow::DT_FLOAT,
                    Placeholder::Shape(tensorflow::TensorShape({1, 3, 3, 1})));
    tensorflow::Output inputB =
        Placeholder(scope.WithOpName("inputB"), tensorflow::DT_FLOAT,
                    Placeholder::Shape(tensorflow::TensorShape({1, 3, 3, 1})));
    tensorflow::Output add = Add(scope.WithOpName("add"), inputB, inputA);

    TF_EXPECT_OK(scope.ToGraphDef(&graph_def));
    return graph_def;
  }

  // Returns a GraphDef representing a simple quantized model.
  tensorflow::GraphDef GetQuantizedGraphDef() {
    tensorflow::GraphDef graph_def;
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    tensorflow::Output input =
        Placeholder(scope.WithOpName("input"), tensorflow::DT_FLOAT,
                    Placeholder::Shape(tensorflow::TensorShape({1, 3, 3, 1})));
    tensorflow::Output zero = Const(scope.WithOpName("zero"), 0.0f, {});
    tensorflow::Output fake_quant =
        FakeQuantWithMinMaxArgs(scope.WithOpName("quant"), zero);
    tensorflow::Output add = Add(scope.WithOpName("add"), input, fake_quant);

    TF_EXPECT_OK(scope.ToGraphDef(&graph_def));
    return graph_def;
  }

  // Gets the values in the input_arrays flag.
  std::vector<string> GetInputArrays() {
    std::vector<string> actual;
    for (const auto& input : model_flags_.input_arrays()) {
      actual.push_back(input.name());
    }
    return actual;
  }

  // Gets the values in the output_arrays flag.
  std::vector<string> GetOutputArrays() {
    std::vector<string> actual(model_flags_.output_arrays().begin(),
                               model_flags_.output_arrays().end());
    return actual;
  }

  // Gets the shape of the given input array.
  string GetInputShape(const string& input_array) {
    for (const auto& input : model_flags_.input_arrays()) {
      if (input.name() == input_array) {
        std::vector<string> dims;
        for (int idx = 0; idx < input.shape().dims_size(); ++idx) {
          dims.push_back(std::to_string(input.shape().dims(idx)));
        }
        return absl::StrJoin(dims, ",");
      }
    }
    return "";
  }

  tensorflow::SavedModelBundle bundle_;
  ParsedTocoFlags parsed_toco_flags_;
  ParsedModelFlags parsed_model_flags_;
  TocoFlags toco_flags_;
  ModelFlags model_flags_;
};

// Tests if input_arrays, output_arrays, inference_type, and output_arrays are
// added to ModelFlags if they are not specified in cmdline arguments.
// Tests if the default batch size replaces a -1 in the first dimension.
TEST_F(TocoSavedModelTest, NoCmdLine) {
  tensorflow::GraphDef graph_def = GetFloatGraphDef({-1, 3, 3, 1});

  ProcessGraphDefMetadata({"input"}, {"add"}, graph_def);
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"input"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"add"}));
  EXPECT_EQ(GetInputShape("input"), "1,3,3,1");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Tests if the order of input_arrays and output_arrays is deterministic when
// they are taken from the SavedModel.
TEST_F(TocoSavedModelTest, NoCmdLineMultipleArrays) {
  tensorflow::GraphDef graph_def = GetComplexFloatGraphDef();

  // Note: The model does not have two outputs. However, the function does not
  // need an accurate output_array list. This is only meant to test order.
  ProcessGraphDefMetadata({"inputB", "inputA"}, {"add", "invalid"}, graph_def);
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"inputA", "inputB"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"add", "invalid"}));
  EXPECT_EQ(GetInputShape("inputA"), "1,3,3,1");
  EXPECT_EQ(GetInputShape("inputB"), "1,3,3,1");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Tests if input_shapes is inferred when input_arrays is passed in via cmdline
// arguments.
TEST_F(TocoSavedModelTest, InputNameWithoutInputShape) {
  parsed_model_flags_.input_arrays.bind()("input");
  tensorflow::GraphDef graph_def = GetFloatGraphDef({2, 3, 3, 1});

  ProcessGraphDefMetadata({"not_used_input"}, {"add"}, graph_def);
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"input"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"add"}));
  EXPECT_EQ(GetInputShape("input"), "2,3,3,1");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Ensures a failure occurs when input_shapes is defined without input_arrays.
TEST_F(TocoSavedModelTest, InputShapeWithoutInputName) {
  parsed_model_flags_.input_shapes.bind()("1,224,224,1:9,12");
  tensorflow::GraphDef graph_def = GetFloatGraphDef({1, 3, 3, 1});

  EXPECT_DEATH(ProcessGraphDefMetadata({"input"}, {"add"}, graph_def),
               "failed: input_shapes.size\\(\\) == "
               "model_flags->input_arrays_size\\(\\)");
}

// Tests if the cmdline values of input_arrays, input_shapes are used when
// specified with an empty GraphDef.
TEST_F(TocoSavedModelTest, InputArraysCmdLine) {
  parsed_model_flags_.input_arrays.bind()("inputA,inputB");
  parsed_model_flags_.input_shapes.bind()("1,224,224,1:9,12");

  ProcessSavedModelMetadata({"input0", "input1"}, {"output0", "output1"});
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"inputA", "inputB"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"output0", "output1"}));
  EXPECT_EQ(GetInputShape("inputA"), "1,224,224,1");
  EXPECT_EQ(GetInputShape("inputB"), "9,12");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Tests if the cmdline values of input_arrays, input_shapes are used when
// specified even if values exist within the GraphDef.
TEST_F(TocoSavedModelTest, InputArraysCmdLineWithGraphDef) {
  parsed_model_flags_.input_arrays.bind()("inputA");
  parsed_model_flags_.input_shapes.bind()("1,224,224,1");
  tensorflow::GraphDef graph_def = GetFloatGraphDef({1, 3, 3, 1});

  ProcessGraphDefMetadata({"inputA"}, {"add"}, graph_def);
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"inputA"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"add"}));
  EXPECT_EQ(GetInputShape("inputA"), "1,224,224,1");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Tests if the cmdline values of input_arrays, input_shapes, inference_type,
// and output_arrays are used when specified with an empty GraphDef.
TEST_F(TocoSavedModelTest, AllParamsCmdLine) {
  parsed_model_flags_.input_arrays.bind()("inputA,inputB");
  parsed_model_flags_.output_arrays.bind()("outputA,outputB");
  parsed_model_flags_.input_shapes.bind()("1,224,224,1:9,12");
  parsed_toco_flags_.inference_type.bind()("FLOAT");

  ProcessSavedModelMetadata({"input0", "input1"}, {"output0", "output1"});
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"inputA", "inputB"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"outputA", "outputB"}));
  EXPECT_EQ(GetInputShape("inputA"), "1,224,224,1");
  EXPECT_EQ(GetInputShape("inputB"), "9,12");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Tests if a quantized graph gives the correct values assuming type is passed
// in via command line.
TEST_F(TocoSavedModelTest, QuantizedNoCmdLine) {
  parsed_toco_flags_.inference_type.bind()("QUANTIZED_UINT8");
  tensorflow::GraphDef graph_def = GetQuantizedGraphDef();

  ProcessGraphDefMetadata({"input"}, {"add"}, graph_def);
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"input"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"add"}));
  EXPECT_EQ(GetInputShape("input"), "1,3,3,1");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::QUANTIZED_UINT8);
}

// Tests if the provided batch size replaces a -1 in the first dimension of
// input shape.
TEST_F(TocoSavedModelTest, MissingShapeParameterValid) {
  parsed_model_flags_.batch_size.bind()(3);
  tensorflow::GraphDef graph_def = GetFloatGraphDef({-1, 3, 3, 1});

  ProcessGraphDefMetadata({"input"}, {"add"}, graph_def);
  EXPECT_EQ(GetInputArrays(), std::vector<string>({"input"}));
  EXPECT_EQ(GetOutputArrays(), std::vector<string>({"add"}));
  EXPECT_EQ(GetInputShape("input"), "3,3,3,1");
  EXPECT_EQ(toco_flags_.inference_type(), IODataType::FLOAT);
}

// Ensures a failure occurs if there is a -1 in a dimension aside from the first
// position of input shape.
TEST_F(TocoSavedModelTest, MissingShapeParameterInvalid) {
  parsed_model_flags_.batch_size.bind()(3);
  tensorflow::GraphDef graph_def = GetFloatGraphDef({1, -1, 3, 1});

  EXPECT_DEATH(ProcessGraphDefMetadata({"input"}, {"add"}, graph_def),
               "A valid input shape was not found for input 'input'.");
}

}  // namespace
}  // namespace toco
