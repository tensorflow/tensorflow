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
#include "tensorflow/lite/toco/toco_convert.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/toco/toco_port.h"

namespace toco {
namespace {

TEST(TocoTest, MissingInputFile) {
  ParsedTocoFlags toco_flags;
  ParsedModelFlags model_flags;
  EXPECT_DEATH(Convert(toco_flags, model_flags).ok(),
               "Missing required flag --input_file");
}

TEST(TocoTest, BadInputFormat) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  string input;
  string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "Unhandled input_format='FILE_FORMAT_UNKNOWN'");
}

TEST(TocoTest, MissingOuputArrays) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  string input;
  string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "This model does not define output arrays, so a --output_arrays "
               "flag must be given on the command-line");
}

TEST(TocoTest, BadOutputArray) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  model_flags.add_output_arrays("output1");
  string input;
  string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "Specified output array .output1. is not produced by any op "
               "in this graph. Is it a typo");
}

TEST(TocoTest, BadOutputFormat) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  model_flags.add_output_arrays("output1");
  string input = R"GraphDef(
    node {
      name: "output1"
      input: "input1"
      input: "input2"
      op: "Sub"
      attr { key: "T" value { type: DT_FLOAT } }
    }
  )GraphDef";

  string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "Unhandled output_format='FILE_FORMAT_UNKNOWN'");
}

TEST(TocoTest, SimpleFloatModel) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  toco_flags.set_output_format(TENSORFLOW_GRAPHDEF);

  // Inputs are automatically selected (but that might not be a good idea).
  model_flags.add_output_arrays("output1");
  string input = R"GraphDef(
    node {
      name: "input1"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_INT64 } }
    }
    node {
      name: "input2"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_INT64 } }
    }
    node {
      name: "output1"
      input: "input1"
      input: "input2"
      op: "Sub"
      attr { key: "T" value { type: DT_FLOAT } }
    }
  )GraphDef";

  string output;
  EXPECT_TRUE(Convert(input, toco_flags, model_flags, &output).ok());
  EXPECT_TRUE(!output.empty());
}

TEST(TocoTest, TransientStringTensors) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);

  // We need to do a couple of things to trigger the transient array
  // initialization code: output format must support memory planning, and the
  // input array must have a shape.
  toco_flags.set_output_format(TFLITE);

  toco::InputArray* input_1 = model_flags.add_input_arrays();
  input_1->set_name("input1");
  toco::InputArray* indices_1 = model_flags.add_input_arrays();
  indices_1->set_name("indices1");

  model_flags.add_output_arrays("output1");
  string input = R"GraphDef(
    node {
      name: "input1"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_STRING } }
      attr { key: "shape" value { shape { dim { size:1 }}}}
    }
    node {
      name: "indices1"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_INT64 } }
    }
    node {
      name: "intermediate1"
      op: "Gather"
      input: "input1"
      input: "indices1"
      attr { key: "Tparams" value { type: DT_STRING } }
      attr { key: "Tindices" value { type: DT_INT64 } }
    }
    node {
      name: "output1"
      op: "Gather"
      input: "intermediate1"
      input: "indices2"
      attr { key: "Tparams" value { type: DT_STRING } }
      attr { key: "Tindices" value { type: DT_INT64 } }
    }
  )GraphDef";

  string output;

  EXPECT_TRUE(Convert(input, toco_flags, model_flags, &output).ok());
  EXPECT_TRUE(!output.empty());
}

}  // namespace
}  // namespace toco

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
