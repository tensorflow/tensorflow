/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TESTING_PARSE_TESTDATA_H_
#define TENSORFLOW_LITE_TESTING_PARSE_TESTDATA_H_

#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/testing/test_runner.h"

namespace tflite {
namespace testing {

// Shape and data for a float tensor
struct FloatTensor {
  std::vector<int> shape;
  std::vector<float> flat_data;
};

// A prescribed input, output example
struct Example {
  std::vector<FloatTensor> inputs;
  std::vector<FloatTensor> outputs;
};

// Parses an example input and output file (used for unit tests)
TfLiteStatus ParseExamples(const char* filename,
                           std::vector<Example>* examples);

// Inputs Tensors into a TensorFlow lite interpreter. Note, this will run
// interpreter.AllocateTensors();
TfLiteStatus FeedExample(tflite::Interpreter* interpreter, const Example&);

// Check outputs against (already) evaluated result.
TfLiteStatus CheckOutputs(tflite::Interpreter* interpreter, const Example&);

// Parses a test description and feeds the given test runner with data.
// The input format is similar to a proto with the following schema:
//
// message TestMessage {
//   // Path to the model to load.
//   string load_model = 1;
//   // Names to initialize the tensor with zeros.
//   string init_state = 2;
//   message Reshape {
//     // Name of the input and csv string of shape of it.
//     map<string, string> input = 1;
//   }
//   repeated Reshape reshape = 3;
//   message Invoke {
//     // Name of this invoke.
//     string id = 1;
//     // Name of the input to the csv string of input value.
//     map<string, string> input = 2;
//     // Name of the output to the csv string of expected output value.
//     map<string, string> output = 3;
//     // Name of the output to the csv string of expected output shape.
//     map<string, string> output_shape = 4;
//   }
//   repeated Invoke invoke = 4;
// }
//
// An example of the ASCII proto:
//   // Loads model 'add.bin' from the TestRunner's model directory.
//   load_model: "add.bin"
//   // Changes the shape of inputs, provided in the same order they appear
//   // in the model, or `input_names` if specified.
//   reshape {
//     input {
//       key: "a"
//       value: "1,224,224,3"
//     }
//     input {
//       key: "b"
//       value: "1,3,4,1"
//     }
//   }
//   // Fills the given persistent tensors with zeros.
//   init_state: "a,b,c,d"
//   // Invokes the interpreter with the given input and checks that it
//   // produces the expected output. Inputs and outputs should be specified in
//   // the order they appear in the model, or `input_names` and `output_names`
//   // if specified.
//   invoke {
//     input {
//       key: "a"
//       value: "1,2,3,4,56"
//     }
//     input {
//       key: "b"
//       value: "0.1,0.2,0.3,4.3,56.4"
//     }
//     output {
//       key: "x"
//       value: "12,3,4,545,3,6"
//     }
//     output {
//       key: "y"
//       value: "0.01,0.02"
//     }
//     output_shape {
//       key: "x"
//       value: "2,3"
//     }
//     output_shape {
//       key: "y"
//       value: "1"
//     }
//   }
bool ParseAndRunTests(std::istream* input, TestRunner* test_runner,
                      int max_invocations = -1);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_PARSE_TESTDATA_H_
