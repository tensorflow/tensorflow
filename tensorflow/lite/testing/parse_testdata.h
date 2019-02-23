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
#include "tensorflow/lite/interpreter.h"
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
// The input format is similar to an ASCII proto:
//   // Loads model 'add.bin' from the TestRunner's model directory.
//   load_model: "add.bin"
//   // Changes the shape of inputs, provided in the same order they appear
//   // in the model.
//   reshape {
//     input: "1,224,224,3"
//     input: "1,3,4,1"
//   }
//   // Fills the given persistent tensors with zeros.
//   init_state: 0,1,2,3
//   // Invokes the interpreter with the given input and checks that it
//   // produces the expected output. Inputs and outputs should be specified in
//   // the order they appear in the model.
//   invoke {
//     input: "1,2,3,4,56"
//     input: "0.1,0.2,0.3,4.3,56.4"
//     output: "12,3,4,545,3"
//     output: "0.01,0.02"
//   }
bool ParseAndRunTests(std::istream* input, TestRunner* test_runner,
                      int max_invocations = -1);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_PARSE_TESTDATA_H_
