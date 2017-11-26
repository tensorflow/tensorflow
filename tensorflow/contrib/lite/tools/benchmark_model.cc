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

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#ifdef TFLITE_CUSTOM_OPS_HEADER
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
#endif

#define LOG(x) std::cerr
#define CHECK(x) if (!(x)) { LOG(ERROR) << #x << "failed"; exit(1); }

namespace tensorflow {
namespace benchmark_tflite_model {

std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;

void InitImpl(const std::string& graph, const std::vector<int>& sizes,
              const std::string& input_layer_type, int num_threads) {
  CHECK(graph.c_str());

  model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << graph;
  }
  LOG(INFO) << "Loaded model " << graph;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter";
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  int input = interpreter->inputs()[0];

  if (input_layer_type != "string") {
    interpreter->ResizeInputTensor(input, sizes);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }
}

int Main(int argc, char** argv) {
  InitImpl("", {}, "", 1);
  return 0;
}

}  // namespace benchmark_tflite_model
}  // namespace tensorflow

int main(int argc, char** argv) {
  return tensorflow::benchmark_tflite_model::Main(argc, argv);
}
