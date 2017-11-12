
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

  if (input_layer_type == "float") {
    // 0-th dimension is batch.
    // FillRandomValue<float>(
    //     interpreter->typed_tensor<float>(input),
    //     std::vector<int>(sizes.begin() + 1, sizes.end()),
    //     []() { return static_cast<float>(rand()) / RAND_MAX - 0.5f; });
  } else if (input_layer_type == "uint8") {
    // 0-th dimension is batch.
    // FillRandomValue<uint8_t>(
    //     interpreter->typed_tensor<uint8_t>(input),
    //     std::vector<int>(sizes.begin() + 1, sizes.end()),
    //     []() { return static_cast<uint8_t>(rand()) % 255; });
  } else if (input_layer_type == "string") {
    tflite::DynamicBuffer buffer;
    // FillRandomString(&buffer, sizes, []() {
    //   return "we're have some friends over saturday to hang out in the yard";
    // });
    buffer.WriteToTensor(interpreter->tensor(input));
  } else {
    LOG(FATAL) << "Unknown input type: " << input_layer_type;
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
