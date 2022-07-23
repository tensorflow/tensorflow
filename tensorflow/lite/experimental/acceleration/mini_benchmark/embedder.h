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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_EMBEDDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_EMBEDDER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
namespace tflite {
namespace acceleration {
// Class to embed a mini-benchmark into a tflite file.
//
// The inputs are:
// - 'main_model': the actual inference graph (e.g., mobilenet classifier)
// - 'jpeg_data': jpeg images used as test data.
// - 'validation_model': a graph that takes as input two sets of values (the
// known-good main model output and the to-be-tested main model output) and
// produces 2 or more outputs where one must be called 'ok' (whether the
// results are good enough) and rest are metrics that were used to determine
// 'ok' and can be used for debugging/telemetry.
// (Known good outputs are produced inside this class, i.e. running TFLite CPU
// on the build host).
//
// The output is:
// - A new benchmark model which has 3 subgraphs. The 'main_model' subgraph, a
// new 'validate' subgraph that invokes the other two subgraphs when required,
// and the 'validation_model' subgraph.
// - The model output is the output of 'validation_model' + output of
// 'main_model'
// - This model has additional buffers that store the 'jpeg_data' and the actual
// outputs.
// - The 'main_model' subgraph is fed the 'jpeg_data' and produces an output
// which is used by the 'validation_model' with the known-good outputs to
// evaluate the model.
// - This entire process is handled end-to-end by the 'validate' subgraph using
// two custom ops: 'validate/call' (implemented in :call in this directory) and
// 'validate/decode_jpeg' (being implemented).
//
// Constraints on inputs:
// - 'main_model' must have a single input of dimensions
//   [1, height, width, 1 or 3]
// - the images encoded in 'jpeg_data' must have same height, width and channels
//   as 'main_model' input
// - the 'validation_model' must have inputs equal to 'main_model' outputs
//   duplicated (e.g, if 'main_model' has outputs with dimensions
//   [1, 10] and [1, 20]; the 'validation_model' must have inputs with
//   dimensions [1, 10], [1, 20], [1, 10], [1, 20]).
// - the 'validation_model' must have 2 or more outputs, and one of them must be
//   called 'ok'.
// - all inputs and outputs must be tensors (not scalars).
//
// TODO(b/172541832):
// - Mark the validation graph so that it's not delegated in the inference case.
// - Allow known-good outputs to be given rather than always being calculated
// inside this class.
class Embedder {
 public:
  // Construct Embedder with inputs. The Model* inputs are owned by the caller
  // and must outlive the Embedder. The `schema` must contain the tflite
  // flatbuffer schema. If the model is quantized, scale and zero_point are
  // ignored.
  Embedder(const Model* main_model, const std::vector<std::string>& jpeg_data,
           float scale, int64_t zero_point, const Model* validation_model,
           const reflection::Schema* schema,
           bool use_ondevice_cpu_for_golden = false);
  // Construct the output model. Calls Finish() on 'fbb'.
  // The 'resolver' must have the call and decode_jpeg ops from this directory
  // registered as 'validation/call' and 'validation/decode_jpeg'.
  absl::Status CreateModelWithEmbeddedValidation(
      flatbuffers::FlatBufferBuilder* fbb,
      ::tflite::ops::builtin::BuiltinOpResolver* resolver);
  // Check that the inputs fulfill the constraints. Called automatically as part
  // of CreateModelWithEmbeddedValidation.
  absl::Status ValidateInputs();

 private:
  const Model* main_model_;
  std::vector<std::string> jpeg_data_;
  int32_t jpeg_output_channels_;
  float scale_;
  int64_t zero_point_;
  const Model* validation_model_;
  const reflection::Schema* schema_;
  bool use_ondevice_cpu_for_golden_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_EMBEDDER_H_
