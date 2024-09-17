/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_VALIDATION_GRAPH_BUILDER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_VALIDATION_GRAPH_BUILDER_H_

#include <stdint.h>
#include <stdlib.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/grafter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {

// Class for building the validation entry-point graph that calls into the main
// graph and a metrics graph. Like this (boxes are tensors with plural names
// meaning possibly multiple tensors, arrows are ops and numbers in parentheses
// are subgraph indices):
// +--------------------------------------+
// | Graph created by this class (1)      |
// |                                      |
// | +-----------input-+                  |
// | |jpeg input       |                  |
// | +-----+-----------+                  |
// |       |                              |
// |       | decode                       |
// |       v                              |
// | +-----+-----------+                  |
// | |quantized image  |                  |
// | +-----+-----------+                  |  +-----------------------+
// |       |                              |  |'main_model' (0)       |
// |       | dequantize (optional)        |  | +---------------+     |
// |       v                              |  | |input          +---+ |
// | +-----+-----------+                  |  | +---------------+   | |
// | |float image      |                  |  |                     ~ |
// | +-----+-----------+                  |  | +---------------+   | |
// |       |  call                        |  | |outputs        +<--+ |
// |       +<------------------------------->+ +---------------+     |
// |       v                              |  |                       |
// | +-----+-----output+ +---------input+ |  +-----------------------+
// | |actual outputs   | |golden outputs| |
// | +-----+-----------+ +-----------+--+ |
// |       |                         |    |
// |       | dequantize (optional)   |    |
// |       |                         |    |
// | +-----+-------------------------+-+  |
// | | dequantized actual and golden   |  |
// | | outputs (validation inputs)     |  |
// | +-----+---------------------------+  |  +-----------------------+
// |       |  call                        |  |'validation model' (2) |
// |       +<------------------------------->+                       |
// |       v                              |  | +---------------+     |
// | +-----+-----output+                  |  | |inputs         +---+ |
// | |results          |                  |  | +---------------+   | |
// | +-----------------+                  |  |                     ~ |
// |                                      |  | +---------------+   | |
// |                                      |  | |outputs        +<--+ |
// |                                      |  | +---------------+     |
// |                                      |  |                       |
// +--------------------------------------+  +-----------------------+
//
// It's important the 'main_model' has subgraph index 0 so that it is used as
// the primary subgraph by the TFLite interpreter. The other indices are
// arbitrary.
// TODO(b/172541832): Handle a main model with more than one subgraph.
//
// Note that the jpeg input is marked as an input in this graph, as TFLite
// graphs must have inputs. However, it will be pre-filled from the jpeg_data
// and doesn't need to be filled by the user of the model.
class ValidationGraphBuilder {
 public:
  ValidationGraphBuilder(const std::string& metric_prefix,
                         const Model* main_model,
                         std::vector<std::string> jpeg_data,
                         int32_t jpeg_output_channels, float scale,
                         int64_t zero_point, const Model* validation_model,
                         const reflection::Schema* schema,
                         bool use_ondevice_cpu_for_golden)
      : metric_prefix_(metric_prefix),
        main_model_(main_model),
        jpeg_data_(jpeg_data),
        jpeg_output_channels_(jpeg_output_channels),
        scale_(scale),
        zero_point_(zero_point),
        validation_model_(validation_model),
        schema_(schema),
        helper_(&fbb_, schema_),
        use_ondevice_cpu_for_golden_(use_ondevice_cpu_for_golden) {}

  ValidationGraphBuilder(const ValidationGraphBuilder&) = delete;
  ValidationGraphBuilder& operator=(const ValidationGraphBuilder&) = delete;

  // Builds the part of the model drawn above until the call to the validation
  // graph. The model is used to generate golden outputs. Calls Finish on the
  // FlatbufferBuilder.
  absl::Status BuildIntermediateModel(flatbuffers::FlatBufferBuilder* fbb);

  // Builds the whole model as drawn above. The subgraph_with_golden_outputs
  // should be the result of invoking subgraph 1 on the output of
  // BuildIntermediateModel(). Calls Finish on the FlatbufferBuilder.
  absl::Status BuildFinalModel(flatbuffers::FlatBufferBuilder* fbb,
                               Subgraph* subgraph_with_golden_outputs);

 private:
  // Allocation of tensors, for communication between methods that create the
  // tensors, the operations and the buffers.
  // (Some of these vectors will always contain only one element, but using the
  // same type for them simplifies the code a lot).
  struct TensorInfo {
    ~TensorInfo() { std::free(jpeg_buffer_contents); }

    std::vector<int32_t> entrypoint_inputs;
    std::vector<int32_t> entrypoint_outputs;
    std::vector<int32_t> jpeg_images;

    // With float main model, both quantized_images and float_images are set,
    // and float_images is the same as main input. With a quantized model
    // only quantized_images is set and it's the same as main input.
    std::vector<int32_t> quantized_images;
    std::vector<int32_t> float_images;

    std::vector<int32_t> main_outputs;  // First half of validation_inputs.
    std::vector<int32_t> validation_inputs;
    // With a float model, validation_inputs is used directly. With a quantized
    // model, the inputs are first dequantized.
    // Some models have a mixture of quantized outputs that need to be
    // dequantized to floats; and integer outputs. For integer outputs
    // kSkippedIndex is used.
    std::vector<int32_t> dequantized_validation_inputs;
    std::vector<int32_t> validation_outputs;

    char* jpeg_buffer_contents = nullptr;
    int32_t jpeg_buffer_length = -1;
    int32_t jpeg_height = -1;
    int32_t jpeg_width = -1;
  };

  static constexpr int32_t kModelVersion = 3;
  static constexpr int32_t kSkippedIndex = -1;
  // Operator code numbering.
  static constexpr int32_t kCallOperatorCode = 0;
  static constexpr int32_t kDequantizeOperatorCode = 1;
  static constexpr int32_t kDecodeJpegOperatorCode = 2;
  // Subgraph numbering.
  static constexpr int32_t kMainSubgraphIndex = 0;
  static constexpr int32_t kValidationSubgraphIndex = 2;

  absl::StatusOr<flatbuffers::Offset<Model>> MakeModel(
      bool intermediate_only, Subgraph* subgraph_with_golden_outputs);

  absl::StatusOr<flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<OperatorCode>>>>
  OperatorCodes();

  absl::StatusOr<
      flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Tensor>>>>
  Tensors(bool intermediate_only, TensorInfo* tensor_info);

  // Create the options for the custom call op (see call.cc for the options
  // format).
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> CallOpCustomOptions(
      int subgraph);

  // Create the options for the custom jpeg op (see decode_jpeg.cc for the
  // options format).
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> JpegOpCustomOptions(
      int height, int width, int channels);

  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Operator>>>
  Operators(bool intermediate_only, const TensorInfo& tensor_info);

  absl::StatusOr<
      flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<SubGraph>>>>
  SubGraphs(bool intermediate_only, TensorInfo* tensor_info);

  absl::StatusOr<
      flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>>
  Buffers(bool intermediate_only, const TensorInfo& tensor_info,
          Subgraph* subgraph_with_golden_outputs);

  const std::string metric_prefix_;
  const Model* main_model_;
  std::vector<std::string> jpeg_data_;
  int32_t jpeg_output_channels_;
  float scale_;
  int64_t zero_point_;
  const Model* validation_model_;
  const reflection::Schema* schema_;
  flatbuffers::FlatBufferBuilder fbb_;
  FlatbufferHelper helper_;
  bool use_ondevice_cpu_for_golden_;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_MODIFIER_VALIDATION_GRAPH_BUILDER_H_
