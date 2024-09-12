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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_TYPES_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_TYPES_H_

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"

namespace toco {

namespace tflite {

struct DataType {
  static ::tflite::TensorType Serialize(ArrayDataType array_data_type);
  static ArrayDataType Deserialize(int tensor_type);
};

struct DataBuffer {
  using FlatBufferOffset = flatbuffers::Offset<flatbuffers::Vector<uint8_t>>;

  // Build the flatbuffer representation of a toco's Array and return the
  // corresponding offset into the flatbuffer. Note that data from the array
  // will be copied into the flatbuffer.
  static FlatBufferOffset Serialize(const Array& array,
                                    flatbuffers::FlatBufferBuilder* builder);
  // Copy data from the given tensor into toco's Array.
  static void Deserialize(const ::tflite::Tensor& tensor,
                          const ::tflite::Buffer& buffer, Array* array);
};

struct Padding {
  static ::tflite::Padding Serialize(PaddingType padding_type);
  static PaddingType Deserialize(int padding);
};

struct ActivationFunction {
  static ::tflite::ActivationFunctionType Serialize(
      FusedActivationFunctionType faf_type);
  static FusedActivationFunctionType Deserialize(int activation_function);
};

}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_TYPES_H_
