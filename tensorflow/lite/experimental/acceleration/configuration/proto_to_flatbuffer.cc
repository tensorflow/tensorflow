/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"

#include <string>

#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/util.h"  // from @flatbuffers
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {

namespace {
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_fbs_contents-inl.h"
}

const ComputeSettings* ConvertFromProto(
    flatbuffers::Parser* parser, const proto::ComputeSettings& proto_settings) {
  std::string json;
  tensorflow::protobuf::util::JsonPrintOptions options;
  options.preserve_proto_field_names = true;
  options.always_print_primitive_fields = true;  // For catching problems.
  auto status = tensorflow::protobuf::util::MessageToJsonString(proto_settings,
                                                                &json, options);
  if (!status.ok()) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to convert to Json: %s",
                    status.ToString().c_str());
    return nullptr;
  }
  if (!parser->Parse(configuration_fbs_contents)) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to parse schema: %s",
                    parser->error_.c_str());
    return nullptr;
  }
  parser->SetRootType("tflite.ComputeSettings");
  if (!parser->Parse(json.c_str())) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to parse json: %s",
                    parser->error_.c_str());
    return nullptr;
  }
  return flatbuffers::GetRoot<ComputeSettings>(
      parser->builder_.GetBufferPointer());
}

}  // namespace tflite
