/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Dumps a TFLite flatbuffer to a textual output format.
// This tool is intended to be used to simplify unit testing/debugging.

#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <iostream>
#include <string>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/minireflect.h"  // from @flatbuffers
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#if FLATBUFFERS_LITTLEENDIAN == 0
#include "tensorflow/lite/core/model_builder.h"
#endif

namespace tflite {
namespace {

// Reads a model from a provided file path and verifies if it is a valid
// flatbuffer, and returns false with the model in serialized_model if valid
// else true.
bool ReadAndVerify(const std::string& file_path,
                   std::string* serialized_model) {
  if (file_path == "-") {
    *serialized_model = std::string{std::istreambuf_iterator<char>(std::cin),
                                    std::istreambuf_iterator<char>()};
  } else {
    std::ifstream t(file_path);
    if (!t.is_open()) {
      std::cerr << "Failed to open input file.\n";
      return true;
    }
    *serialized_model = std::string{std::istreambuf_iterator<char>(t),
                                    std::istreambuf_iterator<char>()};
  }

  flatbuffers::Verifier model_verifier(
      reinterpret_cast<const uint8_t*>(serialized_model->c_str()),
      serialized_model->length());
  if (!model_verifier.VerifyBuffer<Model>()) {
    std::cerr << "Verification failed.\n";
    return true;
  }
  return false;
}

// A FlatBuffer visitor that outputs a FlatBuffer as a string with proper
// indention for sequence fields.
// TODO(wvo): ToStringVisitor already has indentation functionality, use
// that directly instead of this sub-class?
struct IndentedToStringVisitor : flatbuffers::ToStringVisitor {
  std::string indent_str;
  int indent_level;

  IndentedToStringVisitor(const std::string& delimiter,
                          const std::string& indent)
      : ToStringVisitor(delimiter), indent_str(indent), indent_level(0) {}

  void indent() {
    for (int i = 0; i < indent_level; ++i) s.append(indent_str);
  }

  // Adjust indention for fields in sequences.

  void StartSequence() override {
    s += "{";
    s += d;
    ++indent_level;
  }

  void EndSequence() override {
    s += d;
    --indent_level;
    indent();
    s += "}";
  }

  void Field(size_t /*field_idx*/, size_t set_idx,
             flatbuffers::ElementaryType /*type*/, bool /*is_vector*/,
             const flatbuffers::TypeTable* /*type_table*/, const char* name,
             const uint8_t* val) override {
    if (!val) return;
    if (set_idx) {
      s += ",";
      s += d;
    }
    indent();
    if (name) {
      s += name;
      s += ": ";
    }
  }

  void StartVector() override { s += "[ "; }
  void EndVector() override { s += " ]"; }

  void Element(size_t i, flatbuffers::ElementaryType /*type*/,
               const flatbuffers::TypeTable* /*type_table*/,
               const uint8_t* /*val*/) override {
    if (i) s += ", ";
  }
};

void ToString(const std::string& serialized_model) {
  IndentedToStringVisitor visitor(/*delimiter=*/"\n", /*indent=*/"  ");
  IterateFlatBuffer(reinterpret_cast<const uint8_t*>(serialized_model.c_str()),
                    ModelTypeTable(), &visitor);
  std::cout << visitor.s << "\n\n";
}

}  // end namespace
}  // end namespace tflite

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Missing input argument. Usage:\n"
              << argv[0] << " <filename or - for stdin>\n\n"
              << "Converts TensorFlowLite flatbuffer to textual output format. "
              << "One positional input argument representing the source of the "
              << "flatbuffer is supported.\n";
    return 1;
  }

  std::string serialized_model;
  if (tflite::ReadAndVerify(argv[1], &serialized_model)) return 1;
#if FLATBUFFERS_LITTLEENDIAN == 0
  tflite::FlatBufferModel::ByteSwapSerializedModel(&serialized_model);
#endif
  tflite::ToString(serialized_model);
  return 0;
}
