/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_AOT_EMBEDDED_CONSTANT_BUFFERS_H_
#define TENSORFLOW_COMPILER_AOT_EMBEDDED_CONSTANT_BUFFERS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tensorflow {
namespace tfcompile {

// Represents a set of constant buffers embedded into an object file.
struct EmbeddedConstantBuffers {
  struct VariableInfo {
    // variable_name is the name of the variable from variable_decl.
    std::string variable_name;

    // `variable_decl` is an "extern C" array declaration that is used in
    // `expression`.
    std::string variable_decl;

    // `cpp_access_shim` is a C++ expression that receives a pointer to the
    // start of the buffer with size and returns the size and a pointer
    // to the start of the buffer data.
    std::string cpp_access_shim;
  };
  // Variable infos for each constant buffer.
  std::vector<VariableInfo> variable_decls;

  // The contents of the object (".o") file the constant buffers are embedded
  // in.
  std::string object_file_data;
};

// Describes a protocol buffer to embed into an object file.
struct ConstantToEmbed {
  // `symbol_prefix` is prefix that is guaranteed to be unique across the binary
  // or DSO the generated object file will be linked into.
  std::string symbol_prefix;

  // Serializes the size of the `buffer` and it's contents into `data`.
  void SerializeIntoBuffer(absl::Span<const uint8_t> buffer);

  const std::vector<uint8_t>& data() const { return data_buffer; }

 private:
  // `data_buffer` is the constant buffer to be embedded. It containes the
  // number of bytes of the buffer and it's contents.
  std::vector<uint8_t> data_buffer;
};

absl::StatusOr<EmbeddedConstantBuffers> CreateEmbeddedConstantBuffers(
    absl::string_view target_triple,
    absl::Span<ConstantToEmbed> constants_to_embed);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_EMBEDDED_CONSTANT_BUFFERS_H_
