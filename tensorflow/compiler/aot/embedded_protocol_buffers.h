/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// This file defines utilities to help "embed" protocol buffers into object
// (".o") files.  These C++ binaries and shared objects can link in these .o to
// get access to said protocol buffers at runtime.

#ifndef TENSORFLOW_COMPILER_AOT_EMBEDDED_PROTOCOL_BUFFERS_H_
#define TENSORFLOW_COMPILER_AOT_EMBEDDED_PROTOCOL_BUFFERS_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace tfcompile {
using xla::StatusOr;

// Represents a protocol buffer embedded into an object file and describes a way
// to access it at runtime.
struct EmbeddedProtocolBuffer {
  // cpp_shim_expression is a C++ expression that creates an instance of said
  // protocol buffer when executed.
  string cpp_shim_expression;

  // cpp_variable_decl is an "extern C" array declaration that is used in
  // cpp_shim_expression.  It must be visible wherever cpp_shim_expression is
  // emitted.
  string cpp_variable_decl;

  // The contents of the object (".o") file the protocol buffer is embbed in.
  // This needs to be linked in to any program that wants to execute
  // cpp_variable_decl .
  string object_file_data;
};

// Creates an object file that contains `proto`.
//
// `proto` is allowed to be nullptr, in which case the generated C++ shim
// expression is just `nullptr`, and the generated object file does not define
// any symbols.
//
// `target_triple` is the target triple for the target architecture for the
// generated object file.
//
// `symbol_prefix` is prefix that is guaranteed to be unique across the binary
// or DSO the generated object file will be linked into.
//
// `qualified_cpp_protobuf_name` is a qualified ("qualified" as in C++
// namespace qualified) protocol buffer name.  This needs is only used in
// EmbeddedProtocolBuffer::cpp_shim_expression so relatively qualified
// names are fine as long as they're valid wherever cpp_shim_expression
// is emitted.
StatusOr<EmbeddedProtocolBuffer> CreateEmbeddedProtocolBuffer(
    StringPiece target_triple, StringPiece symbol_prefix,
    StringPiece qualified_cpp_protobuf_name,
    const ::tensorflow::protobuf::MessageLite* proto);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_EMBEDDED_PROTOCOL_BUFFERS_H_
