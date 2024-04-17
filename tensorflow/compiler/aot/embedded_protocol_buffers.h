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

#include "absl/types/span.h"
#include "xla/statusor.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace tfcompile {
using absl::StatusOr;

// Represents a set of protocol buffers embedded into an object file and
// describes how to access them at runtime.
struct EmbeddedProtocolBuffers {
  // Each instance CPPShim describes how to generate C++ code to instantiate a
  // protobuf instance from the corresponding static data emitted into the
  // object file.
  struct CPPShim {
    // `expression` is a C++ expression that creates an instance of said
    // protocol buffer when executed.
    string expression;

    // `variable_decl` is an "extern C" array declaration that is used in
    // `expression`.  It must be visible wherever `expression` is emitted.
    string variable_decl;
  };

  // Each cpp_shim corresponds to one embedded protocol buffer.
  std::vector<CPPShim> cpp_shims;

  // The contents of the object (".o") file the protocol buffers are embbed in.
  // This needs to be linked in to any program that wants to execute any of the
  // expressions in `cpp_shims`.
  string object_file_data;
};

// Describes a protocol buffer to embed into an object file.
struct ProtobufToEmbed {
  // `symbol_prefix` is prefix that is guaranteed to be unique across the binary
  // or DSO the generated object file will be linked into.
  string symbol_prefix;

  // `qualified_cpp_protobuf_name` is a qualified ("qualified" as in C++
  // namespace qualified) protocol buffer name.  This is only used in
  // CPPShim::expression so relatively qualified names are fine as long as
  // they're valid wherever CPPShim::expression is emitted.
  string qualified_cpp_protobuf_name;

  // `message` is the protocol buffer to be embedded.  It is allowed to be
  // nullptr, in which case the generated C++ shim expression is just `nullptr`,
  // and the generated object file does not define any symbols.
  const ::tensorflow::protobuf::MessageLite* message;
};

// Embeds a sequence of protocol buffers into an object file.
//
// `target_triple` is the target triple for the target architecture for the
// generated object file.
//
// `protobufs_to_embed` describes the protocol buffers to embed into the
// resulting object file.  The C++ shim for protobufs_to_embed[i] is
// cpp_shims[i] in the returned EmbeddedProtocolBuffers instance.  The contents
// of all the protocol buffers are embedded into a single .o file whose content
// is stored in the object_file_data field in the returned
// EmbeddedProtocolBuffers instance.
absl::StatusOr<EmbeddedProtocolBuffers> CreateEmbeddedProtocolBuffers(
    absl::string_view target_triple,
    absl::Span<const ProtobufToEmbed> protobufs_to_embed);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_EMBEDDED_PROTOCOL_BUFFERS_H_
