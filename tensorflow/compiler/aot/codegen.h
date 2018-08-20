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

#ifndef TENSORFLOW_COMPILER_AOT_CODEGEN_H_
#define TENSORFLOW_COMPILER_AOT_CODEGEN_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace tfcompile {

// CodegenOpts specifies code generation options for the generated header file
// and the generated metadata object file.
struct CodegenOpts {
  // The name of the generated C++ class, wrapping the generated function.
  string class_name;

  // Target triple for the architecture we're targeting.
  string target_triple;

  // Namespaces specifies a list of C++ namespaces to add to the generated
  // header.  If empty, all symbols will be in the global namespace.
  std::vector<string> namespaces;

  // If true, generate name-to-index data for Lookup{Arg,Result}Index methods.
  bool gen_name_to_index = false;

  // If true, generate program shape data for the ProgramShape method.
  bool gen_program_shape = false;

  // If true, emit a serialized HloProfilePrinterData protobuf that can be used
  // to pretty print HLO profile counters.
  bool gen_hlo_profile_printer_data = false;
};

// Describes a generated metadata object file.
struct MetadataResult {
  // These are top level "extern C" declarations that are expected to be visible
  // wherever program_shape_access_shim is emitted.
  std::vector<string> header_variable_decls;

  // program_shape_access_shim is a C++ expression that constructs the
  // xla::ProgramShape instance for the CompileResult passed to
  // GenerateMetadata.
  string program_shape_access_shim;

  // hlo_profile_printer_data_access_shim is a C++ expression that constructs
  // the xla::HloProfilePrinterData instance for the CompileResult passed to
  // GenerateMetadata.  If the xla::HloProfilePrinterData is null then this is a
  // C++ expression that evaluates to nullptr at runtime.
  string hlo_profile_printer_data_access_shim;

  // The contents of the object (".o") file.
  string object_file_data;
};

// Generates a metadata object file according to `opts` and `compile_result`.
// The generated object file is returned via `metadata_result`.
Status GenerateMetadata(const CodegenOpts& opts,
                        const CompileResult& compile_result,
                        MetadataResult* metadata_result);

// GenerateHeader uses the meta-information from compile_result to generate a
// C++ header giving access to the function in the generated object file.  The
// header includes API usage documentation.
//
// metadata_result is an instance of MetadataResult obtained by a previous
// invocation to GenerateMetadata.
Status GenerateHeader(const CodegenOpts& opts, const tf2xla::Config& config,
                      const CompileResult& compile_result,
                      const MetadataResult& metadata_result, string* header);

// ParseCppClass parses `cpp_class` into its `class_name` and `namespaces`
// components.  The syntax is [[<optional_namespace>::],...]<class_name>.  This
// mirrors the C++ syntax for referring to a class, where multiple namespaces
// may precede the class name, separated by double-colons.
Status ParseCppClass(const string& cpp_class, string* class_name,
                     std::vector<string>* namespaces);

// ValidateCppIdent returns OK iff ident is a valid C++ identifier.  The msg is
// appended to error messages.
Status ValidateCppIdent(StringPiece ident, StringPiece msg);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_CODEGEN_H_
