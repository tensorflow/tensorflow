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

namespace tensorflow {
namespace tfcompile {

// HeaderOpts specifies options for header-file generation.
struct HeaderOpts {
  // The name of the generated C++ class, wrapping the generated function.
  string class_name;

  // Namespaces specifies a list of C++ namespaces to add to the generated
  // header.  If empty, all symbols will be in the global namespace.
  std::vector<string> namespaces;
};

// GenerateHeader uses the meta-information from compile_result to generate a
// C++ header giving access to the function in the generated object file.  The
// header includes API usage documentation.
Status GenerateHeader(const HeaderOpts& opts, const Config& config,
                      const CompileResult& compile_result, string* header);

// ParseCppClass parses `cpp_class` into its `class_name` and `namespaces`
// components.  The syntax is [[<optional_namespace>::],...]<class_name>.  This
// mirrors the C++ syntax for referring to a class, where multiple namespaces
// may precede the class name, separated by double-colons.
Status ParseCppClass(const string& cpp_class, string* class_name,
                     std::vector<string>* namespaces);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_CODEGEN_H_
