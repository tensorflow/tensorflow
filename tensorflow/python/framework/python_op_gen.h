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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Returns a string containing the generated Python code for the given Ops.
// ops is a protobuff, typically generated using OpRegistry::Global()->Export.
// api_defs is typically constructed directly from ops.
// hidden_ops should be a list of Op names that should get a leading _
// in the output.
// source_file_name is optional and contains the name of the original C++ source
// file where the ops' REGISTER_OP() calls reside.
string GetPythonOps(const OpList& ops, const ApiDefMap& api_defs,
                    const std::vector<string>& hidden_ops,
                    const string& source_file_name,
                    const std::unordered_set<string> type_annotate_ops);

// Prints the output of GetPrintOps to stdout.
// hidden_ops should be a list of Op names that should get a leading _
// in the output.
// Optional fourth argument is the name of the original C++ source file
// where the ops' REGISTER_OP() calls reside.
void PrintPythonOps(const OpList& ops, const ApiDefMap& api_defs,
                    const std::vector<string>& hidden_ops,
                    const string& source_file_name,
                    const std::unordered_set<string> type_annotate_ops);

// Get the python wrappers for a list of ops in a OpList.
// `op_list_buf` should be a pointer to a buffer containing
// the binary encoded OpList proto, and `op_list_len` should be the
// length of that buffer.
string GetPythonWrappers(const char* op_list_buf, size_t op_list_len);

// Get the type annotation for an arg
// `arg` should be an input or output of an op
// `type_annotations` should contain attr names mapped to TypeVar names
string GetArgAnnotation(
    const OpDef::ArgDef& arg,
    const std::unordered_map<string, string>& type_annotations);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_
