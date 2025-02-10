/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_GEN_LIB_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_GEN_LIB_H_

#include <string>
#include <unordered_map>
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// Forward declare protos so their symbols can be removed from .so exports
class OpDef;

inline string Spaces(int n) { return string(n, ' '); }

// Wrap prefix + str to be at most width characters, indenting every line
// after the first by prefix.size() spaces.  Intended use case is something
// like prefix = "  Foo(" and str is a list of arguments (terminated by a ")").
// TODO(josh11b): Option to wrap on ", " instead of " " when possible.
string WordWrap(absl::string_view prefix, absl::string_view str, int width);

// Looks for an "=" at the beginning of *description.  If found, strips it off
// (and any following spaces) from *description and return true.  Otherwise
// returns false.
bool ConsumeEquals(absl::string_view* description);

// Convert text-serialized protobufs to/from multiline format.
string PBTxtToMultiline(absl::string_view pbtxt,
                        const std::vector<string>& multi_line_fields);
string PBTxtFromMultiline(absl::string_view multiline_pbtxt);

// Takes a list of files with ApiDefs text protos, and allows you to
// look up the specific ApiDef for any given op.
class ApiDefMap {
 public:
  // OpList must be a superset of ops of any subsequently loaded
  // ApiDef.
  explicit ApiDefMap(const OpList& op_list);
  ~ApiDefMap();

  // You can call this method multiple times to load multiple
  // sets of files. Api definitions are merged if the same
  // op definition is loaded multiple times. Later-loaded
  // definitions take precedence.
  // ApiDefs loaded from files must contain a subset of ops defined
  // in the OpList passed to the constructor.
  absl::Status LoadFileList(Env* env, const std::vector<string>& filenames);

  // Load a single file. Api definitions are merged if the same
  // op definition is loaded multiple times. Later-loaded
  // definitions take precedence.
  // ApiDefs loaded from file must contain a subset of ops defined
  // in the OpList passed to the constructor.
  absl::Status LoadFile(Env* env, const string& filename);

  // Load ApiDefs from string containing ApiDefs text proto.
  // api_def_file_contents is expected to be in "multiline format".
  // ApiDefs must contain a subset of ops defined in OpsList
  // passed to the constructor.
  absl::Status LoadApiDef(const string& api_def_file_contents);

  // Updates ApiDef docs. For example, if ApiDef renames an argument
  // or attribute, applies these renames to descriptions as well.
  // UpdateDocs should only be called once after all ApiDefs are loaded
  // since it replaces original op names.
  void UpdateDocs();

  // Look up ApiDef proto based on the given graph op name.
  // If graph op name is not in this ApiDefMap, returns nullptr.
  //
  // Note: Returned ApiDef pointer should stay valid even after calling
  // Load* functions defined above. Subsequent calls to Load* might modify
  // returned ApiDef contents, but should never remove the ApiDef itself.
  const ApiDef* GetApiDef(const string& name) const;

 private:
  std::unordered_map<string, ApiDef> map_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_GEN_LIB_H_
