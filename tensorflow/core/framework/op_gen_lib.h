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

#ifndef TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_
#define TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_

#include <string>
#include <unordered_map>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// Forward declare protos so their symbols can be removed from .so exports
class OpDef;
class OpGenOverride;

inline string Spaces(int n) { return string(n, ' '); }

// Wrap prefix + str to be at most width characters, indenting every line
// after the first by prefix.size() spaces.  Intended use case is something
// like prefix = "  Foo(" and str is a list of arguments (terminated by a ")").
// TODO(josh11b): Option to wrap on ", " instead of " " when possible.
string WordWrap(StringPiece prefix, StringPiece str, int width);

// Looks for an "=" at the beginning of *description.  If found, strips it off
// (and any following spaces) from *description and return true.  Otherwise
// returns false.
bool ConsumeEquals(StringPiece* description);

// Convert text-serialized protobufs to/from multiline format.
string PBTxtToMultiline(StringPiece pbtxt,
                        const std::vector<string>& multi_line_fields);
string PBTxtFromMultiline(StringPiece multiline_pbtxt);

// Takes a list of files with OpGenOverrides text protos, and allows you to
// look up the specific override for any given op.
class OpGenOverrideMap {
 public:
  OpGenOverrideMap();
  ~OpGenOverrideMap();

  // `filenames` is a comma-separated list of file names.  If an op
  // is mentioned in more than one file, the last one takes priority.
  Status LoadFileList(Env* env, const string& filenames);

  // Load a single file.  If more than one file is loaded, later ones
  // take priority for any ops in common.
  Status LoadFile(Env* env, const string& filename);

  // Look up the override for `*op_def` from the loaded files, and
  // mutate `*op_def` to reflect the requested changes. Does not apply
  // 'skip', 'hide', or 'alias' overrides. Caller has to deal with
  // those since they can't be simulated by mutating `*op_def`.
  // Returns nullptr if op is not in any loaded file. Otherwise, the
  // pointer must not be referenced beyond the lifetime of *this or
  // the next file load.
  const OpGenOverride* ApplyOverride(OpDef* op_def) const;

 private:
  std::unordered_map<string, std::unique_ptr<OpGenOverride>> map_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_
