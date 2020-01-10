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
#ifndef TENSORFLOW_CORE_API_DEF_UPDATE_API_DEF_H_
#define TENSORFLOW_CORE_API_DEF_UPDATE_API_DEF_H_
// Functions for updating ApiDef when new ops are added.

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Returns ApiDefs text representation in multi-line format
// constructed based on the given op.
string CreateApiDef(const OpDef& op);

// Removes .Doc call for the given op.
// If unsuccessful, returns original file_contents and prints an error.
// start_location - We search for .Doc call starting at this location
//   in file_contents.
string RemoveDoc(const OpDef& op, const string& file_contents,
                 size_t start_location);

// Creates api_def_*.pbtxt files for any new ops (i.e. ops that don't have an
// api_def_*.pbtxt file yet).
// If op_file_pattern is non-empty, then this method will also
// look for a REGISTER_OP call for the new ops and removes corresponding
// .Doc() calls since the newly generated api_def_*.pbtxt files will
// store the doc strings.
void CreateApiDefs(const OpList& ops, const string& api_def_dir,
                   const string& op_file_pattern);

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_API_DEF_UPDATE_API_DEF_H_
