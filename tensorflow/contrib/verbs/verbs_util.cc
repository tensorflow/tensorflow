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

#include "tensorflow/contrib/verbs/verbs_util.h"

#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/str_util.h"
namespace tensorflow {

// static
string VerbsUtil::AppendStepidToKey(const string& key, int64 step_id) {
  return strings::StrCat(key, ";", step_id);
}

// static
void VerbsUtil::GetKeyAndStepId(const string& key_with_step_id, string& key,
                                int64& step_id) {
  StringPiece s(key_with_step_id);
  // a key (with step_id) has exact 6 parts if split by ";"
  // part 1: src_device;
  // part 2: src_incarnation;
  // part 3: dst_device;
  // part 4: name;
  // part 5: frame_iter.frame_id:frame_iter.iter_id
  // part 6: step_id
  std::vector<string> parts = str_util::Split(s, ';');
  CHECK(parts.size() == 6) << "Key with step_id must have 6 parts";
  strings::safe_strto64(parts[5], &step_id);
  parts.pop_back();                        // remove step_id
  key.assign(str_util::Join(parts, ";"));  // stitch them together
}

}  // namespace tensorflow
