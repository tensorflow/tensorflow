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

#include "tensorflow/compiler/plugin/poplar/driver/single_hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/inplace_util.h"

namespace xla {
namespace poplarplugin {

unsigned SingleHloMatcher::ReplaceNodes() {
  unsigned int replacement_count = 0;
  for (int pattern = 0; pattern < matches_.size(); pattern++) {
    for (HloMatcherMatched& match : matches_[pattern]) {
      if (match.ok) {
        auto& fuse = fuse_info_[pattern];
        std::string name = op_prefix_ + fuse.name;
        const OutlinedInfo outlined_info =
            OutlineExpressionFromComputation(match, name, fuse.op_index);
        replacement_count += MarkReplacedInstructions(outlined_info);
      }
    }
  }
  return replacement_count;
}

}  // namespace poplarplugin
}  // namespace xla
