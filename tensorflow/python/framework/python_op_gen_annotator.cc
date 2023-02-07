/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/framework/python_op_gen_annotator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/python/framework/kythe_metadata.pb.h"
#include "tensorflow/python/framework/op_reg_offset.pb.h"

namespace tensorflow {
namespace python_op_gen_internal {

void GeneratedCodeAnnotator::AddAnnotation(const OpDef& op_def,
                                           absl::string_view function_name,
                                           uint32_t offset_start) {
  const uint32_t start_byte = base_pos_ + offset_start;
  const uint32_t end_byte = start_byte + function_name.size();
  byte_offsets_map_[op_def.name()].generated_start = start_byte;
  byte_offsets_map_[op_def.name()].generated_end = end_byte;
}

void GeneratedCodeAnnotator::FillSourceOffsets(
    const OpRegOffsets& op_reg_offsets) {
  for (const OpRegOffset& offset : op_reg_offsets.offsets()) {
    if (byte_offsets_map_.find(offset.name()) != byte_offsets_map_.end()) {
      byte_offsets_map_[offset.name()].file_path = offset.filepath();
      byte_offsets_map_[offset.name()].source_start = offset.start();
      byte_offsets_map_[offset.name()].source_end = offset.end();
    }
  }
}

string GeneratedCodeAnnotator::BuildKytheMetadata() {
  GeneratedCodeInfo generated_code_info;
  generated_code_info.set_type(GeneratedCodeInfo::KYTHE0);
  for (const auto& [name, offsets] : byte_offsets_map_) {
    if (offsets.file_path.empty()) {
      continue;
    }
    MappingRule* meta = generated_code_info.add_meta();
    meta->set_type(MappingRule::ANCHOR_ANCHOR);
    meta->set_edge("/kythe/edge/imputes");
    meta->set_source_begin(offsets.source_start);
    meta->set_source_end(offsets.source_end);
    meta->set_target_begin(offsets.generated_start);
    meta->set_target_end(offsets.generated_end);

    VName* vname = meta->mutable_source_vname();
    vname->set_signature(absl::StrFormat(
        "@%d:%d@tensorflow/op#%s#%s#%s", offsets.source_start,
        offsets.source_end, name, kKytheCorpus, offsets.file_path));
    vname->set_corpus(std::string(kKytheCorpus));
    vname->set_path(offsets.file_path);
    vname->set_language("c++");
  }
  return "# kythe.proto.metadata.GeneratedCodeInfo:" +
         absl::Base64Escape(generated_code_info.SerializeAsString());
}

}  // namespace python_op_gen_internal
}  // namespace tensorflow
