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

#include "tensorflow/core/framework/op_gen_lib.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

string WordWrap(StringPiece prefix, StringPiece str, int width) {
  const string indent_next_line = "\n" + Spaces(prefix.size());
  width -= prefix.size();
  string result;
  strings::StrAppend(&result, prefix);

  while (!str.empty()) {
    if (static_cast<int>(str.size()) <= width) {
      // Remaining text fits on one line.
      strings::StrAppend(&result, str);
      break;
    }
    auto space = str.rfind(' ', width);
    if (space == StringPiece::npos) {
      // Rather make a too-long line and break at a space.
      space = str.find(' ');
      if (space == StringPiece::npos) {
        strings::StrAppend(&result, str);
        break;
      }
    }
    // Breaking at character at position <space>.
    StringPiece to_append = str.substr(0, space);
    str.remove_prefix(space + 1);
    // Remove spaces at break.
    while (to_append.ends_with(" ")) {
      to_append.remove_suffix(1);
    }
    while (str.Consume(" ")) {
    }

    // Go on to the next line.
    strings::StrAppend(&result, to_append);
    if (!str.empty()) strings::StrAppend(&result, indent_next_line);
  }

  return result;
}

bool ConsumeEquals(StringPiece* description) {
  if (description->Consume("=")) {
    while (description->Consume(" ")) {  // Also remove spaces after "=".
    }
    return true;
  }
  return false;
}

Status OpGenOverrideMap::LoadFileList(Env* env, const string& filenames) {
  std::vector<string> v = str_util::Split(filenames, ",");
  for (const string& f : v) {
    TF_RETURN_IF_ERROR(LoadFile(env, f));
  }
  return Status::OK();
}

Status OpGenOverrideMap::LoadFile(Env* env, const string& filename) {
  if (filename.empty()) return Status::OK();
  string contents;
  TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &contents));
  OpGenOverrides all;
  protobuf::TextFormat::ParseFromString(contents, &all);
  for (const auto& one : all.op()) {
    map_[one.name()] = one;
  }
  return Status::OK();
}

static void StringReplace(const string& from, const string& to, string* s) {
  // Split *s into pieces delimited by `from`.
  std::vector<string> split;
  string::size_type pos = 0;
  while (pos < s->size()) {
    auto found = s->find(from, pos);
    if (found == string::npos) {
      split.push_back(s->substr(pos));
      break;
    } else {
      split.push_back(s->substr(pos, found - pos));
      pos = found + from.size();
    }
  }
  // Join the pieces back together with a new delimiter.
  *s = str_util::Join(split, to.c_str());
}

static void RenameInDocs(const string& from, const string& to, OpDef* op_def) {
  const string from_quoted = strings::StrCat("`", from, "`");
  const string to_quoted = strings::StrCat("`", to, "`");
  for (int i = 0; i < op_def->input_arg_size(); ++i) {
    if (!op_def->input_arg(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    op_def->mutable_input_arg(i)->mutable_description());
    }
  }
  for (int i = 0; i < op_def->output_arg_size(); ++i) {
    if (!op_def->output_arg(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    op_def->mutable_output_arg(i)->mutable_description());
    }
  }
  for (int i = 0; i < op_def->attr_size(); ++i) {
    if (!op_def->attr(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    op_def->mutable_attr(i)->mutable_description());
    }
  }
  if (!op_def->summary().empty()) {
    StringReplace(from_quoted, to_quoted, op_def->mutable_summary());
  }
  if (!op_def->description().empty()) {
    StringReplace(from_quoted, to_quoted, op_def->mutable_description());
  }
}

const OpGenOverride* OpGenOverrideMap::ApplyOverride(OpDef* op_def) const {
  // Look up
  const auto iter = map_.find(op_def->name());
  if (iter == map_.end()) return nullptr;
  const OpGenOverride& proto = iter->second;

  // Apply overrides from `proto`.
  if (!proto.rename_to().empty()) {
    op_def->set_name(proto.rename_to());
    RenameInDocs(proto.name(), proto.rename_to(), op_def);
  }
  for (const auto& attr_default : proto.attr_default()) {
    bool found = false;
    for (int i = 0; i < op_def->attr_size(); ++i) {
      if (op_def->attr(i).name() == attr_default.name()) {
        *op_def->mutable_attr(i)->mutable_default_value() =
            attr_default.value();
        found = true;
        break;
      }
    }
    if (!found) {
      LOG(WARNING) << proto.name() << " can't find attr " << attr_default.name()
                   << " to override default";
    }
  }
  for (const auto& attr_rename : proto.attr_rename()) {
    bool found = false;
    for (int i = 0; i < op_def->attr_size(); ++i) {
      if (op_def->attr(i).name() == attr_rename.from()) {
        *op_def->mutable_attr(i)->mutable_name() = attr_rename.to();
        found = true;
        break;
      }
    }
    if (found) {
      RenameInDocs(attr_rename.from(), attr_rename.to(), op_def);
    } else {
      LOG(WARNING) << proto.name() << " can't find attr " << attr_rename.from()
                   << " to rename";
    }
  }
  for (const auto& input_rename : proto.input_rename()) {
    bool found = false;
    for (int i = 0; i < op_def->input_arg_size(); ++i) {
      if (op_def->input_arg(i).name() == input_rename.from()) {
        *op_def->mutable_input_arg(i)->mutable_name() = input_rename.to();
        found = true;
        break;
      }
    }
    if (found) {
      RenameInDocs(input_rename.from(), input_rename.to(), op_def);
    } else {
      LOG(WARNING) << proto.name() << " can't find input "
                   << input_rename.from() << " to rename";
    }
  }
  for (const auto& output_rename : proto.output_rename()) {
    bool found = false;
    for (int i = 0; i < op_def->output_arg_size(); ++i) {
      if (op_def->output_arg(i).name() == output_rename.from()) {
        *op_def->mutable_output_arg(i)->mutable_name() = output_rename.to();
        found = true;
        break;
      }
    }
    if (found) {
      RenameInDocs(output_rename.from(), output_rename.to(), op_def);
    } else {
      LOG(WARNING) << proto.name() << " can't find output "
                   << output_rename.from() << " to rename";
    }
  }

  return &proto;
}

}  // namespace tensorflow
