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
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_overrides.pb.h"
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

// Split `*orig` into two pieces at the first occurrence of `split_ch`.
// Returns whether `split_ch` was found. Afterwards, `*before_split`
// contains the maximum prefix of the input `*orig` that doesn't
// contain `split_ch`, and `*orig` contains everything after the
// first `split_ch`.
static bool SplitAt(char split_ch, StringPiece* orig,
                    StringPiece* before_split) {
  auto pos = orig->find(split_ch);
  if (pos == StringPiece::npos) {
    *before_split = *orig;
    orig->clear();
    return false;
  } else {
    *before_split = orig->substr(0, pos);
    orig->remove_prefix(pos + 1);
    return true;
  }
}

// Does this line start with "<spaces><field>:" where "<field>" is
// in multi_line_fields? Sets *colon_pos to the position of the colon.
static bool StartsWithFieldName(StringPiece line,
                                const std::vector<string>& multi_line_fields) {
  StringPiece up_to_colon;
  if (!SplitAt(':', &line, &up_to_colon)) return false;
  while (up_to_colon.Consume(" "))
    ;  // Remove leading spaces.
  for (const auto& field : multi_line_fields) {
    if (up_to_colon == field) {
      return true;
    }
  }
  return false;
}

static bool ConvertLine(StringPiece line,
                        const std::vector<string>& multi_line_fields,
                        string* ml) {
  // Is this a field we should convert?
  if (!StartsWithFieldName(line, multi_line_fields)) {
    return false;
  }
  // Has a matching field name, so look for "..." after the colon.
  StringPiece up_to_colon;
  StringPiece after_colon = line;
  SplitAt(':', &after_colon, &up_to_colon);
  while (after_colon.Consume(" "))
    ;  // Remove leading spaces.
  if (!after_colon.Consume("\"")) {
    // We only convert string fields, so don't convert this line.
    return false;
  }
  auto last_quote = after_colon.rfind('\"');
  if (last_quote == StringPiece::npos) {
    // Error: we don't see the expected matching quote, abort the conversion.
    return false;
  }
  StringPiece escaped = after_colon.substr(0, last_quote);
  StringPiece suffix = after_colon.substr(last_quote + 1);
  // We've now parsed line into '<up_to_colon>: "<escaped>"<suffix>'

  string unescaped;
  if (!str_util::CUnescape(escaped, &unescaped, nullptr)) {
    // Error unescaping, abort the conversion.
    return false;
  }
  // No more errors possible at this point.

  // Find a string to mark the end that isn't in unescaped.
  string end = "END";
  for (int s = 0; unescaped.find(end) != string::npos; ++s) {
    end = strings::StrCat("END", s);
  }

  // Actually start writing the converted output.
  strings::StrAppend(ml, up_to_colon, ": <<", end, "\n", unescaped, "\n", end);
  if (!suffix.empty()) {
    // Output suffix, in case there was a trailing comment in the source.
    strings::StrAppend(ml, suffix);
  }
  strings::StrAppend(ml, "\n");
  return true;
}

string PBTxtToMultiline(StringPiece pbtxt,
                        const std::vector<string>& multi_line_fields) {
  string ml;
  // Probably big enough, since the input and output are about the
  // same size, but just a guess.
  ml.reserve(pbtxt.size() * (17. / 16));
  StringPiece line;
  while (!pbtxt.empty()) {
    // Split pbtxt into its first line and everything after.
    SplitAt('\n', &pbtxt, &line);
    // Convert line or output it unchanged
    if (!ConvertLine(line, multi_line_fields, &ml)) {
      strings::StrAppend(&ml, line, "\n");
    }
  }
  return ml;
}

// Given a single line of text `line` with first : at `colon`, determine if
// there is an "<<END" expression after the colon and if so return true and set
// `*end` to everything after the "<<".
static bool FindMultiline(StringPiece line, size_t colon, string* end) {
  if (colon == StringPiece::npos) return false;
  line.remove_prefix(colon + 1);
  while (line.Consume(" ")) {
  }
  if (line.Consume("<<")) {
    *end = line.ToString();
    return true;
  }
  return false;
}

string PBTxtFromMultiline(StringPiece multiline_pbtxt) {
  string pbtxt;
  // Probably big enough, since the input and output are about the
  // same size, but just a guess.
  pbtxt.reserve(multiline_pbtxt.size() * (33. / 32));
  StringPiece line;
  while (!multiline_pbtxt.empty()) {
    // Split multiline_pbtxt into its first line and everything after.
    if (!SplitAt('\n', &multiline_pbtxt, &line)) {
      strings::StrAppend(&pbtxt, line);
      break;
    }

    string end;
    auto colon = line.find(':');
    if (!FindMultiline(line, colon, &end)) {
      // Normal case: not a multi-line string, just output the line as-is.
      strings::StrAppend(&pbtxt, line, "\n");
      continue;
    }

    // Multi-line case:
    //     something: <<END
    // xx
    // yy
    // END
    // Should be converted to:
    //     something: "xx\nyy"

    // Output everything up to the colon ("    something:").
    strings::StrAppend(&pbtxt, line.substr(0, colon + 1));

    // Add every line to unescaped until we see the "END" string.
    string unescaped;
    bool first = true;
    string suffix;
    while (!multiline_pbtxt.empty()) {
      SplitAt('\n', &multiline_pbtxt, &line);
      if (line.Consume(end)) break;
      if (first) {
        first = false;
      } else {
        unescaped.push_back('\n');
      }
      strings::StrAppend(&unescaped, line);
      line.clear();
    }

    // Escape what we extracted and then output it in quotes.
    strings::StrAppend(&pbtxt, " \"", str_util::CEscape(unescaped), "\"", line,
                       "\n");
  }
  return pbtxt;
}

OpGenOverrideMap::OpGenOverrideMap() {}
OpGenOverrideMap::~OpGenOverrideMap() {}

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
    map_[one.name()].reset(new OpGenOverride(one));
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
  const OpGenOverride& proto = *iter->second;

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
