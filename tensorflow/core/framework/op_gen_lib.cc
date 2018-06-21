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

#include <algorithm>
#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

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
    while (str_util::EndsWith(to_append, " ")) {
      to_append.remove_suffix(1);
    }
    while (str_util::ConsumePrefix(&str, " ")) {
    }

    // Go on to the next line.
    strings::StrAppend(&result, to_append);
    if (!str.empty()) strings::StrAppend(&result, indent_next_line);
  }

  return result;
}

bool ConsumeEquals(StringPiece* description) {
  if (str_util::ConsumePrefix(description, "=")) {
    while (str_util::ConsumePrefix(description,
                                   " ")) {  // Also remove spaces after "=".
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
    *orig = StringPiece();
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
  while (str_util::ConsumePrefix(&up_to_colon, " "))
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
  while (str_util::ConsumePrefix(&after_colon, " "))
    ;  // Remove leading spaces.
  if (!str_util::ConsumePrefix(&after_colon, "\"")) {
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
  while (str_util::ConsumePrefix(&line, " ")) {
  }
  if (str_util::ConsumePrefix(&line, "<<")) {
    *end = std::string(line);
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
      if (str_util::ConsumePrefix(&line, end)) break;
      if (first) {
        first = false;
      } else {
        unescaped.push_back('\n');
      }
      strings::StrAppend(&unescaped, line);
      line = StringPiece();
    }

    // Escape what we extracted and then output it in quotes.
    strings::StrAppend(&pbtxt, " \"", str_util::CEscape(unescaped), "\"", line,
                       "\n");
  }
  return pbtxt;
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
      if (pos == s->size()) {  // handle case where `from` is at the very end.
        split.push_back("");
      }
    }
  }
  // Join the pieces back together with a new delimiter.
  *s = str_util::Join(split, to.c_str());
}

static void RenameInDocs(const string& from, const string& to,
                         ApiDef* api_def) {
  const string from_quoted = strings::StrCat("`", from, "`");
  const string to_quoted = strings::StrCat("`", to, "`");
  for (int i = 0; i < api_def->in_arg_size(); ++i) {
    if (!api_def->in_arg(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    api_def->mutable_in_arg(i)->mutable_description());
    }
  }
  for (int i = 0; i < api_def->out_arg_size(); ++i) {
    if (!api_def->out_arg(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    api_def->mutable_out_arg(i)->mutable_description());
    }
  }
  for (int i = 0; i < api_def->attr_size(); ++i) {
    if (!api_def->attr(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    api_def->mutable_attr(i)->mutable_description());
    }
  }
  if (!api_def->summary().empty()) {
    StringReplace(from_quoted, to_quoted, api_def->mutable_summary());
  }
  if (!api_def->description().empty()) {
    StringReplace(from_quoted, to_quoted, api_def->mutable_description());
  }
}

namespace {

// Initializes given ApiDef with data in OpDef.
void InitApiDefFromOpDef(const OpDef& op_def, ApiDef* api_def) {
  api_def->set_graph_op_name(op_def.name());
  api_def->set_visibility(ApiDef::VISIBLE);

  auto* endpoint = api_def->add_endpoint();
  endpoint->set_name(op_def.name());

  for (const auto& op_in_arg : op_def.input_arg()) {
    auto* api_in_arg = api_def->add_in_arg();
    api_in_arg->set_name(op_in_arg.name());
    api_in_arg->set_rename_to(op_in_arg.name());
    api_in_arg->set_description(op_in_arg.description());

    *api_def->add_arg_order() = op_in_arg.name();
  }
  for (const auto& op_out_arg : op_def.output_arg()) {
    auto* api_out_arg = api_def->add_out_arg();
    api_out_arg->set_name(op_out_arg.name());
    api_out_arg->set_rename_to(op_out_arg.name());
    api_out_arg->set_description(op_out_arg.description());
  }
  for (const auto& op_attr : op_def.attr()) {
    auto* api_attr = api_def->add_attr();
    api_attr->set_name(op_attr.name());
    api_attr->set_rename_to(op_attr.name());
    if (op_attr.has_default_value()) {
      *api_attr->mutable_default_value() = op_attr.default_value();
    }
    api_attr->set_description(op_attr.description());
  }
  api_def->set_summary(op_def.summary());
  api_def->set_description(op_def.description());
}

// Updates base_arg based on overrides in new_arg.
void MergeArg(ApiDef::Arg* base_arg, const ApiDef::Arg& new_arg) {
  if (!new_arg.rename_to().empty()) {
    base_arg->set_rename_to(new_arg.rename_to());
  }
  if (!new_arg.description().empty()) {
    base_arg->set_description(new_arg.description());
  }
}

// Updates base_attr based on overrides in new_attr.
void MergeAttr(ApiDef::Attr* base_attr, const ApiDef::Attr& new_attr) {
  if (!new_attr.rename_to().empty()) {
    base_attr->set_rename_to(new_attr.rename_to());
  }
  if (new_attr.has_default_value()) {
    *base_attr->mutable_default_value() = new_attr.default_value();
  }
  if (!new_attr.description().empty()) {
    base_attr->set_description(new_attr.description());
  }
}

// Updates base_api_def based on overrides in new_api_def.
Status MergeApiDefs(ApiDef* base_api_def, const ApiDef& new_api_def) {
  // Merge visibility
  if (new_api_def.visibility() != ApiDef::DEFAULT_VISIBILITY) {
    base_api_def->set_visibility(new_api_def.visibility());
  }
  // Merge endpoints
  if (new_api_def.endpoint_size() > 0) {
    base_api_def->clear_endpoint();
    std::copy(
        new_api_def.endpoint().begin(), new_api_def.endpoint().end(),
        protobuf::RepeatedFieldBackInserter(base_api_def->mutable_endpoint()));
  }
  // Merge args
  for (const auto& new_arg : new_api_def.in_arg()) {
    bool found_base_arg = false;
    for (int i = 0; i < base_api_def->in_arg_size(); ++i) {
      auto* base_arg = base_api_def->mutable_in_arg(i);
      if (base_arg->name() == new_arg.name()) {
        MergeArg(base_arg, new_arg);
        found_base_arg = true;
        break;
      }
    }
    if (!found_base_arg) {
      return errors::FailedPrecondition("Argument ", new_arg.name(),
                                        " not defined in base api for ",
                                        base_api_def->graph_op_name());
    }
  }
  for (const auto& new_arg : new_api_def.out_arg()) {
    bool found_base_arg = false;
    for (int i = 0; i < base_api_def->out_arg_size(); ++i) {
      auto* base_arg = base_api_def->mutable_out_arg(i);
      if (base_arg->name() == new_arg.name()) {
        MergeArg(base_arg, new_arg);
        found_base_arg = true;
        break;
      }
    }
    if (!found_base_arg) {
      return errors::FailedPrecondition("Argument ", new_arg.name(),
                                        " not defined in base api for ",
                                        base_api_def->graph_op_name());
    }
  }
  // Merge arg order
  if (new_api_def.arg_order_size() > 0) {
    // Validate that new arg_order is correct.
    if (new_api_def.arg_order_size() != base_api_def->arg_order_size()) {
      return errors::FailedPrecondition(
          "Invalid number of arguments ", new_api_def.arg_order_size(), " for ",
          base_api_def->graph_op_name(),
          ". Expected: ", base_api_def->arg_order_size());
    }
    if (!std::is_permutation(new_api_def.arg_order().begin(),
                             new_api_def.arg_order().end(),
                             base_api_def->arg_order().begin())) {
      return errors::FailedPrecondition(
          "Invalid arg_order: ", str_util::Join(new_api_def.arg_order(), ", "),
          " for ", base_api_def->graph_op_name(),
          ". All elements in arg_order override must match base arg_order: ",
          str_util::Join(base_api_def->arg_order(), ", "));
    }

    base_api_def->clear_arg_order();
    std::copy(
        new_api_def.arg_order().begin(), new_api_def.arg_order().end(),
        protobuf::RepeatedFieldBackInserter(base_api_def->mutable_arg_order()));
  }
  // Merge attributes
  for (const auto& new_attr : new_api_def.attr()) {
    bool found_base_attr = false;
    for (int i = 0; i < base_api_def->attr_size(); ++i) {
      auto* base_attr = base_api_def->mutable_attr(i);
      if (base_attr->name() == new_attr.name()) {
        MergeAttr(base_attr, new_attr);
        found_base_attr = true;
        break;
      }
    }
    if (!found_base_attr) {
      return errors::FailedPrecondition("Attribute ", new_attr.name(),
                                        " not defined in base api for ",
                                        base_api_def->graph_op_name());
    }
  }
  // Merge summary
  if (!new_api_def.summary().empty()) {
    base_api_def->set_summary(new_api_def.summary());
  }
  // Merge description
  auto description = new_api_def.description().empty()
                         ? base_api_def->description()
                         : new_api_def.description();

  if (!new_api_def.description_prefix().empty()) {
    description =
        strings::StrCat(new_api_def.description_prefix(), "\n", description);
  }
  if (!new_api_def.description_suffix().empty()) {
    description =
        strings::StrCat(description, "\n", new_api_def.description_suffix());
  }
  base_api_def->set_description(description);
  return Status::OK();
}
}  // namespace

ApiDefMap::ApiDefMap(const OpList& op_list) {
  for (const auto& op : op_list.op()) {
    ApiDef api_def;
    InitApiDefFromOpDef(op, &api_def);
    map_[op.name()] = api_def;
  }
}

ApiDefMap::~ApiDefMap() {}

Status ApiDefMap::LoadFileList(Env* env, const std::vector<string>& filenames) {
  for (const auto& filename : filenames) {
    TF_RETURN_IF_ERROR(LoadFile(env, filename));
  }
  return Status::OK();
}

Status ApiDefMap::LoadFile(Env* env, const string& filename) {
  if (filename.empty()) return Status::OK();
  string contents;
  TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &contents));
  TF_RETURN_IF_ERROR(LoadApiDef(contents));
  return Status::OK();
}

Status ApiDefMap::LoadApiDef(const string& api_def_file_contents) {
  const string contents = PBTxtFromMultiline(api_def_file_contents);
  ApiDefs api_defs;
  protobuf::TextFormat::ParseFromString(contents, &api_defs);
  for (const auto& api_def : api_defs.op()) {
    // Check if the op definition is loaded. If op definition is not
    // loaded, then we just skip this ApiDef.
    if (map_.find(api_def.graph_op_name()) != map_.end()) {
      // Overwrite current api def with data in api_def.
      TF_RETURN_IF_ERROR(MergeApiDefs(&map_[api_def.graph_op_name()], api_def));
    }
  }
  return Status::OK();
}

void ApiDefMap::UpdateDocs() {
  for (auto& name_and_api_def : map_) {
    auto& api_def = name_and_api_def.second;
    CHECK_GT(api_def.endpoint_size(), 0);
    const string canonical_name = api_def.endpoint(0).name();
    if (api_def.graph_op_name() != canonical_name) {
      RenameInDocs(api_def.graph_op_name(), canonical_name, &api_def);
    }
    for (const auto& in_arg : api_def.in_arg()) {
      if (in_arg.name() != in_arg.rename_to()) {
        RenameInDocs(in_arg.name(), in_arg.rename_to(), &api_def);
      }
    }
    for (const auto& out_arg : api_def.out_arg()) {
      if (out_arg.name() != out_arg.rename_to()) {
        RenameInDocs(out_arg.name(), out_arg.rename_to(), &api_def);
      }
    }
    for (const auto& attr : api_def.attr()) {
      if (attr.name() != attr.rename_to()) {
        RenameInDocs(attr.name(), attr.rename_to(), &api_def);
      }
    }
  }
}

const tensorflow::ApiDef* ApiDefMap::GetApiDef(const string& name) const {
  return gtl::FindOrNull(map_, name);
}
}  // namespace tensorflow
