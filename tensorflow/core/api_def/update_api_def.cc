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
#include "tensorflow/core/api_def/update_api_def.h"

#include <ctype.h>

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/api_def/excluded_ops.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace {
constexpr char kApiDefFileFormat[] = "api_def_%s.pbtxt";
// TODO(annarev): look into supporting other prefixes, not just 'doc'.
constexpr char kDocStart[] = ".Doc(R\"doc(";
constexpr char kDocEnd[] = ")doc\")";

// Updates api_def based on the given op.
void FillBaseApiDef(ApiDef* api_def, const OpDef& op) {
  api_def->set_graph_op_name(op.name());
  // Add arg docs
  for (auto& input_arg : op.input_arg()) {
    if (!input_arg.description().empty()) {
      auto* api_def_in_arg = api_def->add_in_arg();
      api_def_in_arg->set_name(input_arg.name());
      api_def_in_arg->set_description(input_arg.description());
    }
  }
  for (auto& output_arg : op.output_arg()) {
    if (!output_arg.description().empty()) {
      auto* api_def_out_arg = api_def->add_out_arg();
      api_def_out_arg->set_name(output_arg.name());
      api_def_out_arg->set_description(output_arg.description());
    }
  }
  // Add attr docs
  for (auto& attr : op.attr()) {
    if (!attr.description().empty()) {
      auto* api_def_attr = api_def->add_attr();
      api_def_attr->set_name(attr.name());
      api_def_attr->set_description(attr.description());
    }
  }
  // Add docs
  api_def->set_summary(op.summary());
  api_def->set_description(op.description());
}

// Returns true if op has any description or summary.
bool OpHasDocs(const OpDef& op) {
  if (!op.summary().empty() || !op.description().empty()) {
    return true;
  }
  for (const auto& arg : op.input_arg()) {
    if (!arg.description().empty()) {
      return true;
    }
  }
  for (const auto& arg : op.output_arg()) {
    if (!arg.description().empty()) {
      return true;
    }
  }
  for (const auto& attr : op.attr()) {
    if (!attr.description().empty()) {
      return true;
    }
  }
  return false;
}

// Returns true if summary and all descriptions are the same in op1
// and op2.
bool CheckDocsMatch(const OpDef& op1, const OpDef& op2) {
  if (op1.summary() != op2.summary() ||
      op1.description() != op2.description() ||
      op1.input_arg_size() != op2.input_arg_size() ||
      op1.output_arg_size() != op2.output_arg_size() ||
      op1.attr_size() != op2.attr_size()) {
    return false;
  }
  // Iterate over args and attrs to compare their docs.
  for (int i = 0; i < op1.input_arg_size(); ++i) {
    if (op1.input_arg(i).description() != op2.input_arg(i).description()) {
      return false;
    }
  }
  for (int i = 0; i < op1.output_arg_size(); ++i) {
    if (op1.output_arg(i).description() != op2.output_arg(i).description()) {
      return false;
    }
  }
  for (int i = 0; i < op1.attr_size(); ++i) {
    if (op1.attr(i).description() != op2.attr(i).description()) {
      return false;
    }
  }
  return true;
}

// Returns true if descriptions and summaries in op match a
// given single doc-string.
bool ValidateOpDocs(const OpDef& op, const string& doc) {
  OpDefBuilder b(op.name());
  // We don't really care about type we use for arguments and
  // attributes. We just want to make sure attribute and argument names
  // are added so that descriptions can be assigned to them when parsing
  // documentation.
  for (const auto& arg : op.input_arg()) {
    b.Input(arg.name() + ":string");
  }
  for (const auto& arg : op.output_arg()) {
    b.Output(arg.name() + ":string");
  }
  for (const auto& attr : op.attr()) {
    b.Attr(attr.name() + ":string");
  }
  b.Doc(doc);
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(b.Finalize(&op_reg_data));
  return CheckDocsMatch(op, op_reg_data.op_def);
}
}  // namespace

string RemoveDoc(const OpDef& op, const string& file_contents,
                 size_t start_location) {
  // Look for a line starting with .Doc( after the REGISTER_OP.
  const auto doc_start_location = file_contents.find(kDocStart, start_location);
  const string format_error = strings::Printf(
      "Could not find %s doc for removal. Make sure the doc is defined with "
      "'%s' prefix and '%s' suffix or remove the doc manually.",
      op.name().c_str(), kDocStart, kDocEnd);
  if (doc_start_location == string::npos) {
    std::cerr << format_error << std::endl;
    LOG(ERROR) << "Didn't find doc start";
    return file_contents;
  }
  const auto doc_end_location = file_contents.find(kDocEnd, doc_start_location);
  if (doc_end_location == string::npos) {
    LOG(ERROR) << "Didn't find doc start";
    std::cerr << format_error << std::endl;
    return file_contents;
  }

  const auto doc_start_size = sizeof(kDocStart) - 1;
  string doc_text = file_contents.substr(
      doc_start_location + doc_start_size,
      doc_end_location - doc_start_location - doc_start_size);

  // Make sure the doc text we found actually matches OpDef docs to
  // avoid removing incorrect text.
  if (!ValidateOpDocs(op, doc_text)) {
    LOG(ERROR) << "Invalid doc: " << doc_text;
    std::cerr << format_error << std::endl;
    return file_contents;
  }
  // Remove .Doc call.
  auto before_doc = file_contents.substr(0, doc_start_location);
  absl::StripTrailingAsciiWhitespace(&before_doc);
  return before_doc +
         file_contents.substr(doc_end_location + sizeof(kDocEnd) - 1);
}

namespace {
// Remove .Doc calls that follow REGISTER_OP calls for the given ops.
// We search for REGISTER_OP calls in the given op_files list.
void RemoveDocs(const std::vector<const OpDef*>& ops,
                const std::vector<string>& op_files) {
  // Set of ops that we already found REGISTER_OP calls for.
  std::set<string> processed_ops;

  for (const auto& file : op_files) {
    string file_contents;
    bool file_contents_updated = false;
    TF_CHECK_OK(ReadFileToString(Env::Default(), file, &file_contents));

    for (auto op : ops) {
      if (processed_ops.find(op->name()) != processed_ops.end()) {
        // We already found REGISTER_OP call for this op in another file.
        continue;
      }
      string register_call =
          strings::Printf("REGISTER_OP(\"%s\")", op->name().c_str());
      const auto register_call_location = file_contents.find(register_call);
      // Find REGISTER_OP(OpName) call.
      if (register_call_location == string::npos) {
        continue;
      }
      std::cout << "Removing .Doc call for " << op->name() << " from " << file
                << "." << std::endl;
      file_contents = RemoveDoc(*op, file_contents, register_call_location);
      file_contents_updated = true;

      processed_ops.insert(op->name());
    }
    if (file_contents_updated) {
      TF_CHECK_OK(WriteStringToFile(Env::Default(), file, file_contents))
          << "Could not remove .Doc calls in " << file
          << ". Make sure the file is writable.";
    }
  }
}
}  // namespace

// Returns ApiDefs text representation in multi-line format
// constructed based on the given op.
string CreateApiDef(const OpDef& op) {
  ApiDefs api_defs;
  FillBaseApiDef(api_defs.add_op(), op);

  const std::vector<string> multi_line_fields = {"description"};
  std::string new_api_defs_str;
  ::tensorflow::protobuf::TextFormat::PrintToString(api_defs,
                                                    &new_api_defs_str);
  return PBTxtToMultiline(new_api_defs_str, multi_line_fields);
}

// Creates ApiDef files for any new ops.
// If op_file_pattern is not empty, then also removes .Doc calls from
// new op registrations in these files.
void CreateApiDefs(const OpList& ops, const string& api_def_dir,
                   const string& op_file_pattern) {
  auto* excluded_ops = GetExcludedOps();
  std::vector<const OpDef*> new_ops_with_docs;

  for (const auto& op : ops.op()) {
    if (excluded_ops->find(op.name()) != excluded_ops->end()) {
      continue;
    }
    // Form the expected ApiDef path.
    string file_path =
        io::JoinPath(tensorflow::string(api_def_dir), kApiDefFileFormat);
    file_path = strings::Printf(file_path.c_str(), op.name().c_str());

    // Create ApiDef if it doesn't exist.
    if (!Env::Default()->FileExists(file_path).ok()) {
      std::cout << "Creating ApiDef file " << file_path << std::endl;
      const auto& api_def_text = CreateApiDef(op);
      TF_CHECK_OK(WriteStringToFile(Env::Default(), file_path, api_def_text));

      if (OpHasDocs(op)) {
        new_ops_with_docs.push_back(&op);
      }
    }
  }
  if (!op_file_pattern.empty()) {
    std::vector<string> op_files;
    TF_CHECK_OK(Env::Default()->GetMatchingPaths(op_file_pattern, &op_files));
    RemoveDocs(new_ops_with_docs, op_files);
  }
}
}  // namespace tensorflow
