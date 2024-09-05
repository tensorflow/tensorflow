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

#include "tensorflow/core/ops/compat/op_compatibility_lib.h"

#include <stdio.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

static string OpsHistoryDirectory(const string& ops_prefix,
                                  const string& history_version) {
  return io::JoinPath(ops_prefix,
                      strings::StrCat("compat/ops_history_", history_version));
}

static string OpsHistoryFile(const string& ops_prefix,
                             const string& history_version) {
  return io::JoinPath(ops_prefix, strings::StrCat("compat/ops_history.",
                                                  history_version, ".pbtxt"));
}

static string FileNameFromOpName(const string& op_name) {
  return strings::StrCat(op_name, ".pbtxt");
}

static void AddNewOpToHistory(const OpDef& op,
                              OpCompatibilityLib::OpHistory* out_op_history) {
  if (out_op_history != nullptr) {
    out_op_history->emplace_back(FileNameFromOpName(op.name()), OpList());
    *out_op_history->back().second.add_op() = op;
  }
}

static Status ReadOpHistory(Env* env, const string& file,
                            const string& directory,
                            OpCompatibilityLib::OpHistory* out) {
  // Read op history form `directory` if it exists there.
  std::vector<string> matching_files;
  Status status = env->GetMatchingPaths(io::JoinPath(directory, "*.pbtxt"),
                                        &matching_files);
  if (status.ok() && !matching_files.empty()) {
    printf("Reading op history from %s/*.pbtxt...\n", directory.c_str());
    std::sort(matching_files.begin(), matching_files.end());
    for (const string& full_file : matching_files) {
      string op_history_str;
      TF_RETURN_IF_ERROR(ReadFileToString(env, full_file, &op_history_str));
      OpList in_op_history;
      protobuf::TextFormat::ParseFromString(op_history_str, &in_op_history);
      const string file_tail = FileNameFromOpName(in_op_history.op(0).name());
      const string expected = io::JoinPath(directory, file_tail);
      if (full_file != expected) {
        return errors::Internal("Expected file paths to match but '", full_file,
                                "' != '", expected, "'");
      }
      out->emplace_back(file_tail, in_op_history);
    }
  } else {  // Otherwise, fall back to reading op history from `file`.
    printf("Reading op history from %s...\n", file.c_str());
    string op_history_str;
    TF_RETURN_IF_ERROR(ReadFileToString(env, file, &op_history_str));
    OpList in_op_history;
    protobuf::TextFormat::ParseFromString(op_history_str, &in_op_history);
    // Convert from a linear OpList to OpHistory format with one OpList per
    // unique op name.
    int start = 0;
    while (start < in_op_history.op_size()) {
      int end = start + 1;
      while (end < in_op_history.op_size() &&
             in_op_history.op(start).name() == in_op_history.op(end).name()) {
        ++end;
      }
      AddNewOpToHistory(in_op_history.op(start), out);
      for (++start; start < end; ++start) {
        *out->back().second.add_op() = in_op_history.op(start);
      }
    }
  }
  return absl::OkStatus();
}

OpCompatibilityLib::OpCompatibilityLib(const string& ops_prefix,
                                       const string& history_version,
                                       const std::set<string>* stable_ops)
    : ops_file_(io::JoinPath(ops_prefix, "ops.pbtxt")),
      op_history_file_(OpsHistoryFile(ops_prefix, history_version)),
      op_history_directory_(OpsHistoryDirectory(ops_prefix, history_version)),
      stable_ops_(stable_ops) {
  // Get the sorted list of all registered OpDefs.
  printf("Getting all registered ops...\n");
  OpRegistry::Global()->Export(false, &op_list_);
}

Status OpCompatibilityLib::ValidateCompatible(Env* env, int* changed_ops,
                                              int* added_ops,
                                              OpHistory* out_op_history) {
  *changed_ops = 0;
  *added_ops = 0;

  // Strip docs out of op_list_.
  RemoveDescriptionsFromOpList(&op_list_);

  if (stable_ops_ != nullptr) {
    printf("Verifying no stable ops have been removed...\n");
    std::vector<string> removed;
    // We rely on stable_ops_ and op_list_ being in sorted order.
    auto iter = stable_ops_->begin();
    for (int cur = 0; iter != stable_ops_->end() && cur < op_list_.op_size();
         ++cur) {
      const string& op_name = op_list_.op(cur).name();
      while (op_name > *iter) {
        removed.push_back(*iter);
        ++iter;
      }
      if (op_name == *iter) {
        ++iter;
      }
    }
    for (; iter != stable_ops_->end(); ++iter) {
      removed.push_back(*iter);
    }
    if (!removed.empty()) {
      return errors::InvalidArgument("Error, stable op(s) removed: ",
                                     absl::StrJoin(removed, ", "));
    }
  }

  OpHistory in_op_history;
  TF_RETURN_IF_ERROR(ReadOpHistory(env, op_history_file_, op_history_directory_,
                                   &in_op_history));

  int cur = 0;
  int hist = 0;

  printf("Verifying updates are compatible...\n");
  // Note: Op history is one OpList per unique op name in alphabetical order.
  // Within the OplList it has versions in oldest-first order.
  while (cur < op_list_.op_size() && hist < in_op_history.size()) {
    const OpDef& cur_op = op_list_.op(cur);
    const string& cur_op_name = cur_op.name();
    const OpList& history_op_list = in_op_history[hist].second;
    const string& history_op_name = history_op_list.op(0).name();
    if (stable_ops_ != nullptr && stable_ops_->count(cur_op_name) == 0) {
      // Ignore unstable op.
      for (++cur; cur < op_list_.op_size(); ++cur) {
        if (op_list_.op(cur).name() != cur_op_name) break;
      }
    } else if (cur_op_name < history_op_name) {
      // New op: add it.
      AddNewOpToHistory(cur_op, out_op_history);
      ++*added_ops;
      ++cur;
    } else if (cur_op_name > history_op_name) {
      if (stable_ops_ != nullptr) {
        // Okay to remove ops from the history that have been made unstable.
        ++hist;
      } else {
        // Op removed: error.
        return errors::InvalidArgument("Error, removed op: ",
                                       SummarizeOpDef(history_op_list.op(0)));
      }
    } else {
      // Op match.
      if (out_op_history != nullptr) {
        // Copy from in_op_history to *out_op_history.
        out_op_history->push_back(in_op_history[hist]);
      }

      const int end = history_op_list.op_size();
      // Is the last op in the history the same as the current op?
      // Compare using their serialized representations.
      string history_str, cur_str;
      history_op_list.op(end - 1).SerializeToString(&history_str);
      cur_op.SerializeToString(&cur_str);

      if (history_str != cur_str) {
        // Op changed, verify the change is compatible.
        for (int i = 0; i < end; ++i) {
          TF_RETURN_IF_ERROR(OpDefCompatible(history_op_list.op(i), cur_op));
        }

        // Verify default value of attrs has not been removed or modified
        // as compared to only the last historical version.
        TF_RETURN_IF_ERROR(
            OpDefAttrDefaultsUnchanged(history_op_list.op(end - 1), cur_op));

        // Check that attrs missing from history_op_list.op(0) don't change
        // their defaults.
        if (end > 1) {
          TF_RETURN_IF_ERROR(OpDefAddedDefaultsUnchanged(
              history_op_list.op(0), history_op_list.op(end - 1), cur_op));
        }

        // Compatible! Add changed op to the end of the history.
        if (out_op_history != nullptr) {
          *out_op_history->back().second.add_op() = cur_op;
        }
        ++*changed_ops;
      }

      // Advance past this op.
      ++hist;
      ++cur;
    }
  }

  // Error if missing ops.
  if (stable_ops_ == nullptr && hist < in_op_history.size()) {
    return errors::InvalidArgument(
        "Error, removed op: ",
        SummarizeOpDef(in_op_history[hist].second.op(0)));
  }

  // Add remaining new ops.
  for (; cur < op_list_.op_size(); ++cur) {
    const string& op_name = op_list_.op(cur).name();
    if (stable_ops_ != nullptr && stable_ops_->count(op_name) == 0) {
      // Ignore unstable op.
    } else {
      AddNewOpToHistory(op_list_.op(cur), out_op_history);
      ++*added_ops;
    }
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
