/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_utils.h"

#include <stdio.h>

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {
string FormatNumber(int64_t n) {
  if (n < 1000) {
    return absl::StrFormat("%d", n);
  } else if (n < 1000000) {
    return absl::StrFormat("%.2fk", n / 1000.0);
  } else if (n < 1000000000) {
    return absl::StrFormat("%.2fm", n / 1000000.0);
  } else {
    return absl::StrFormat("%.2fb", n / 1000000000.0);
  }
}

string FormatTime(int64_t micros) {
  if (micros < 1000) {
    return absl::StrFormat("%dus", micros);
  } else if (micros < 1000000) {
    return absl::StrFormat("%.2fms", micros / 1000.0);
  } else {
    return absl::StrFormat("%.2fsec", micros / 1000000.0);
  }
}

string FormatMemory(int64_t bytes) {
  if (bytes < 1000) {
    return absl::StrFormat("%dB", bytes);
  } else if (bytes < 1000000) {
    return absl::StrFormat("%.2fKB", bytes / 1000.0);
  } else {
    return absl::StrFormat("%.2fMB", bytes / 1000000.0);
  }
}

string FormatShapes(const std::vector<int64_t>& shape) {
  return absl::StrJoin(shape, "x");
}

string StringReplace(const string& str, const string& oldsub,
                     const string& newsub) {
  string out = str;
  RE2::GlobalReplace(&out, oldsub, newsub);
  return out;
}

namespace {
string StripQuote(const string& s) {
  int start = s.find_first_not_of("\"\'");
  int end = s.find_last_not_of("\"\'");
  if (start == s.npos || end == s.npos) return "";

  return s.substr(start, end - start + 1);
}

tensorflow::Status ReturnError(const std::vector<string>& pieces, int idx) {
  string val;
  if (pieces.size() > idx + 1) {
    val = pieces[idx + 1];
  }
  return tensorflow::Status(
      absl::StatusCode::kInvalidArgument,
      absl::StrCat("Invalid option '", pieces[idx], "' value: '", val, "'"));
}

bool CaseEqual(StringPiece s1, StringPiece s2) {
  if (s1.size() != s2.size()) return false;
  return absl::AsciiStrToLower(s1) == absl::AsciiStrToLower(s2);
}

bool StringToBool(StringPiece str, bool* value) {
  CHECK(value != nullptr) << "NULL output boolean given.";
  if (CaseEqual(str, "true") || CaseEqual(str, "t") || CaseEqual(str, "yes") ||
      CaseEqual(str, "y") || CaseEqual(str, "1")) {
    *value = true;
    return true;
  }
  if (CaseEqual(str, "false") || CaseEqual(str, "f") || CaseEqual(str, "no") ||
      CaseEqual(str, "n") || CaseEqual(str, "0")) {
    *value = false;
    return true;
  }
  return false;
}
}  // namespace

tensorflow::Status ParseCmdLine(const string& line, string* cmd,
                                tensorflow::tfprof::Options* opts) {
  std::vector<string> pieces = absl::StrSplit(line, ' ', absl::SkipEmpty());

  std::vector<string> cmds_str(kCmds, kCmds + sizeof(kCmds) / sizeof(*kCmds));
  if (std::find(cmds_str.begin(), cmds_str.end(), pieces[0]) ==
      cmds_str.end()) {
    return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                              "First string must be a valid command.");
  }
  *cmd = pieces[0];

  for (int i = 1; i < pieces.size(); ++i) {
    if (pieces[i] == string(tensorflow::tfprof::kOptions[0])) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->max_depth)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[1]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[2]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_peak_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[3]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_residual_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[4]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_output_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[5]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[6]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_accelerator_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[7]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_cpu_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[8]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_params)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[9]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_float_ops)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[10]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->min_occurrence)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[11]) {
      if (pieces.size() <= i + 1 ||
          !absl::SimpleAtoi(pieces[i + 1], &opts->step)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[12]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      std::set<string> order_by_set(
          kOrderBy, kOrderBy + sizeof(kOrderBy) / sizeof(*kOrderBy));
      auto order_by = order_by_set.find(pieces[i + 1]);
      if (order_by == order_by_set.end()) {
        return ReturnError(pieces, i);
      }
      opts->order_by = *order_by;
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[13]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->account_type_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[14]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->start_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[15]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->trim_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[16]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->show_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[17]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->hide_name_regexes =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[18]) {
      if ((pieces.size() > i + 1 && absl::StartsWith(pieces[i + 1], "-")) ||
          pieces.size() == i + 1) {
        opts->account_displayed_op_only = true;
      } else if (!StringToBool(pieces[i + 1],
                               &opts->account_displayed_op_only)) {
        return ReturnError(pieces, i);
      } else {
        ++i;
      }
    } else if (pieces[i] == tensorflow::tfprof::kOptions[19]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      std::set<string> shown_set(kShown,
                                 kShown + sizeof(kShown) / sizeof(*kShown));
      std::vector<string> requested_vector =
          absl::StrSplit(StripQuote(pieces[i + 1]), ',', absl::SkipEmpty());
      std::set<string> requested_set(requested_vector.begin(),
                                     requested_vector.end());
      for (const string& requested : requested_set) {
        if (shown_set.find(requested) == shown_set.end()) {
          return ReturnError(pieces, i);
        }
      }
      opts->select = requested_set;
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[20]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }

      tensorflow::Status s =
          ParseOutput(pieces[i + 1], &opts->output_type, &opts->output_options);
      if (!s.ok()) return s;
      ++i;
    } else {
      return ReturnError(pieces, i);
    }
  }
  return OkStatus();
}

void PrintHelp() {
  absl::PrintF(
      "See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/"
      "README.md for profiler tutorial.\n");
  absl::PrintF(
      "See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/"
      "g3doc/command_line.md for command line tool tutorial.\n");
  absl::PrintF(
      "profiler --profile_path=<ProfileProto binary file> # required\n"
      "\nOr:\n\n"
      "profiler --graph_path=<GraphDef proto file>  "
      "# Contains model graph info (no needed for eager execution)\n"
      "         --run_meta_path=<RunMetadata proto file>  "
      "# Contains runtime info. Optional.\n"
      "         --run_log_path=<OpLogProto proto file>  "
      "# Contains extra source code, flops, custom type info. Optional\n\n");
  absl::PrintF(
      "\nTo skip interactive mode, append one of the following commands:\n"
      "  scope: Organize profiles based on name scopes.\n"
      "  graph: Organize profiles based on graph node input/output.\n"
      "  op: Organize profiles based on operation type.\n"
      "  code: Organize profiles based on python codes (need op_log_path).\n"
      "  advise: Auto-profile and advise. (experimental)\n"
      "  set: Set options that will be default for follow up commands.\n"
      "  help: Show helps.\n");
  fflush(stdout);
}

static const char* const kTotalMicrosHelp =
    "total execution time: Sum of accelerator execution time and cpu execution "
    "time.";
static const char* const kAccMicrosHelp =
    "accelerator execution time: Time spent executing on the accelerator. "
    "This is normally measured by the actual hardware library.";
static const char* const kCPUHelp =
    "cpu execution time: The time from the start to the end of the operation. "
    "It's the sum of actual cpu run time plus the time that it spends waiting "
    "if part of computation is launched asynchronously.";
static const char* const kBytes =
    "requested bytes: The memory requested by the operation, accumulatively.";
static const char* const kPeakBytes =
    "peak bytes: The peak amount of memory that the operation is holding at "
    "some point.";
static const char* const kResidualBytes =
    "residual bytes: The memory not de-allocated after the operation finishes.";
static const char* const kOutputBytes =
    "output bytes: The memory that is output from the operation (not "
    "necessarily allocated by the operation)";
static const char* const kOccurrence =
    "occurrence: The number of times it occurs";
static const char* const kInputShapes =
    "input shape: The shape of input tensors";
static const char* const kDevice = "device: which device is placed on.";
static const char* const kFloatOps =
    "flops: Number of float operations. Note: Please read the implementation "
    "for the math behind it.";
static const char* const kParams =
    "param: Number of parameters (in the Variable).";
static const char* const kTensorValue = "tensor_value: Not supported now.";
static const char* const kOpTypes =
    "op_types: The attributes of the operation, includes the Kernel name "
    "device placed on and user-defined strings.";

static const char* const kScope =
    "scope: The nodes in the model graph are organized by their names, which "
    "is hierarchical like filesystem.";
static const char* const kCode =
    "code: When python trace is available, the nodes are python lines and "
    "their are organized by the python call stack.";
static const char* const kOp =
    "op: The nodes are operation kernel type, such as MatMul, Conv2D. Graph "
    "nodes belonging to the same type are aggregated together.";
static const char* const kAdvise =
    "advise: Automatically profile and discover issues. (Experimental)";
static const char* const kSet =
    "set: Set a value for an option for future use.";
static const char* const kHelp = "help: Print helping messages.";

string QueryDoc(const string& cmd, const Options& opts) {
  string cmd_help = "";
  if (cmd == kCmds[0]) {
    cmd_help = kScope;
  } else if (cmd == kCmds[1]) {
    cmd_help = kScope;
  } else if (cmd == kCmds[2]) {
    cmd_help = kCode;
  } else if (cmd == kCmds[3]) {
    cmd_help = kOp;
  } else if (cmd == kCmds[4]) {
    cmd_help = kAdvise;
  } else if (cmd == kCmds[5]) {
    cmd_help = kSet;
  } else if (cmd == kCmds[6]) {
    cmd_help = kHelp;
  } else {
    cmd_help = "Unknown command: " + cmd;
  }

  std::vector<string> helps;
  for (const string& s : opts.select) {
    if (s == kShown[0]) {
      helps.push_back(kBytes);
    } else if (s == kShown[1]) {
      helps.push_back(
          absl::StrCat(kTotalMicrosHelp, "\n", kCPUHelp, "\n", kAccMicrosHelp));
    } else if (s == kShown[2]) {
      helps.push_back(kParams);
    } else if (s == kShown[3]) {
      helps.push_back(kFloatOps);
    } else if (s == kShown[4]) {
      helps.push_back(kTensorValue);
    } else if (s == kShown[5]) {
      helps.push_back(kDevice);
    } else if (s == kShown[6]) {
      helps.push_back(kOpTypes);
    } else if (s == kShown[7]) {
      helps.push_back(kOccurrence);
    } else if (s == kShown[8]) {
      helps.push_back(kInputShapes);
    } else if (s == kShown[9]) {
      helps.push_back(kAccMicrosHelp);
    } else if (s == kShown[10]) {
      helps.push_back(kCPUHelp);
    } else if (s == kShown[11]) {
      helps.push_back(kPeakBytes);
    } else if (s == kShown[12]) {
      helps.push_back(kResidualBytes);
    } else if (s == kShown[13]) {
      helps.push_back(kOutputBytes);
    } else {
      helps.push_back("Unknown select: " + s);
    }
  }
  return absl::StrCat("\nDoc:\n", cmd_help, "\n", absl::StrJoin(helps, "\n"),
                      "\n\n");
}

}  // namespace tfprof
}  // namespace tensorflow
