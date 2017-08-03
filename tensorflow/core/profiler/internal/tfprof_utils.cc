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

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace tfprof {
string FormatNumber(int64 n) {
  if (n < 1000) {
    return strings::Printf("%lld", n);
  } else if (n < 1000000) {
    return strings::Printf("%.2fk", n / 1000.0);
  } else if (n < 1000000000) {
    return strings::Printf("%.2fm", n / 1000000.0);
  } else {
    return strings::Printf("%.2fb", n / 1000000000.0);
  }
}

string FormatTime(int64 micros) {
  if (micros < 1000) {
    return strings::Printf("%lldus", micros);
  } else if (micros < 1000000) {
    return strings::Printf("%.2fms", micros / 1000.0);
  } else {
    return strings::Printf("%.2fsec", micros / 1000000.0);
  }
}

string FormatMemory(int64 bytes) {
  if (bytes < 1000) {
    return strings::Printf("%lldB", bytes);
  } else if (bytes < 1000000) {
    return strings::Printf("%.2fKB", bytes / 1000.0);
  } else {
    return strings::Printf("%.2fMB", bytes / 1000000.0);
  }
}

string FormatShapes(const std::vector<int64>& shape) {
  return str_util::Join(shape, "x");
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
      tensorflow::error::INVALID_ARGUMENT,
      strings::StrCat("Invalid option '", pieces[idx], "' value: '", val, "'"));
}

bool CaseEqual(StringPiece s1, StringPiece s2) {
  if (s1.size() != s2.size()) return false;
  return str_util::Lowercase(s1) == str_util::Lowercase(s2);
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
  std::vector<string> pieces =
      str_util::Split(line, ' ', str_util::SkipEmpty());

  std::vector<string> cmds_str(kCmds, kCmds + sizeof(kCmds) / sizeof(*kCmds));
  if (std::find(cmds_str.begin(), cmds_str.end(), pieces[0]) ==
      cmds_str.end()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "First string must be a valid command.");
  }
  *cmd = pieces[0];

  for (int i = 1; i < pieces.size(); ++i) {
    if (pieces[i] == string(tensorflow::tfprof::kOptions[0])) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto32(pieces[i + 1], &opts->max_depth)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[1]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[2]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_peak_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[3]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_residual_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[4]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_output_bytes)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[5]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[6]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1],
                                 &opts->min_accelerator_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[7]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_cpu_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[8]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_params)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[9]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_float_ops)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[10]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_occurrence)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[11]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->step)) {
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
      opts->account_type_regexes = str_util::Split(StripQuote(pieces[i + 1]),
                                                   ',', str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[14]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->start_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                 str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[15]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->trim_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[16]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->show_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[17]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->hide_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[18]) {
      if ((pieces.size() > i + 1 && pieces[i + 1].find("-") == 0) ||
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
      std::vector<string> requested_vector = str_util::Split(
          StripQuote(pieces[i + 1]), ',', str_util::SkipEmpty());
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
  return tensorflow::Status::OK();
}

void PrintHelp() {
  printf(
      "See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/"
      "README.md for profiler tutorial.\n");
  printf(
      "See https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/"
      "g3doc/command_line.md for command line tool tutorial.\n");
  printf(
      "profiler --graph_path=<GraphDef proto file>  # required\n"
      "         --run_meta_patn=<RunMetadata proto file>  # optional\n"
      "         --run_log_path=<OpLogProto proto file>  # optional\n\n");
  printf(
      "\nCommands:\n"
      "  scope: Organize profiles based on name scopes.\n"
      "  graph: Organize profiles based on graph node input/output.\n"
      "  op: Organize profiles based on operation type.\n"
      "  code: Organize profiles based on python codes (need op_log_path).\n"
      "  advise: Auto-profile and advise.\n"
      "  set: Set options that will be default for follow up commands.\n"
      "  help: Show helps.\n");
  fflush(stdout);
}

}  // namespace tfprof
}  // namespace tensorflow
