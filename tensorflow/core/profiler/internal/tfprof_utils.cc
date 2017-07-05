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
          !strings::safe_strto64(pieces[i + 1], &opts->min_micros)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[3]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_params)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[4]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_float_ops)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[5]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->min_occurrence)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[6]) {
      if (pieces.size() <= i + 1 ||
          !strings::safe_strto64(pieces[i + 1], &opts->step)) {
        return ReturnError(pieces, i);
      }
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[7]) {
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
    } else if (pieces[i] == tensorflow::tfprof::kOptions[8]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->account_type_regexes = str_util::Split(StripQuote(pieces[i + 1]),
                                                   ',', str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[9]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->start_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                 str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[10]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->trim_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[11]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->show_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[12]) {
      if (pieces.size() <= i + 1) {
        return ReturnError(pieces, i);
      }
      opts->hide_name_regexes = str_util::Split(StripQuote(pieces[i + 1]), ',',
                                                str_util::SkipEmpty());
      ++i;
    } else if (pieces[i] == tensorflow::tfprof::kOptions[13]) {
      if ((pieces.size() > i + 1 && pieces[i + 1].find("-") == 0) ||
          pieces.size() == i + 1) {
        opts->account_displayed_op_only = true;
      } else if (!StringToBool(pieces[i + 1],
                               &opts->account_displayed_op_only)) {
        return ReturnError(pieces, i);
      } else {
        ++i;
      }
    } else if (pieces[i] == tensorflow::tfprof::kOptions[14]) {
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
    } else if (pieces[i] == tensorflow::tfprof::kOptions[15]) {
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
      "\nSee go/tfprof for detail tutorial.\n"
      "\nCommands\n\n"
      "  scope: Each op has its op name in TensorFlow, such as 'n1', 'n1/n2', "
      "'n1/n2/n3'. 'n1/n2' is a child of 'n1'. 'scope' command builds "
      "a name scope tree and aggregates statistics based on it.\n\n"
      "  graph: ops in TensorFlow are organized as a graph based on their "
      "the source (inputs) and sink (outputs). 'graph' command builds "
      "a graph pointing *from output to input*, and aggregates "
      "statistics based on it.\n\n"
      "  set: Set options that will be default for follow up commands.\n\n"
      "  help: Show helps.\n"
      "\nOptions\n\n"
      "Press Enter in CLI to see default option values.\n\n"
      "  -max_depth: Show ops that are at most this number of hops from "
      "starting op in the tree/graph structure.\n\n"
      "  -min_bytes: Show ops that request at least this number of bytes.\n\n"
      "  -min_micros: Show ops that spend at least this number of micros to "
      "run.\n\n"
      "  -min_params: Show ops that contains at least this number of "
      "parameters.\n\n"
      "  -min_float_ops: Show ops that contain at least this number of "
      "float operations. Only available if an op has "
      "op.RegisterStatistics() defined and OpLog is "
      "provided\n\n"
      "  -min_occurrence: Show the op types that are at least used this number "
      "of times. Only available in op view.\n\n"
      "  -step: Show the stats of a step when multiple steps of "
      "RunMetadata were added. By default (-1), show the average of all steps."
      "  -order_by: Order the results by [name|depth|bytes|micros"
      "|accelerator_micros|cpu_micros|params|float_ops]\n\n"
      "  -account_type_regexes: Account and display the ops whose types match "
      "one of the type regexes specified. tfprof "
      "allow user to define extra op types for ops "
      "through tensorflow.tfprof.OpLog proto. regexes "
      "are comma-sperated.\n\n"
      "  -start_name_regexes: Show ops starting from the ops that matches the "
      "regexes, recursively. regexes are "
      "comma-separated.\n\n"
      "  -trim_name_regexes: Hide ops starting from the ops that matches the "
      "regexes, recursively, regexes are comma-seprated. "
      "\n\n"
      "  -show_name_regexes: Show ops that match the regexes. regexes are "
      "comma-seprated.\n\n"
      "  -hide_name_regexes: Hide ops that match the regexes. regexes are "
      "comma-seprated.\n\n"
      ""
      "  Notes: For each op, -acount_type_regexes is first evaluated, "
      "only ops with types matching the specified regexes are accounted and "
      "selected for displayed. -start/trim/show/hide_name_regexes are used "
      "to further filter ops for display. -start_name_regexes is evaluated "
      "first to search the starting ops to display. Descendants of starting "
      "ops are then evaluated against show/hide_name_regexes to make display "
      "decision. If an op matches trim_name_regexes, all its descendants are "
      "hidden.\n"
      "Ops statistics are *accounted even if they are hidden* as long as "
      "they match the -account_xxx options.\n\n"
      "  -account_displayed_op_only: If True, only account the statistics of "
      "ops eventually displayed. If False, account all "
      "op statistics matching -account_type_regexes recursively.\n\n"
      "  -select: Comma-separated list of metrics to show: [bytes|micros|"
      "accelerator_micros|cpu_micros|params|float_ops|tensor_value|device|"
      "op_types|input_shapes]."
      "\n\n"
      "  -dump_to_file: Dump the output to a file, instead of terminal.\n\n"
      ""
      "Examples\n"
      "  Assuming a toy model:\n"
      "    intput(typeB)->conv2d_1(typeA)->conv2d_2(typeA)->"
      "fc(typeA)->cost(typeA)->summarize(typeC)\n"
      "  Command:\n"
      "    tfprof> graph -account_type_regexes typeA -start_name_regexes "
      "cost.* -show_name_regexes conv2d.* -max_depth 10\n\n"
      "  The above command only aggregate statistics of all ops of typeA ("
      "hence ignoring input(typeB)). It will start looking for candidate to "
      "display from cost.* and finally displays conv2d_1 and conv2d_2.\n\n");
  fflush(stdout);
}

}  // namespace tfprof
}  // namespace tensorflow
