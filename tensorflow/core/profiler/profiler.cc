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

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "linenoise.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/profiler/internal/advisor/tfprof_advisor.h"
#include "tensorflow/core/profiler/internal/tfprof_options.h"
#include "tensorflow/core/profiler/internal/tfprof_stats.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"

namespace tensorflow {
namespace tfprof {
void completion(const char* buf, linenoiseCompletions* lc) {
  string buf_str = buf;
  if (buf_str.find(" ") == buf_str.npos) {
    for (const char* opt : kCmds) {
      if (string(opt).find(buf_str) == 0) {
        linenoiseAddCompletion(lc, opt);
      }
    }
    return;
  }

  string prefix;
  int last_dash = buf_str.find_last_of(' ');
  if (last_dash != string::npos) {
    prefix = buf_str.substr(0, last_dash + 1);
    buf_str = buf_str.substr(last_dash + 1, kint32max);
  }
  for (const char* opt : kOptions) {
    if (string(opt).find(buf_str) == 0) {
      linenoiseAddCompletion(lc, (prefix + opt).c_str());
    }
  }
}

int Run(int argc, char** argv) {
  string FLAGS_profile_path = "";
  string FLAGS_graph_path = "";
  string FLAGS_run_meta_path = "";
  string FLAGS_op_log_path = "";
  string FLAGS_checkpoint_path = "";
  int32 FLAGS_max_depth = 10;
  int64 FLAGS_min_bytes = 0;
  int64 FLAGS_min_peak_bytes = 0;
  int64 FLAGS_min_residual_bytes = 0;
  int64 FLAGS_min_output_bytes = 0;
  int64 FLAGS_min_micros = 0;
  int64 FLAGS_min_accelerator_micros = 0;
  int64 FLAGS_min_cpu_micros = 0;
  int64 FLAGS_min_params = 0;
  int64 FLAGS_min_float_ops = 0;
  int64 FLAGS_min_occurrence = 0;
  int64 FLAGS_step = -1;
  string FLAGS_order_by = "name";
  string FLAGS_account_type_regexes = ".*";
  string FLAGS_start_name_regexes = ".*";
  string FLAGS_trim_name_regexes = "";
  string FLAGS_show_name_regexes = ".*";
  string FLAGS_hide_name_regexes;
  bool FLAGS_account_displayed_op_only = false;
  string FLAGS_select = "micros";
  string FLAGS_output = "";
  for (int i = 0; i < argc; i++) {
    fprintf(stderr, "%s\n", argv[i]);
  }

  std::vector<Flag> flag_list = {
      Flag("profile_path", &FLAGS_profile_path, "Profile binary file name."),
      Flag("graph_path", &FLAGS_graph_path, "GraphDef proto text file name"),
      Flag("run_meta_path", &FLAGS_run_meta_path,
           "Comma-separated list of RunMetadata proto binary "
           "files. Each file is given step number 0,1,2,etc"),
      Flag("op_log_path", &FLAGS_op_log_path,
           "tensorflow::tfprof::OpLogProto proto binary file name"),
      Flag("checkpoint_path", &FLAGS_checkpoint_path,
           "TensorFlow Checkpoint file name"),
      Flag("max_depth", &FLAGS_max_depth, "max depth"),
      Flag("min_bytes", &FLAGS_min_bytes, "min_bytes"),
      Flag("min_peak_bytes", &FLAGS_min_peak_bytes, "min_peak_bytes"),
      Flag("min_residual_bytes", &FLAGS_min_residual_bytes,
           "min_residual_bytes"),
      Flag("min_output_bytes", &FLAGS_min_output_bytes, "min_output_bytes"),
      Flag("min_micros", &FLAGS_min_micros, "min micros"),
      Flag("min_accelerator_micros", &FLAGS_min_accelerator_micros,
           "min acclerator_micros"),
      Flag("min_cpu_micros", &FLAGS_min_cpu_micros, "min_cpu_micros"),
      Flag("min_params", &FLAGS_min_params, "min params"),
      Flag("min_float_ops", &FLAGS_min_float_ops, "min float ops"),
      Flag("min_occurrence", &FLAGS_min_occurrence, "min occurrence"),
      Flag("step", &FLAGS_step,
           "The stats of which step to use. By default average"),
      Flag("order_by", &FLAGS_order_by, "order by"),
      Flag("account_type_regexes", &FLAGS_start_name_regexes,
           "start name regexes"),
      Flag("trim_name_regexes", &FLAGS_trim_name_regexes, "trim name regexes"),
      Flag("show_name_regexes", &FLAGS_show_name_regexes, "show name regexes"),
      Flag("hide_name_regexes", &FLAGS_hide_name_regexes, "hide name regexes"),
      Flag("account_displayed_op_only", &FLAGS_account_displayed_op_only,
           "account displayed op only"),
      Flag("select", &FLAGS_select, "select"),
      Flag("output", &FLAGS_output, "output"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  bool parse_ok = Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    printf("%s", usage.c_str());
    return (2);
  }
  port::InitMain(argv[0], &argc, &argv);

  if (!FLAGS_profile_path.empty() && !FLAGS_graph_path.empty()) {
    fprintf(stderr,
            "both --graph_path and --profile_path are set. "
            "Ignore graph_path\n");
  }

  std::vector<string> account_type_regexes =
      str_util::Split(FLAGS_account_type_regexes, ',', str_util::SkipEmpty());
  std::vector<string> start_name_regexes =
      str_util::Split(FLAGS_start_name_regexes, ',', str_util::SkipEmpty());
  std::vector<string> trim_name_regexes =
      str_util::Split(FLAGS_trim_name_regexes, ',', str_util::SkipEmpty());
  std::vector<string> show_name_regexes =
      str_util::Split(FLAGS_show_name_regexes, ',', str_util::SkipEmpty());
  std::vector<string> hide_name_regexes =
      str_util::Split(FLAGS_hide_name_regexes, ',', str_util::SkipEmpty());
  std::vector<string> select =
      str_util::Split(FLAGS_select, ',', str_util::SkipEmpty());

  string output_type;
  std::map<string, string> output_options;
  Status s = ParseOutput(FLAGS_output, &output_type, &output_options);
  CHECK(s.ok()) << s.ToString();

  string cmd = "";
  if (argc == 1 && FLAGS_graph_path.empty() && FLAGS_profile_path.empty()) {
    PrintHelp();
    return 0;
  } else if (argc > 1) {
    if (string(argv[1]) == kCmds[6]) {
      PrintHelp();
      return 0;
    }
    if (string(argv[1]) == kCmds[0] || string(argv[1]) == kCmds[1] ||
        string(argv[1]) == kCmds[2] || string(argv[1]) == kCmds[3] ||
        string(argv[1]) == kCmds[4]) {
      cmd = argv[1];
    }
  }

  printf("Reading Files...\n");
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader;
  TF_Status* status = TF_NewStatus();
  if (!FLAGS_checkpoint_path.empty()) {
    ckpt_reader.reset(
        new checkpoint::CheckpointReader(FLAGS_checkpoint_path, status));
    if (TF_GetCode(status) != TF_OK) {
      fprintf(stderr, "%s\n", TF_Message(status));
      TF_DeleteStatus(status);
      return 1;
    }
    TF_DeleteStatus(status);
  }

  std::unique_ptr<TFStats> tf_stat;
  if (!FLAGS_profile_path.empty()) {
    tf_stat.reset(new TFStats(FLAGS_profile_path, std::move(ckpt_reader)));
  } else {
    printf(
        "Try to use a single --profile_path instead of "
        "graph_path,op_log_path,run_meta_path\n");
    std::unique_ptr<GraphDef> graph(new GraphDef());
    TF_CHECK_OK(
        ReadProtoFile(Env::Default(), FLAGS_graph_path, graph.get(), false));

    std::unique_ptr<OpLogProto> op_log(new OpLogProto());
    if (!FLAGS_op_log_path.empty()) {
      string op_log_str;
      s = ReadFileToString(Env::Default(), FLAGS_op_log_path, &op_log_str);
      if (!s.ok()) {
        fprintf(stderr, "Failed to read op_log_path: %s\n",
                s.ToString().c_str());
        return 1;
      }
      if (!ParseProtoUnlimited(op_log.get(), op_log_str)) {
        fprintf(stderr, "Failed to parse op_log_path\n");
        return 1;
      }
    }
    tf_stat.reset(new TFStats(std::move(graph), nullptr, std::move(op_log),
                              std::move(ckpt_reader)));

    std::vector<string> run_meta_files =
        str_util::Split(FLAGS_run_meta_path, ',', str_util::SkipEmpty());
    for (int i = 0; i < run_meta_files.size(); ++i) {
      std::unique_ptr<RunMetadata> run_meta(new RunMetadata());
      s = ReadProtoFile(Env::Default(), run_meta_files[i], run_meta.get(),
                        true);
      if (!s.ok()) {
        fprintf(stderr, "Failed to read run_meta_path %s. Status: %s\n",
                run_meta_files[i].c_str(), s.ToString().c_str());
        return 1;
      }
      tf_stat->AddRunMeta(i, std::move(run_meta));
      fprintf(stdout, "run graph coverage: %.2f\n", tf_stat->run_coverage());
    }
  }

  if (cmd == kCmds[4]) {
    tf_stat->BuildAllViews();
    Advisor(tf_stat.get()).Advise(Advisor::DefaultOptions());
    return 0;
  }

  Options opts(
      FLAGS_max_depth, FLAGS_min_bytes, FLAGS_min_peak_bytes,
      FLAGS_min_residual_bytes, FLAGS_min_output_bytes, FLAGS_min_micros,
      FLAGS_min_accelerator_micros, FLAGS_min_cpu_micros, FLAGS_min_params,
      FLAGS_min_float_ops, FLAGS_min_occurrence, FLAGS_step, FLAGS_order_by,
      account_type_regexes, start_name_regexes, trim_name_regexes,
      show_name_regexes, hide_name_regexes, FLAGS_account_displayed_op_only,
      select, output_type, output_options);

  if (cmd == kCmds[2] || cmd == kCmds[3]) {
    tf_stat->BuildView(cmd);
    tf_stat->ShowMultiGraphNode(cmd, opts);
    return 0;
  } else if (cmd == kCmds[0] || cmd == kCmds[1]) {
    tf_stat->BuildView(cmd);
    tf_stat->ShowGraphNode(cmd, opts);
    return 0;
  }

  linenoiseSetCompletionCallback(completion);
  linenoiseHistoryLoad(".tfprof_history.txt");

  bool looped = false;
  while (true) {
    char* line = linenoise("tfprof> ");
    if (line == nullptr) {
      if (!looped) {
        fprintf(stderr,
                "Cannot start interative shell, "
                "use 'bazel-bin' instead of 'bazel run'.\n");
      }
      break;
    }
    looped = true;
    string line_s = line;
    free(line);

    if (line_s.empty()) {
      printf("%s", opts.ToString().c_str());
      continue;
    }
    linenoiseHistoryAdd(line_s.c_str());
    linenoiseHistorySave(".tfprof_history.txt");

    Options new_opts = opts;
    Status s = ParseCmdLine(line_s, &cmd, &new_opts);
    if (!s.ok()) {
      fprintf(stderr, "E: %s\n", s.ToString().c_str());
      continue;
    }
    if (cmd == kCmds[5]) {
      opts = new_opts;
    } else if (cmd == kCmds[6]) {
      PrintHelp();
    } else if (cmd == kCmds[2] || cmd == kCmds[3]) {
      tf_stat->BuildView(cmd);
      tf_stat->ShowMultiGraphNode(cmd, new_opts);
    } else if (cmd == kCmds[0] || cmd == kCmds[1]) {
      tf_stat->BuildView(cmd);
      tf_stat->ShowGraphNode(cmd, new_opts);
    } else if (cmd == kCmds[4]) {
      tf_stat->BuildAllViews();
      Advisor(tf_stat.get()).Advise(Advisor::DefaultOptions());
    }
  }
  return 0;
}
}  // namespace tfprof
}  // namespace tensorflow

int main(int argc, char** argv) { return tensorflow::tfprof::Run(argc, argv); }
