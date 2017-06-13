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
#include "tensorflow/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/tools/tfprof/internal/tfprof_stats.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/tools/tfprof/tfprof_log.pb.h"

using tensorflow::str_util::Split;

void completion(const char* buf, linenoiseCompletions* lc) {
  tensorflow::string buf_str = buf;
  if (buf_str.find(" ") == buf_str.npos) {
    for (const char* opt : tensorflow::tfprof::kCmds) {
      if (tensorflow::string(opt).find(buf_str) == 0) {
        linenoiseAddCompletion(lc, opt);
      }
    }
    return;
  }

  tensorflow::string prefix;
  int last_dash = buf_str.find_last_of(' ');
  if (last_dash != tensorflow::string::npos) {
    prefix = buf_str.substr(0, last_dash + 1);
    buf_str = buf_str.substr(last_dash + 1, tensorflow::kint32max);
  }
  for (const char* opt : tensorflow::tfprof::kOptions) {
    if (tensorflow::string(opt).find(buf_str) == 0) {
      linenoiseAddCompletion(lc, (prefix + opt).c_str());
    }
  }
}

int main(int argc, char** argv) {
  tensorflow::string FLAGS_graph_path = "";
  tensorflow::string FLAGS_run_meta_path = "";
  tensorflow::string FLAGS_op_log_path = "";
  tensorflow::string FLAGS_checkpoint_path = "";
  tensorflow::int32 FLAGS_max_depth = 10;
  tensorflow::int64 FLAGS_min_bytes = 0;
  tensorflow::int64 FLAGS_min_micros = 0;
  tensorflow::int64 FLAGS_min_params = 0;
  tensorflow::int64 FLAGS_min_float_ops = 0;
  tensorflow::int64 FLAGS_min_occurrence = 0;
  tensorflow::int64 FLAGS_step = -1;
  tensorflow::string FLAGS_order_by = "name";
  tensorflow::string FLAGS_account_type_regexes = ".*";
  tensorflow::string FLAGS_start_name_regexes = ".*";
  tensorflow::string FLAGS_trim_name_regexes = "";
  tensorflow::string FLAGS_show_name_regexes = ".*";
  tensorflow::string FLAGS_hide_name_regexes;
  bool FLAGS_account_displayed_op_only = false;
  tensorflow::string FLAGS_select = "params";
  tensorflow::string FLAGS_output = "";
  for (int i = 0; i < argc; i++) {
    fprintf(stderr, "%s\n", argv[i]);
  }

  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("graph_path", &FLAGS_graph_path,
                       "GraphDef proto text file name"),
      tensorflow::Flag("run_meta_path", &FLAGS_run_meta_path,
                       "Comma-separated list of RunMetadata proto binary "
                       "files. Each file is given step number 0,1,2,etc"),
      tensorflow::Flag("op_log_path", &FLAGS_op_log_path,
                       "tensorflow::tfprof::OpLog proto binary file name"),
      tensorflow::Flag("checkpoint_path", &FLAGS_checkpoint_path,
                       "TensorFlow Checkpoint file name"),
      tensorflow::Flag("max_depth", &FLAGS_max_depth, "max depth"),
      tensorflow::Flag("min_bytes", &FLAGS_min_bytes, "min_bytes"),
      tensorflow::Flag("min_micros", &FLAGS_min_micros, "min micros"),
      tensorflow::Flag("min_params", &FLAGS_min_params, "min params"),
      tensorflow::Flag("min_float_ops", &FLAGS_min_float_ops, "min float ops"),
      tensorflow::Flag("min_occurrence", &FLAGS_min_occurrence,
                       "min occurrence"),
      tensorflow::Flag("step", &FLAGS_step,
                       "The stats of which step to use. By default average"),
      tensorflow::Flag("order_by", &FLAGS_order_by, "order by"),
      tensorflow::Flag("account_type_regexes", &FLAGS_start_name_regexes,
                       "start name regexes"),
      tensorflow::Flag("trim_name_regexes", &FLAGS_trim_name_regexes,
                       "trim name regexes"),
      tensorflow::Flag("show_name_regexes", &FLAGS_show_name_regexes,
                       "show name regexes"),
      tensorflow::Flag("hide_name_regexes", &FLAGS_hide_name_regexes,
                       "hide name regexes"),
      tensorflow::Flag("account_displayed_op_only",
                       &FLAGS_account_displayed_op_only,
                       "account displayed op only"),
      tensorflow::Flag("select", &FLAGS_select, "select"),
      tensorflow::Flag("output", &FLAGS_output, "output"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    printf("%s", usage.c_str());
    return (2);
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  fprintf(stderr, "%s\n", FLAGS_graph_path.c_str());

  std::vector<tensorflow::string> account_type_regexes =
      Split(FLAGS_account_type_regexes, ',', tensorflow::str_util::SkipEmpty());
  std::vector<tensorflow::string> start_name_regexes =
      Split(FLAGS_start_name_regexes, ',', tensorflow::str_util::SkipEmpty());
  std::vector<tensorflow::string> trim_name_regexes =
      Split(FLAGS_trim_name_regexes, ',', tensorflow::str_util::SkipEmpty());
  std::vector<tensorflow::string> show_name_regexes =
      Split(FLAGS_show_name_regexes, ',', tensorflow::str_util::SkipEmpty());
  std::vector<tensorflow::string> hide_name_regexes =
      Split(FLAGS_hide_name_regexes, ',', tensorflow::str_util::SkipEmpty());
  std::vector<tensorflow::string> select =
      Split(FLAGS_select, ',', tensorflow::str_util::SkipEmpty());

  tensorflow::string output_type;
  std::map<tensorflow::string, tensorflow::string> output_options;
  tensorflow::Status s = tensorflow::tfprof::ParseOutput(
      FLAGS_output, &output_type, &output_options);
  CHECK(s.ok()) << s.ToString();

  tensorflow::string cmd = "";
  if (argc == 1 && FLAGS_graph_path.empty()) {
    printf("1) go/tfprof: Tutorial.\n");
    printf("2) tfprof help: Detail help information.\n");
    printf(
        "3) tfprof --graph_path <GraphDef proto text file>: "
        "Profiling model structure, tensor shape and # parameters.\n");
    printf(
        "4) tfprof --graph_path <GraphDef proto text file> \\\n"
        "          --run_meta_path <RunMetadata proto binary file> \\\n"
        "          --op_log_path <tensorflow::tfprof::OpLog proto binary file> "
        "\\\n"
        "          --checkpoint_path <TensorFlow Checkpoint file>: "
        "Profiling everything!\n");
    return 0;
  } else if (argc > 1) {
    if (tensorflow::string(argv[1]) == tensorflow::tfprof::kCmds[5]) {
      tensorflow::tfprof::PrintHelp();
      return 0;
    }
    if (tensorflow::string(argv[1]) == tensorflow::tfprof::kCmds[0] ||
        tensorflow::string(argv[1]) == tensorflow::tfprof::kCmds[1] ||
        tensorflow::string(argv[1]) == tensorflow::tfprof::kCmds[2] ||
        tensorflow::string(argv[1]) == tensorflow::tfprof::kCmds[3]) {
      cmd = argv[1];
    }
  }

  printf("Reading Files...\n");
  std::unique_ptr<tensorflow::GraphDef> graph(new tensorflow::GraphDef());
  TF_CHECK_OK(tensorflow::tfprof::ReadGraphDef(tensorflow::Env::Default(),
                                               FLAGS_graph_path, graph.get()));

  std::unique_ptr<tensorflow::tfprof::OpLog> op_log(
      new tensorflow::tfprof::OpLog());
  if (!FLAGS_op_log_path.empty()) {
    tensorflow::string op_log_str;
    s = tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                     FLAGS_op_log_path, &op_log_str);
    if (!s.ok()) {
      fprintf(stderr, "Failed to read op_log_path: %s\n", s.ToString().c_str());
      return 1;
    }
    if (!tensorflow::ParseProtoUnlimited(op_log.get(), op_log_str)) {
      fprintf(stderr, "Failed to parse op_log_path\n");
      return 1;
    }
  }

  std::unique_ptr<tensorflow::checkpoint::CheckpointReader> ckpt_reader;
  TF_Status* status = TF_NewStatus();
  if (!FLAGS_checkpoint_path.empty()) {
    ckpt_reader.reset(new tensorflow::checkpoint::CheckpointReader(
        FLAGS_checkpoint_path, status));
    if (TF_GetCode(status) != TF_OK) {
      fprintf(stderr, "%s\n", TF_Message(status));
      TF_DeleteStatus(status);
      return 1;
    }
    TF_DeleteStatus(status);
  }

  tensorflow::tfprof::TFStats tf_stat(
      std::move(graph), nullptr, std::move(op_log), std::move(ckpt_reader));

  std::vector<tensorflow::string> run_meta_files =
      Split(FLAGS_run_meta_path, ',', tensorflow::str_util::SkipEmpty());
  for (int i = 0; i < run_meta_files.size(); ++i) {
    std::unique_ptr<tensorflow::RunMetadata> run_meta(
        new tensorflow::RunMetadata());
    s = ReadBinaryProto(tensorflow::Env::Default(), run_meta_files[i],
                        run_meta.get());
    if (!s.ok()) {
      fprintf(stderr, "Failed to read run_meta_path %s. Status: %s\n",
              run_meta_files[i].c_str(), s.ToString().c_str());
      return 1;
    }
    tf_stat.ParseRunMeta(i, std::move(run_meta));
  }

  tensorflow::tfprof::Options opts(
      FLAGS_max_depth, FLAGS_min_bytes, FLAGS_min_micros, FLAGS_min_params,
      FLAGS_min_float_ops, FLAGS_min_occurrence, FLAGS_step, FLAGS_order_by,
      account_type_regexes, start_name_regexes, trim_name_regexes,
      show_name_regexes, hide_name_regexes, FLAGS_account_displayed_op_only,
      select, output_type, output_options);

  if (cmd == tensorflow::tfprof::kCmds[2] ||
      cmd == tensorflow::tfprof::kCmds[3]) {
    tf_stat.ShowMultiGraphNode(cmd, opts);
    return 0;
  } else if (cmd == tensorflow::tfprof::kCmds[0] ||
             cmd == tensorflow::tfprof::kCmds[1]) {
    tf_stat.ShowGraphNode(cmd, opts);
    return 0;
  }

  linenoiseSetCompletionCallback(completion);
  linenoiseHistoryLoad(".tfprof_history.txt");

  for (char* line = nullptr; (line = linenoise("tfprof> ")) != nullptr;) {
    tensorflow::string line_s = line;
    free(line);

    if (line_s.empty()) {
      printf("%s", opts.ToString().c_str());
      continue;
    }
    linenoiseHistoryAdd(line_s.c_str());
    linenoiseHistorySave(".tfprof_history.txt");

    tensorflow::tfprof::Options new_opts = opts;
    tensorflow::Status s =
        tensorflow::tfprof::ParseCmdLine(line_s, &cmd, &new_opts);
    if (!s.ok()) {
      fprintf(stderr, "E: %s\n", s.ToString().c_str());
      continue;
    }
    if (cmd == tensorflow::tfprof::kCmds[4]) {
      opts = new_opts;
    } else if (cmd == tensorflow::tfprof::kCmds[5]) {
      tensorflow::tfprof::PrintHelp();
    } else if (cmd == tensorflow::tfprof::kCmds[2] ||
               cmd == tensorflow::tfprof::kCmds[3]) {
      tf_stat.ShowMultiGraphNode(cmd, new_opts);
    } else if (cmd == tensorflow::tfprof::kCmds[0] ||
               cmd == tensorflow::tfprof::kCmds[1]) {
      tf_stat.ShowGraphNode(cmd, new_opts);
    }
  }
  return 0;
}
