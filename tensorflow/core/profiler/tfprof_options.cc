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

#include "tensorflow/core/profiler/tfprof_options.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/tfprof_options.pb.h"

namespace tensorflow {
namespace tfprof {
namespace {
string KeyValueToStr(const std::map<string, string>& kv_map) {
  std::vector<string> kv_vec;
  kv_vec.reserve(kv_map.size());
  for (const auto& pair : kv_map) {
    kv_vec.push_back(absl::StrCat(pair.first, "=", pair.second));
  }
  return absl::StrJoin(kv_vec, ",");
}
}  // namespace

tensorflow::Status ParseOutput(const string& output_opt, string* output_type,
                               std::map<string, string>* output_options) {
  // The default is to use stdout.
  if (output_opt.empty()) {
    *output_type = kOutput[1];
    return OkStatus();
  }

  std::set<string> output_types(kOutput,
                                kOutput + sizeof(kOutput) / sizeof(*kOutput));
  auto opt_split = output_opt.find(':');
  std::vector<string> kv_split;
  if (opt_split == output_opt.npos) {
    if (output_types.find(output_opt) == output_types.end()) {
      return tensorflow::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("E.g. Unknown output type: %s, Valid types: %s\n",
                          output_opt, absl::StrJoin(output_types, ",")));
    }
    *output_type = output_opt;
  } else {
    *output_type = output_opt.substr(0, opt_split);
    if (output_types.find(*output_type) == output_types.end()) {
      return tensorflow::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("E.g. Unknown output type: %s, Valid types: %s\n",
                          *output_type, absl::StrJoin(output_types, ",")));
    }
    kv_split = absl::StrSplit(output_opt.substr(opt_split + 1), ',',
                              absl::SkipEmpty());
  }

  std::set<string> valid_options;
  std::set<string> required_options;
  if (*output_type == kOutput[0]) {
    valid_options.insert(
        kTimelineOpts,
        kTimelineOpts + sizeof(kTimelineOpts) / sizeof(*kTimelineOpts));
    required_options.insert(
        kTimelineRequiredOpts,
        kTimelineRequiredOpts +
            sizeof(kTimelineRequiredOpts) / sizeof(*kTimelineRequiredOpts));
  } else if (*output_type == kOutput[2]) {
    valid_options.insert(kFileOpts,
                         kFileOpts + sizeof(kFileOpts) / sizeof(*kFileOpts));
    required_options.insert(kFileRequiredOpts,
                            kFileRequiredOpts + sizeof(kFileRequiredOpts) /
                                                    sizeof(*kFileRequiredOpts));
  } else if (*output_type == kOutput[3]) {
    valid_options.insert(kPprofOpts,
                         kPprofOpts + sizeof(kPprofOpts) / sizeof(*kPprofOpts));
    required_options.insert(
        kPprofRequiredOpts,
        kPprofRequiredOpts +
            sizeof(kPprofRequiredOpts) / sizeof(*kPprofRequiredOpts));
  }

  for (const string& kv_str : kv_split) {
    const std::vector<string> kv =
        absl::StrSplit(kv_str, '=', absl::SkipEmpty());
    if (kv.size() < 2) {
      return tensorflow::Status(
          absl::StatusCode::kInvalidArgument,
          "Visualize format: -output timeline:key=value,key=value,...");
    }
    if (valid_options.find(kv[0]) == valid_options.end()) {
      return tensorflow::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Unrecognized options %s for output_type: %s\n",
                          kv[0], *output_type));
    }
    const std::vector<string> kv_without_key(kv.begin() + 1, kv.end());
    (*output_options)[kv[0]] = absl::StrJoin(kv_without_key, "=");
  }

  for (const string& opt : required_options) {
    if (output_options->find(opt) == output_options->end()) {
      return tensorflow::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Missing required output_options for %s\n"
                          "E.g. -output %s:%s=...\n",
                          *output_type, *output_type, opt));
    }
  }
  return OkStatus();
}

tensorflow::Status Options::FromProtoStr(const string& opts_proto_str,
                                         Options* opts) {
  OptionsProto opts_pb;
  if (!opts_pb.ParseFromString(opts_proto_str)) {
    return tensorflow::Status(
        absl::StatusCode::kInternal,
        absl::StrCat("Failed to parse option string from Python API: ",
                     opts_proto_str));
  }

  string output_type;
  std::map<string, string> output_options;
  tensorflow::Status s =
      ParseOutput(opts_pb.output(), &output_type, &output_options);
  if (!s.ok()) return s;

  if (!opts_pb.dump_to_file().empty()) {
    absl::FPrintF(stderr,
                  "-dump_to_file option is deprecated. "
                  "Please use -output file:outfile=<filename>\n");
    absl::FPrintF(stderr,
                  "-output %s is overwritten with -output file:outfile=%s\n",
                  opts_pb.output(), opts_pb.dump_to_file());
    output_type = kOutput[2];
    output_options.clear();
    output_options[kFileOpts[0]] = opts_pb.dump_to_file();
  }

  *opts = Options(
      opts_pb.max_depth(), opts_pb.min_bytes(), opts_pb.min_peak_bytes(),
      opts_pb.min_residual_bytes(), opts_pb.min_output_bytes(),
      opts_pb.min_micros(), opts_pb.min_accelerator_micros(),
      opts_pb.min_cpu_micros(), opts_pb.min_params(), opts_pb.min_float_ops(),
      opts_pb.min_occurrence(), opts_pb.step(), opts_pb.order_by(),
      std::vector<string>(opts_pb.account_type_regexes().begin(),
                          opts_pb.account_type_regexes().end()),
      std::vector<string>(opts_pb.start_name_regexes().begin(),
                          opts_pb.start_name_regexes().end()),
      std::vector<string>(opts_pb.trim_name_regexes().begin(),
                          opts_pb.trim_name_regexes().end()),
      std::vector<string>(opts_pb.show_name_regexes().begin(),
                          opts_pb.show_name_regexes().end()),
      std::vector<string>(opts_pb.hide_name_regexes().begin(),
                          opts_pb.hide_name_regexes().end()),
      opts_pb.account_displayed_op_only(),
      std::vector<string>(opts_pb.select().begin(), opts_pb.select().end()),
      output_type, output_options);
  return OkStatus();
}

std::string Options::ToString() const {
  // clang-format off
  const std::string s = absl::StrFormat(
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%d\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s:%s\n",
      kOptions[0],  max_depth,
      kOptions[1],  min_bytes,
      kOptions[2],  min_peak_bytes,
      kOptions[3],  min_residual_bytes,
      kOptions[4],  min_output_bytes,
      kOptions[5],  min_micros,
      kOptions[6],  min_accelerator_micros,
      kOptions[7],  min_cpu_micros,
      kOptions[8],  min_params,
      kOptions[9],  min_float_ops,
      kOptions[10], min_occurrence,
      kOptions[11], step,
      kOptions[12], order_by,
      kOptions[13], absl::StrJoin(account_type_regexes, ","),
      kOptions[14], absl::StrJoin(start_name_regexes, ","),
      kOptions[15], absl::StrJoin(trim_name_regexes, ","),
      kOptions[16], absl::StrJoin(show_name_regexes, ","),
      kOptions[17], absl::StrJoin(hide_name_regexes, ","),
      kOptions[18], (account_displayed_op_only ? "true" : "false"),
      kOptions[19], absl::StrJoin(select, ","),
      kOptions[20], output_type, KeyValueToStr(output_options));
  // clang-format on
  return s;
}

}  // namespace tfprof
}  // namespace tensorflow
