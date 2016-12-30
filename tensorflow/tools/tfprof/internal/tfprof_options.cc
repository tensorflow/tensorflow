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

#include "tensorflow/tools/tfprof/internal/tfprof_options.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/tools/tfprof/tfprof_options.pb.h"

namespace tensorflow {
namespace tfprof {

Options Options::FromProtoStr(const string& opts_proto_str) {
  OptionsProto opts_pb;
  CHECK(opts_pb.ParseFromString(opts_proto_str));
  Options opts(
      opts_pb.max_depth(), opts_pb.min_bytes(), opts_pb.min_micros(),
      opts_pb.min_params(), opts_pb.min_float_ops(),
      std::vector<string>(opts_pb.device_regexes().begin(),
                          opts_pb.device_regexes().end()),
      opts_pb.order_by(),
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
      opts_pb.viz(), opts_pb.dump_to_file());
  return opts;
}

string Options::ToString() const {
  const string s = strings::Printf(
      "%-28s%d\n"
      "%-28s%lld\n"
      "%-28s%lld\n"
      "%-28s%lld\n"
      "%-28s%lld\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n"
      "%-28s%s\n",
      kOptions[0], max_depth, kOptions[1], min_bytes, kOptions[2], min_micros,
      kOptions[3], min_params, kOptions[4], min_float_ops, kOptions[5],
      str_util::Join(device_regexes, ",").c_str(), kOptions[6],
      order_by.c_str(), kOptions[7],
      str_util::Join(account_type_regexes, ",").c_str(), kOptions[8],
      str_util::Join(start_name_regexes, ",").c_str(), kOptions[9],
      str_util::Join(trim_name_regexes, ",").c_str(), kOptions[10],
      str_util::Join(show_name_regexes, ",").c_str(), kOptions[11],
      str_util::Join(hide_name_regexes, ",").c_str(), kOptions[12],
      (account_displayed_op_only ? "true" : "false"), kOptions[13],
      str_util::Join(select, ",").c_str(), kOptions[14],
      (viz ? "true" : "false"), kOptions[15], dump_to_file.c_str());
  return s;
}

}  // namespace tfprof
}  // namespace tensorflow
