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

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_options.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace tfprof {

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
