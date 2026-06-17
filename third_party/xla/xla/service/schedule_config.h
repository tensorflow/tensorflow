/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SCHEDULE_CONFIG_H_
#define XLA_SERVICE_SCHEDULE_CONFIG_H_

#include <string>
#include <vector>

#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Program's schedule configuration.
struct ScheduleConfig {
  std::vector<std::string> schedule;
  bool operator==(const ScheduleConfig& other) const {
    return schedule == other.schedule;
  }
  static ScheduleConfig FromProto(const ScheduleConfigProto& proto);
  static ScheduleConfigProto ToProto(const ScheduleConfig& config);
};

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULE_CONFIG_H_
