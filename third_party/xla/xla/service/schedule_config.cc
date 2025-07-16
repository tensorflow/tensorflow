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

#include "xla/service/schedule_config.h"

#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

ScheduleConfigProto ScheduleConfig::ToProto(const ScheduleConfig& config) {
  ScheduleConfigProto schedule_config_proto;
  for (const auto& instruction : config.schedule) {
    schedule_config_proto.add_sequence()->set_name(instruction);
  }
  return schedule_config_proto;
}

ScheduleConfig ScheduleConfig::FromProto(const ScheduleConfigProto& proto) {
  ScheduleConfig config;
  for (const auto& instruction : proto.sequence()) {
    config.schedule.push_back(instruction.name());
  }
  return config;
}

}  // namespace xla
