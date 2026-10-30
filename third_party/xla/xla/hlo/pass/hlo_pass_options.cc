/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/pass/hlo_pass_options.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

HloPassOptions::HloPassOptions(std::initializer_list<Option> options) {
  for (const Option& option : options) {
    CHECK(options_.try_emplace(option.name, option.value).second)
        << "Duplicate pass option '" << option.name << "'";
  }
}

HloPassOptions::HloPassOptions(const tsl::protobuf::Message& message) {
  const tsl::protobuf::Reflection* reflection = message.GetReflection();
  std::vector<const tsl::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const tsl::protobuf::FieldDescriptor* field : fields) {
    CHECK(!field->is_repeated()) << "Repeated field '" << field->full_name()
                                 << "' is not representable as a pass option";
    HloPassOptionValue& value = options_[field->name()];
    switch (field->cpp_type()) {
      case tsl::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        value.set_bool_val(reflection->GetBool(message, field));
        break;
      case tsl::protobuf::FieldDescriptor::CPPTYPE_INT64:
        value.set_int_val(reflection->GetInt64(message, field));
        break;
      default:
        LOG(FATAL) << "Field '" << field->full_name() << "' of type "
                   << field->cpp_type_name()
                   << " is not representable as a pass option";
    }
  }
}

namespace {

absl::string_view TypeName(const HloPassOptionValue& value) {
  switch (value.value_case()) {
    case HloPassOptionValue::kBoolVal:
      return "bool";
    case HloPassOptionValue::kIntVal:
      return "int64";
    case HloPassOptionValue::VALUE_NOT_SET:
      return "unset";
  }
}

}  // namespace

absl::Status HloPassOptions::CheckAllConsumed() const {
  // Order only affects which of several unconsumed options gets reported.
  // NOLINTNEXTLINE(google*-custom-deterministic-iteration-order)
  for (const auto& [key, value] : options_) {
    if (!consumed_keys_.contains(key)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Pass option '", key, "' (", TypeName(value),
                       ") was not consumed: unknown name or wrong type"));
    }
  }
  return absl::OkStatus();
}

HloPassOptions::ProtoMap HloPassOptions::ToProto() const {
  return ProtoMap(options_.begin(), options_.end());
}

void HloPassOptions::MergeInto(tsl::protobuf::Message& message) const {
  const tsl::protobuf::Descriptor* descriptor = message.GetDescriptor();
  const tsl::protobuf::Reflection* reflection = message.GetReflection();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const tsl::protobuf::FieldDescriptor* field = descriptor->field(i);
    if (field->is_repeated()) {
      continue;
    }
    switch (field->cpp_type()) {
      case tsl::protobuf::FieldDescriptor::CPPTYPE_BOOL:
        if (std::optional<bool> value = Get<bool>(field->name())) {
          reflection->SetBool(&message, field, *value);
        }
        break;
      case tsl::protobuf::FieldDescriptor::CPPTYPE_INT64:
        if (std::optional<int64_t> value = Get<int64_t>(field->name())) {
          reflection->SetInt64(&message, field, *value);
        }
        break;
      default:
        // Not representable as a pass option; a same-named option stays
        // unconsumed.
        break;
    }
  }
}

}  // namespace xla
