/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

#include "include/json/json.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/human_readable_json.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/any.h"

#include <stdlib.h>
#include <sstream>
#include <string>

namespace xla {
namespace poplarplugin {
std::string GetPoplibsCustomOpTargetString(PoplibsOp::Lib lib,
                                           PoplibsOp::Op op) {
  return PoplibsOp_Lib_Name(lib) + "::" + PoplibsOp_Op_Name(op);
}

absl::optional<std::pair<PoplibsOp::Lib, PoplibsOp::Op>> GetPoplibsCustomOp(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kCustomCall) {
    std::vector<std::string> split =
        absl::StrSplit(inst->custom_call_target(), "::");
    if (split.size() != 2) {
      return absl::nullopt;
    }
    PoplibsOp::Lib lib;
    bool lib_parsed = PoplibsOp_Lib_Parse(split[0], &lib);
    if (!lib_parsed) {
      return absl::nullopt;
    }

    PoplibsOp::Op op;
    bool op_parsed = PoplibsOp_Op_Parse(split[1], &op);
    if (!op_parsed) {
      return absl::nullopt;
    }
    return std::make_pair(lib, op);
  }
  return absl::nullopt;
}

const bool IsPoplibsCustomOp(const HloInstruction* inst) {
  return GetPoplibsCustomOp(inst) != absl::nullopt;
}

const bool IsPoplibsCustomOp(const HloInstruction* inst, PoplibsOp::Lib lib,
                             PoplibsOp::Op op) {
  auto ret = GetPoplibsCustomOp(inst);
  if (!ret) {
    return false;
  }
  return ret->first == lib && ret->second == op;
}

const bool IsPoplibsCustomOpElementwise(const HloInstruction* inst) {
  if (!IsPoplibsCustomOp(inst)) {
    return false;
  }
  auto ret = GetPoplibsCustomOp(inst);
  if (!ret) {
    return false;
  }

  switch (ret->first) {
    case PoplibsOp::Popops: {
      switch (ret->second) {
        default: { return false; }
      }
      break;
    }
    case PoplibsOp::Poprand: {
      switch (ret->second) {
        case PoplibsOp::TruncatedNormal: {
          return true;
        }
        default: { return false; }
      }
    }
    default: { return false; }
  }
}

namespace IPUCustomKernelsUtil {

AttributeMap::AttributeMap() {}
AttributeMap::AttributeMap(const HloInstruction* custom_call) {
  CHECK_EQ(custom_call->opcode(), HloOpcode::kCustomCall);

  std::string attributes_json =
      static_cast<const HloCustomCallInstruction*>(custom_call)->opaque();

  Json::Reader reader;
  bool parsed = reader.parse(attributes_json.c_str(), attributes_);
  if (!parsed) {
    LOG(FATAL) << "Could not parse the call target for custom op as JSON "
               << attributes_json;
  }
}

namespace {
template <typename T>
Json::Value GetAsJsonValue(const T& val) {
  return Json::Value(val);
}
template <>
Json::Value GetAsJsonValue(const tensorflow::DataType& val) {
  return Json::Value(DataType_Name(val));
}
template <>
Json::Value GetAsJsonValue(const int64& val) {
  return Json::Value(Json::Value::Int64(val));
}
template <>
Json::Value GetAsJsonValue(const uint64& val) {
  return Json::Value(Json::Value::UInt64(val));
}
}  // namespace

void AttributeMap::AddAttribute(const std::string& field_name,
                                const absl::any& attr) {
  const std::type_info& tinfo = attr.type();
  if (tinfo == typeid(float)) {
    auto casted_val = absl::any_cast<float>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(int)) {
    auto casted_val = absl::any_cast<int>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(bool)) {
    auto casted_val = absl::any_cast<bool>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(uint64)) {
    auto casted_val = absl::any_cast<uint64>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(int64)) {
    auto casted_val = absl::any_cast<int64>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(tensorflow::DataType)) {
    auto casted_val = absl::any_cast<tensorflow::DataType>(attr);
    attributes_[field_name] = GetAsJsonValue(casted_val);

  } else if (tinfo == typeid(absl::flat_hash_set<int64>)) {
    auto casted_vals = absl::any_cast<absl::flat_hash_set<int64>>(attr);
    // Always create the field.
    auto& values = attributes_[field_name];
    values = Json::arrayValue;
    for (auto val : casted_vals) {
      values.append(GetAsJsonValue(val));
    }

  } else if (tinfo == typeid(absl::flat_hash_map<int64, int64>)) {
    auto casted_vals = absl::any_cast<absl::flat_hash_map<int64, int64>>(attr);

    auto& keys = attributes_[field_name]["keys"];
    auto& values = attributes_[field_name]["values"];
    keys = Json::arrayValue;
    values = Json::arrayValue;
    for (auto pair : casted_vals) {
      keys.append(GetAsJsonValue(pair.first));
      values.append(GetAsJsonValue(pair.second));
    }
  } else if (tinfo == typeid(Window)) {
    auto casted_val = absl::any_cast<Window>(attr);
    std::string window_proto_str;
    if (!tensorflow::ProtoToHumanReadableJson(casted_val, &window_proto_str)
             .ok()) {
      LOG(FATAL) << "Could not parse the window.";
    }
    attributes_[field_name] = GetAsJsonValue(window_proto_str);
  } else {
    LOG(FATAL) << "Unsupported attribute value type " << tinfo.name();
  }
}

StatusOr<std::string> AttributeMap::GetAttributeAsString(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  return attributes_[field_name].asString();
}

StatusOr<float> AttributeMap::GetAttributeAsFloat(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  return attributes_[field_name].asFloat();
}

StatusOr<int> AttributeMap::GetAttributeAsInt(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  return attributes_[field_name].asInt();
}

StatusOr<uint64> AttributeMap::GetAttributeAsUInt64(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  return attributes_[field_name].asUInt64();
}

StatusOr<bool> AttributeMap::GetAttributeAsBool(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  return attributes_[field_name].asBool();
}

StatusOr<tensorflow::DataType> AttributeMap::GetAttributeAsTFDataType(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  const std::string dtype_string = attributes_[field_name].asString();
  tensorflow::DataType data_type;
  if (!DataType_Parse(dtype_string, &data_type)) {
    return xla::FailedPrecondition("Could not parse the DataType %s.",
                                   dtype_string.c_str());
  }
  return data_type;
}

StatusOr<absl::flat_hash_set<int64>> AttributeMap::GetAttributeFlatHashSet(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name) || !attributes_[field_name].isArray()) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  absl::flat_hash_set<int64> result;
  for (auto val : attributes_[field_name]) {
    result.insert(val.asInt64());
  }
  return result;
}

StatusOr<absl::flat_hash_map<int64, int64>>
AttributeMap::GetAttributeFlatHashMap(const std::string& field_name) const {
  if (!attributes_.isMember(field_name) ||
      !attributes_[field_name].isMember("keys") ||
      !attributes_[field_name].isMember("values")) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  auto keys = attributes_[field_name]["keys"];
  auto values = attributes_[field_name]["values"];
  if (keys.size() != values.size()) {
    return xla::FailedPrecondition("Corrupted hash map %s for the custom op.",
                                   field_name.c_str());
  }
  absl::flat_hash_map<int64, int64> result;
  for (int i = 0; i < keys.size(); i++) {
    int64 key = keys[i].asInt64();
    int64 value = values[i].asInt64();
    result[key] = value;
  }
  return result;
}

StatusOr<Window> AttributeMap::GetAttributeAsWindow(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  std::string window_proto_str = attributes_[field_name].asString();
  Window window;
  TF_RETURN_IF_ERROR(
      tensorflow::HumanReadableJsonToProto(window_proto_str, &window));
  return window;
}

const std::string AttributeMap::Serialise() {
  Json::FastWriter fastWriter;
  return fastWriter.write(attributes_);
}

}  // namespace IPUCustomKernelsUtil
}  // namespace poplarplugin
}  // namespace xla
