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

#include "include/json/json.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.pb.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

#include <stdlib.h>
#include <sstream>
#include <string>

namespace xla {
namespace poplarplugin {

std::string PoplibsLibToString(const PoplibsLib& poplibs_lib) {
  static std::vector<std::string> names = {"Poplin", "Popnn", "Popops",
                                           "Poprand"};
  if (names.size() != static_cast<uint32>(PoplibsLib::_NumLibs)) {
    LOG(FATAL) << "The number of Poplibs libraries does not match.";
  }
  return names[static_cast<uint32>(poplibs_lib)];
}

StatusOr<PoplibsLib> StringToPoplibsLib(const std::string& name) {
  static absl::flat_hash_map<std::string, PoplibsLib> mapping = {
      {"Poplin", PoplibsLib::Poplin},
      {"Popnn", PoplibsLib::Popnn},
      {"Popops", PoplibsLib::Popops},
      {"Poprand", PoplibsLib::Poprand}};
  if (mapping.size() != static_cast<uint32>(PoplibsLib::_NumLibs)) {
    LOG(FATAL) << "The number of Poplibs libraries does not match.";
  }
  return mapping.at(name);
}

std::string PoplibsOpToString(const PoplibsOp& poplibs_op) {
  static std::vector<std::string> names = {"LstmLayerFwd", "LstmLayerBwd",
                                           "Sqrt", "Rsqrt"};
  if (names.size() != static_cast<uint32>(PoplibsOp::_NumOps)) {
    LOG(FATAL) << "The number of Poplibs Custom Ops does not match.";
  }
  return names[static_cast<uint32>(poplibs_op)];
}

StatusOr<PoplibsOp> StringToPoplibsOp(const std::string& name) {
  static absl::flat_hash_map<std::string, PoplibsOp> mapping = {
      // Poplin:
      // Popnn:
      {"LstmLayerFwd", PoplibsOp::LstmLayerFwd},
      {"LstmLayerBwd", PoplibsOp::LstmLayerBwd},
      // Popops:
      {"Sqrt", PoplibsOp::Sqrt},
      {"Rsqrt", PoplibsOp::Rsqrt},
      // Poprand:
  };
  if (mapping.size() != static_cast<uint32>(PoplibsOp::_NumOps)) {
    LOG(FATAL) << "The number of Poplibs Custom Ops does not match.";
  }
  return mapping.at(name);
}

std::string GetPoplibsCustomOpTargetString(const PoplibsLib& poplibs_lib,
                                           const PoplibsOp& poplibs_op) {
  return PoplibsLibToString(poplibs_lib) + "::" + PoplibsOpToString(poplibs_op);
}

StatusOr<std::pair<PoplibsLib, PoplibsOp>> GetPoplibsCustomOp(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kCustomCall) {
    std::vector<std::string> split =
        absl::StrSplit(inst->custom_call_target(), "::");
    if (split.size() != 2) {
      return xla::FailedPrecondition("Call target %s is not a PoplibsCustomOp.",
                                     inst->custom_call_target().c_str());
    }
    auto statusor_poplibs_lib = StringToPoplibsLib(split[0]);
    if (!statusor_poplibs_lib.ok()) {
      return xla::FailedPrecondition(
          "Call target %s is not a custom call to a Poplibs library.",
          inst->custom_call_target().c_str());
    }
    auto statusor_poplibs_op = StringToPoplibsOp(split[1]);
    if (!statusor_poplibs_op.ok()) {
      return xla::FailedPrecondition(
          "Call target %s is not a custom call to a Poplibs op.",
          inst->custom_call_target().c_str());
    }
    return std::make_pair(statusor_poplibs_lib.ValueOrDie(),
                          statusor_poplibs_op.ValueOrDie());
  }
  return xla::FailedPrecondition(
      "Instruction %s is not a HloInstruction::kCustomCall.",
      inst->name().c_str());
}

const bool IsPoplibsCustomOp(const HloInstruction* inst) {
  return GetPoplibsCustomOp(inst).ok();
}

const bool IsPoplibsCustomOpElementwise(const HloInstruction* inst) {
  auto statusor = GetPoplibsCustomOp(inst);
  if (!statusor.ok()) {
    return false;
  }
  PoplibsLib poplibs_lib;
  PoplibsOp poplibs_op;
  std::tie(poplibs_lib, poplibs_op) = statusor.ValueOrDie();
  switch (poplibs_lib) {
    case PoplibsLib::Popops: {
      switch (poplibs_op) {
        case PoplibsOp::Sqrt:
        case PoplibsOp::Rsqrt: {
          return true;
        }
        default: { return false; }
      }
      break;
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

template <typename T>
void AttributeMap::AddAttribute(const std::string& field_name, const T& attr) {
  attributes_[field_name] = attr;
}

template <>
void AttributeMap::AddAttribute(const std::string& field_name,
                                const tensorflow::DataType& attr) {
  attributes_[field_name] = DataType_Name(attr);
}

template <>
void AttributeMap::AddAttribute(const std::string& field_name,
                                const absl::flat_hash_set<int64>& attr) {
  attributes_[field_name] = absl::StrJoin(attr, ",");
}

template <>
void AttributeMap::AddAttribute(const std::string& field_name,
                                const uint64& attr) {
  attributes_[field_name] = Json::Value::UInt64(attr);
}

template void AttributeMap::AddAttribute<float>(const std::string&,
                                                const float&);
template void AttributeMap::AddAttribute<int>(const std::string&, const int&);
template void AttributeMap::AddAttribute<uint64>(const std::string&,
                                                 const uint64&);
template void AttributeMap::AddAttribute<bool>(const std::string&, const bool&);
template void AttributeMap::AddAttribute<std::string>(const std::string&,
                                                      const std::string&);
template void AttributeMap::AddAttribute<tensorflow::DataType>(
    const std::string&, const tensorflow::DataType&);
template void AttributeMap::AddAttribute<absl::flat_hash_set<int64>>(
    const std::string&, const absl::flat_hash_set<int64>&);

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

StatusOr<absl::flat_hash_set<int64>>
AttributeMap::GetAttributeAsInt64FlatHashSet(
    const std::string& field_name) const {
  if (!attributes_.isMember(field_name)) {
    return xla::FailedPrecondition(
        "Could not obtain the field %s for the custom op.", field_name.c_str());
  }
  const std::string result_string = attributes_[field_name].asString();
  absl::flat_hash_set<int64> result;
  std::vector<std::string> split =
      absl::StrSplit(result_string, ",", absl::SkipEmpty());
  for (auto num_string : split) {
    result.insert(std::stoll(num_string));
  }
  return result;
}

const std::string AttributeMap::Serialise() {
  Json::FastWriter fastWriter;
  return fastWriter.write(attributes_);
}

}  // namespace IPUCustomKernelsUtil
}  // namespace poplarplugin
}  // namespace xla
