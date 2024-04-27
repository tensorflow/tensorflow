/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/tools/proto_splitter/cc/util.h"

#include <cstdint>
#include <functional>
#include <ios>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "riegeli/base/maker.h"  // from @riegeli
#include "riegeli/base/types.h"  // from @riegeli
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {

using ::tensorflow::proto_splitter::ChunkedField;

namespace {
absl::StatusOr<int> FieldInt(const FieldType& field) {
  switch (field.index()) {
    case 0: {  // std::string
      int value;
      if (absl::SimpleAtoi(std::get<std::string>(field), &value)) return value;
      return absl::InvalidArgumentError(absl::StrCat(
          "Unable to convert '", std::get<std::string>(field), "' to int."));
      break;
    }
    case 1:  // int
      return std::get<int>(field);
    case 2:  // bool
      return std::get<bool>(field) ? 1 : 0;
    default:
      // This should not happen.
      return absl::InvalidArgumentError("Invalid field type.");
  }
}
absl::StatusOr<bool> FieldBool(const FieldType& field) {
  switch (field.index()) {
    case 0: {  // std::string
      std::string s = std::get<std::string>(field);
      if (s == "true") return true;
      if (s == "false") return false;
      return absl::InvalidArgumentError("Unable to convert '" + s +
                                        "' to bool.");
    }
    case 1:  // int
      return std::get<int>(field) != 0;
    case 2:  // bool
      return std::get<bool>(field);
    default:
      // This should not happen.
      return absl::InvalidArgumentError("Invalid field type.");
  }
}
std::string FieldString(const FieldType& field) {
  switch (field.index()) {
    case 0:  // std::string
      return std::get<std::string>(field);
    case 1:  // int
      return std::to_string(std::get<int>(field));
    case 2:  // bool
      return std::get<bool>(field) ? "true" : "false";
    default:
      // This should not happen.
      CHECK(false) << "Invalid field type: " << field.index();  // Crash OK.
      return "";
  }
}

template <typename FieldCallback, typename MapKeyCallback,
          typename ListIndexCallback>
absl::Status WalkFields(const tsl::protobuf::Descriptor& desc,
                        const std::vector<FieldType>& fields,
                        FieldCallback field_callback,
                        MapKeyCallback map_key_callback,
                        ListIndexCallback list_index_callback) {
  const tsl::protobuf::Descriptor* message_desc = &desc;
  std::vector<FieldType>::const_iterator it = fields.begin();

  while (it != fields.end()) {
    const tsl::protobuf::FieldDescriptor* field_desc;
    if (std::holds_alternative<std::string>(*it)) {
      field_desc = message_desc->FindFieldByName(std::get<std::string>(*it));
    } else if (std::holds_alternative<int>(*it)) {
      field_desc = message_desc->FindFieldByNumber(std::get<int>(*it));
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid field, expected int or str: ", FieldString(*it)));
    }

    if (field_desc == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("Field not found: ", FieldString(*it)));
    }
    TF_RETURN_IF_ERROR(field_callback(field_desc));
    message_desc = field_desc->message_type();

    it++;
    // Handle special field types (map key and list index).
    if (it != fields.end()) {
      if (field_desc->is_map()) {
        auto key_field = message_desc->FindFieldByNumber(1);
        auto status = map_key_callback(*key_field, *it);
        if (!status.ok()) {
          return status;
        }
        // The next element in `fields` must be a field in the value message.
        message_desc = message_desc->FindFieldByName("value")->message_type();
        it++;
      } else if (field_desc->is_repeated()) {
        auto status = list_index_callback(*it);
        if (!status.ok()) {
          return status;
        }
        it++;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status AddMapKey(const tsl::protobuf::FieldDescriptor& key_field,
                       const FieldType& map_key, ChunkedField& chunked_field) {
  auto msg = chunked_field.add_field_tag()->mutable_map_key();
  switch (key_field.type()) {
    case tsl::protobuf::FieldDescriptor::TYPE_BOOL: {
      TF_ASSIGN_OR_RETURN(auto key, FieldBool(map_key));
      msg->set_boolean(key);
      break;
    }
    case tsl::protobuf::FieldDescriptor::TYPE_STRING: {
      msg->set_s(FieldString(map_key));
      break;
    }
    case tsl::protobuf::FieldDescriptor::TYPE_INT32: {
      TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
      msg->set_i32(key);
      break;
    }
    case tsl::protobuf::FieldDescriptor::TYPE_INT64: {
      TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
      msg->set_i64(key);
      break;
    }
    case tsl::protobuf::FieldDescriptor::TYPE_UINT32: {
      TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
      msg->set_ui32(key);
      break;
    }
    case tsl::protobuf::FieldDescriptor::TYPE_UINT64: {
      TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
      msg->set_ui64(key);
      break;
    }
    case tsl::protobuf::FieldDescriptor::TYPE_FIXED64:
    case tsl::protobuf::FieldDescriptor::TYPE_FIXED32:
    case tsl::protobuf::FieldDescriptor::TYPE_SINT32:
    case tsl::protobuf::FieldDescriptor::TYPE_SINT64:
    case tsl::protobuf::FieldDescriptor::TYPE_SFIXED32:
    case tsl::protobuf::FieldDescriptor::TYPE_SFIXED64:
      return absl::UnimplementedError(
          "Fixed map key types not implemented yet.");
    case tsl::protobuf::FieldDescriptor::TYPE_DOUBLE:
    case tsl::protobuf::FieldDescriptor::TYPE_FLOAT:
    case tsl::protobuf::FieldDescriptor::TYPE_ENUM:
    case tsl::protobuf::FieldDescriptor::TYPE_BYTES:
    case tsl::protobuf::FieldDescriptor::TYPE_MESSAGE:
    case tsl::protobuf::FieldDescriptor::TYPE_GROUP:
      return absl::FailedPreconditionError(absl::StrCat(
          "Encountered type that is not supported for MapKey.type: ",
          key_field.type()));

    default:
      return absl::FailedPreconditionError("Proto field type not recognized.");
  }
  return absl::OkStatus();
}

absl::StatusOr<FieldType> GetMapKeyFromFieldIndex(
    ::tensorflow::proto_splitter::FieldIndex field_index) {
  if (!field_index.has_map_key())
    return absl::FailedPreconditionError(
        "Field index doesn't contain a map key.");

  switch (field_index.map_key().type_case()) {
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kBoolean:
      return field_index.map_key().boolean();
      break;
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kS:
      return field_index.map_key().s();
      break;
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kI32:
      return field_index.map_key().i32();
      break;
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kI64:
      // Cast to int type, which may be lossy. We'll deal with it when it
      // becomes an issue.
      return static_cast<int>(field_index.map_key().i64());
      break;
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kUi32:
      return static_cast<int>(field_index.map_key().ui32());
      break;
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::kUi64:
      return static_cast<int>(field_index.map_key().ui64());
      break;
    case ::tensorflow::proto_splitter::FieldIndex::MapKey::TypeCase::
        TYPE_NOT_SET:
    default:
      return absl::FailedPreconditionError(
          absl::StrCat("Unknown map key type: ", field_index.DebugString()));
  }
}

}  // namespace

absl::StatusOr<const std::vector<Field>> GetFieldTypes(
    const tsl::protobuf::RepeatedPtrField<
        ::tensorflow::proto_splitter::FieldIndex>& field_tags) {
  std::vector<Field> fields;
  for (int fti = 0; fti < field_tags.size();) {
    switch (field_tags[fti].kind_case()) {
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::kField:
        fields.push_back(
            Field(static_cast<int>(field_tags[fti].field()), std::nullopt));
        fti++;
        if (fti == field_tags.size()) break;
        // Multiple field tags may correspond to a single field when the field
        // is repeated or a map.
        if (field_tags[fti].has_index()) {
          fields.back().second = static_cast<int>(field_tags[fti++].index());
        } else if (field_tags[fti].has_map_key()) {
          TF_ASSIGN_OR_RETURN(const FieldType& map_key,
                              GetMapKeyFromFieldIndex(field_tags[fti++]));
          fields.back().second = map_key;
        }
        break;
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::kIndex:
        return absl::FailedPreconditionError(
            "Index doesn't belong to any field.");
        break;
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::kMapKey:
        return absl::FailedPreconditionError(
            "Map key doesn't belong to any field.");
        break;
      case ::tensorflow::proto_splitter::FieldIndex::KindCase::KIND_NOT_SET:
      default:
        return absl::FailedPreconditionError(absl::StrCat(
            "Unknown field kind: ", field_tags[fti].DebugString()));
    }
  }
  return fields;
}

absl::Status SetRepeatedFieldElement(
    tsl::protobuf::Message* message,
    const tsl::protobuf::FieldDescriptor* field_desc, uint64_t field_index,
    const std::string& chunk,
    std::function<absl::Status(void)> message_callback) {
  if (field_desc->is_map())
    return absl::FailedPreconditionError("Field is a map.");
  const tsl::protobuf::Reflection* reflection = message->GetReflection();
  if (field_index >= reflection->FieldSize(*message, field_desc))
    return absl::OutOfRangeError(
        absl::StrCat("Field index out of range: ", field_index));

  switch (field_desc->cpp_type()) {
    case tsl::protobuf::FieldDescriptor::CPPTYPE_INT32:
      reflection->SetRepeatedInt32(message, field_desc, std::stoi(chunk),
                                   field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_INT64:
      reflection->SetRepeatedInt64(message, field_desc, std::stoll(chunk),
                                   field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT32:
      reflection->SetRepeatedUInt32(message, field_desc, std::stoul(chunk),
                                    field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT64:
      reflection->SetRepeatedUInt64(message, field_desc, std::stoull(chunk),
                                    field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetRepeatedDouble(message, field_desc, std::stod(chunk),
                                    field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      reflection->SetRepeatedFloat(message, field_desc, std::stof(chunk),
                                   field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_BOOL:
      bool b;
      std::istringstream(chunk) >> std::boolalpha >> b;
      reflection->SetRepeatedBool(message, field_desc, b, field_index);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      reflection->SetRepeatedEnum(
          message, field_desc, field_index,
          field_desc->enum_type()->FindValueByName(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_STRING:
      reflection->SetRepeatedString(message, field_desc, field_index, chunk);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
      return message_callback();
      break;
    default:
      return absl::FailedPreconditionError(absl::StrCat(
          "Proto field type not recognized: ", field_desc->cpp_type_name()));
      break;
  }
  return absl::OkStatus();
}

absl::Status SetFieldElement(
    tsl::protobuf::Message* message,
    const tsl::protobuf::FieldDescriptor* field_desc, const std::string& chunk,
    std::function<absl::Status(void)> message_callback) {
  const tsl::protobuf::Reflection* reflection = message->GetReflection();

  switch (field_desc->cpp_type()) {
    case tsl::protobuf::FieldDescriptor::CPPTYPE_INT32:
      reflection->SetInt32(message, field_desc, std::stoi(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_INT64:
      reflection->SetInt64(message, field_desc, std::stoll(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT32:
      reflection->SetUInt32(message, field_desc, std::stoul(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT64:
      reflection->SetUInt64(message, field_desc, std::stoull(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetDouble(message, field_desc, std::stod(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      reflection->SetFloat(message, field_desc, std::stof(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_BOOL:
      bool b;
      std::istringstream(chunk) >> std::boolalpha >> b;
      reflection->SetBool(message, field_desc, b);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      reflection->SetEnum(message, field_desc,
                          field_desc->enum_type()->FindValueByName(chunk));
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_STRING:
      reflection->SetString(message, field_desc, chunk);
      break;
    case tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
      return message_callback();
      break;
    default:
      return absl::FailedPreconditionError("Proto field type not recognized.");
      break;
  }
  return absl::OkStatus();
}

absl::Status AddMapEntry(tsl::protobuf::Message* message,
                         const tsl::protobuf::FieldDescriptor* field_desc,
                         FieldType map_key) {
  tsl::protobuf::Message* map_entry =
      message->GetReflection()->AddMessage(message, field_desc);
  const tsl::protobuf::Reflection* reflection = map_entry->GetReflection();
  const tsl::protobuf::FieldDescriptor* key =
      field_desc->message_type()->FindFieldByNumber(1);
  switch (key->cpp_type()) {
    case tsl::protobuf::FieldDescriptor::CPPTYPE_INT32: {
      TF_ASSIGN_OR_RETURN(auto map_key_int32, FieldInt(map_key));
      reflection->SetInt32(map_entry, key, map_key_int32);
      break;
    }
    case tsl::protobuf::FieldDescriptor::CPPTYPE_INT64: {
      TF_ASSIGN_OR_RETURN(auto map_key_int64, FieldInt(map_key));
      reflection->SetInt64(map_entry, key, map_key_int64);
      break;
    }
    case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT32: {
      TF_ASSIGN_OR_RETURN(auto map_key_uint32, FieldInt(map_key));
      reflection->SetUInt32(map_entry, key, map_key_uint32);
      break;
    }
    case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT64: {
      TF_ASSIGN_OR_RETURN(auto map_key_uint64, FieldInt(map_key));
      reflection->SetUInt64(map_entry, key, map_key_uint64);
      break;
    }
    case tsl::protobuf::FieldDescriptor::CPPTYPE_BOOL: {
      TF_ASSIGN_OR_RETURN(auto map_key_bool, FieldBool(map_key));
      reflection->SetBool(map_entry, key, map_key_bool);
      break;
    }
    case tsl::protobuf::FieldDescriptor::CPPTYPE_STRING: {
      reflection->SetString(map_entry, key, FieldString(map_key));
      break;
    }
    default:
      return absl::FailedPreconditionError(absl::StrCat(
          "Proto field type not recognized.", key->cpp_type_name()));
  }
  return absl::OkStatus();
}

absl::StatusOr<MutableFieldResult> GetMutableField(
    tsl::protobuf::Message* message, const std::vector<FieldType>& fields) {
  tsl::protobuf::Message* parent = message;
  const tsl::protobuf::FieldDescriptor* field = nullptr;
  int index = -1;

  auto field_callback =
      [&parent, &field, &index](
          const tsl::protobuf::FieldDescriptor* field_desc) -> absl::Status {
    const tsl::protobuf::Reflection* reflection = parent->GetReflection();
    if (field != nullptr) {
      // Update the parent proto using the previous FieldDescriptor.
      if (field->is_map()) {
        parent = reflection->MutableRepeatedMessage(parent, field, index);
        parent = parent->GetReflection()->MutableMessage(
            parent, parent->GetDescriptor()->FindFieldByNumber(2));
        index = -1;
      } else if (field->is_repeated()) {
        parent = reflection->MutableRepeatedMessage(parent, field, index);
        index = -1;
      } else {
        parent = reflection->MutableMessage(parent, field);
      }
    }

    field = field_desc;
    return absl::OkStatus();
  };

  auto map_key_callback = [&parent, &field, &index](
                              const tsl::protobuf::FieldDescriptor& key_field,
                              const FieldType& map_key) -> absl::Status {
    // Finds the index of the map field w/ the corresponding map_key.
    const tsl::protobuf::Reflection* reflection = parent->GetReflection();

    TF_ASSIGN_OR_RETURN(int found_index,
                        FindMapKey(*parent, *field, &key_field, map_key));

    if (found_index == -1) {
      return absl::NotFoundError(
          absl::StrCat("Error when getting map field, couldn't find key: ",
                       FieldString(map_key)));
    }
    if (reflection->FieldSize(*parent, field) <= found_index) {
      return absl::NotFoundError(absl::StrCat(
          "Can't access index ", index, " in field '", field->name(),
          "' (size = ", reflection->FieldSize(*parent, field), ")."));
    }
    index = found_index;
    return absl::OkStatus();
  };
  auto list_index_callback = [&parent, &field,
                              &index](FieldType list_index) -> absl::Status {
    TF_ASSIGN_OR_RETURN(index, FieldInt(list_index));
    const tsl::protobuf::Reflection* reflection = parent->GetReflection();
    if (reflection->FieldSize(*parent, field) <= index) {
      return absl::NotFoundError(absl::StrCat(
          "Can't access index ", index, " in field '", field->name(),
          "' (size = ", reflection->FieldSize(*parent, field), ")."));
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(WalkFields(*message->GetDescriptor(), fields,
                                field_callback, map_key_callback,
                                list_index_callback));

  MutableFieldResult result{.parent = parent, .field = field, .index = index};
  return result;
}

absl::StatusOr<MutableFieldResult> GetMutableField(
    tsl::protobuf::Message* message, const Field& field) {
  std::vector<FieldType> fields = {field.first};
  if (field.second != std::nullopt) fields.push_back(field.second.value());
  return GetMutableField(message, fields);
}

absl::StatusOr<MutableFieldResult> GetMutableField(
    tsl::protobuf::Message* message, const FieldType& field_type) {
  std::vector<FieldType> fields = {field_type};
  return GetMutableField(message, fields);
}

absl::StatusOr<FieldResult> GetField(const tsl::protobuf::Message& message,
                                     const std::vector<FieldType>& fields) {
  const tsl::protobuf::Message* parent = &message;
  const tsl::protobuf::FieldDescriptor* field = nullptr;
  int index = -1;

  auto field_callback =
      [&parent, &field, &index](
          const tsl::protobuf::FieldDescriptor* field_desc) -> absl::Status {
    const tsl::protobuf::Reflection* reflection = parent->GetReflection();
    if (field != nullptr) {
      // Update the parent proto using the previous FieldDescriptor.
      if (field->is_map()) {
        parent = &reflection->GetRepeatedMessage(*parent, field, index);
        parent = &parent->GetReflection()->GetMessage(
            *parent, parent->GetDescriptor()->FindFieldByNumber(2));
        index = -1;
      } else if (field->is_repeated()) {
        parent = &reflection->GetRepeatedMessage(*parent, field, index);
        index = -1;
      } else {
        parent = &reflection->GetMessage(*parent, field);
      }
    }

    field = field_desc;
    return absl::OkStatus();
  };

  auto map_key_callback = [&parent, &field, &index](
                              const tsl::protobuf::FieldDescriptor& key_field,
                              const FieldType& map_key) -> absl::Status {
    // Finds the index of the map field w/ the corresponding map_key.
    const tsl::protobuf::Reflection* reflection = parent->GetReflection();

    TF_ASSIGN_OR_RETURN(int found_index,
                        FindMapKey(*parent, *field, &key_field, map_key));

    if (found_index == -1) {
      return absl::NotFoundError(
          absl::StrCat("Error when getting map field, couldn't find key: ",
                       FieldString(map_key)));
    }
    if (reflection->FieldSize(*parent, field) <= found_index) {
      return absl::NotFoundError(absl::StrCat(
          "Can't access index ", index, " in field '", field->name(),
          "' (size = ", reflection->FieldSize(*parent, field), ")."));
    }
    index = found_index;
    return absl::OkStatus();
  };
  auto list_index_callback = [&parent, &field,
                              &index](FieldType list_index) -> absl::Status {
    TF_ASSIGN_OR_RETURN(index, FieldInt(list_index));
    const tsl::protobuf::Reflection* reflection = parent->GetReflection();
    if (reflection->FieldSize(*parent, field) <= index) {
      return absl::NotFoundError(absl::StrCat(
          "Can't access index ", index, " in field '", field->name(),
          "' (size = ", reflection->FieldSize(*parent, field), ")."));
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(WalkFields(*message.GetDescriptor(), fields,
                                field_callback, map_key_callback,
                                list_index_callback));

  FieldResult result{
      .parent = parent,
      .field = field,
      .index = index,
  };
  return result;
}

absl::Status AddFieldTag(const tsl::protobuf::Descriptor& desc,
                         const std::vector<FieldType>& fields,
                         ChunkedField& chunked_field) {
  auto field_callback =
      [&chunked_field](
          const tsl::protobuf::FieldDescriptor* field_desc) -> absl::Status {
    chunked_field.add_field_tag()->set_field(field_desc->number());
    return absl::OkStatus();
  };
  auto map_key_callback = [&chunked_field](
                              const tsl::protobuf::FieldDescriptor& key_field,
                              FieldType map_key) -> absl::Status {
    return AddMapKey(key_field, map_key, chunked_field);
  };
  auto list_index_callback =
      [&chunked_field](FieldType list_index) -> absl::Status {
    TF_ASSIGN_OR_RETURN(auto index, FieldInt(list_index));
    chunked_field.add_field_tag()->set_index(index);
    return absl::OkStatus();
  };

  return WalkFields(desc, fields, field_callback, map_key_callback,
                    list_index_callback);
}

absl::Status AddFieldTag(const tsl::protobuf::Descriptor& desc,
                         const Field& field, ChunkedField& chunked_field) {
  std::vector<FieldType> fields = {field.first};
  if (field.second != std::nullopt) fields.push_back(field.second.value());
  return AddFieldTag(desc, fields, chunked_field);
}

absl::StatusOr<int> FindMapKey(const tsl::protobuf::Message& parent,
                               const tsl::protobuf::FieldDescriptor& map_field,
                               const tsl::protobuf::FieldDescriptor* key_field,
                               FieldType map_key) {
  const tsl::protobuf::Reflection* reflection = parent.GetReflection();

  if (!map_field.is_map()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "FindMapKey() was given a non map field: ", map_field.full_name()));
  }

  if (key_field == nullptr) {
    key_field = map_field.message_type()->FindFieldByNumber(1);
  }

  bool found = false;
  for (int i = 0; i < reflection->FieldSize(parent, &map_field) && !found;
       i++) {
    auto kv_pair = &reflection->GetRepeatedMessage(parent, &map_field, i);
    auto kv_reflection = kv_pair->GetReflection();
    switch (key_field->cpp_type()) {
      case tsl::protobuf::FieldDescriptor::CPPTYPE_INT32: {
        TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
        if (kv_reflection->GetInt32(*kv_pair, key_field) == key) {
          found = true;
        }
        break;
      }
      case tsl::protobuf::FieldDescriptor::CPPTYPE_INT64: {
        TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
        if (kv_reflection->GetInt64(*kv_pair, key_field) == key) {
          found = true;
        }
        break;
      }
      case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT32: {
        TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
        if (kv_reflection->GetUInt32(*kv_pair, key_field) == key) {
          found = true;
        }
        break;
      }
      case tsl::protobuf::FieldDescriptor::CPPTYPE_UINT64: {
        TF_ASSIGN_OR_RETURN(auto key, FieldInt(map_key));
        if (kv_reflection->GetUInt64(*kv_pair, key_field) == key) {
          found = true;
        }
        break;
      }
      case tsl::protobuf::FieldDescriptor::CPPTYPE_BOOL: {
        TF_ASSIGN_OR_RETURN(auto key, FieldBool(map_key));
        if (kv_reflection->GetBool(*kv_pair, key_field) == key) {
          found = true;
        }
        break;
      }
      case tsl::protobuf::FieldDescriptor::CPPTYPE_STRING: {
        if (kv_reflection->GetString(*kv_pair, key_field) ==
            FieldString(map_key)) {
          found = true;
        }
        break;
      }
      case tsl::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      case tsl::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      case tsl::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      case tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
        return absl::FailedPreconditionError(absl::StrCat(
            "Type is not supported for MapKey.type: ", key_field->cpp_type()));
      default:
        return absl::FailedPreconditionError(
            "Proto field type not recognized.");
    }
    if (found) return i;
  }
  return -1;
}

namespace {

const std::vector<std::pair<double, std::string>>& ByteUnits() {
  static const auto& byte_units =
      *(new std::vector<std::pair<double, std::string>>(
          {{1, "B"}, {1 << 10, "KiB"}, {1 << 20, "MiB"}, {1 << 30, "GiB"}}));
  return byte_units;
}

}  // namespace

std::string HumanReadableBytes(int64_t byte_count) {
  for (int i = 1; i < ByteUnits().size(); i++) {
    if (byte_count < ByteUnits()[i].first) {
      return absl::StrFormat("%.1f%s", byte_count / ByteUnits()[i - 1].first,
                             ByteUnits()[i - 1].second);
    }
  }
  int i = ByteUnits().size() - 1;
  return absl::StrFormat("%.1f%s", byte_count / ByteUnits()[i].first,
                         ByteUnits()[i].second);
}

std::string HumanReadableDuration(int64_t microseconds) {
  if (microseconds < 1000) {
    return absl::StrFormat("%d microseconds", microseconds);
  } else if (microseconds < 1e6) {
    return absl::StrFormat("%.2f ms", microseconds / 1e3);
  } else {
    return absl::StrFormat("%.2f s", microseconds / 1e6);
  }
}

absl::StatusOr<riegeli::RecordReader<riegeli::FdReader<>>> GetRiegeliReader(
    absl::string_view cpb_file) {
  riegeli::RecordReader reader(riegeli::Maker<riegeli::FdReader>(cpb_file));
  if (!reader.ok()) {
    return reader.status();
  }
  return reader;
}

absl::StatusOr<::tensorflow::proto_splitter::ChunkMetadata> GetChunkMetadata(
    riegeli::RecordReaderBase& reader) {
  ::tensorflow::proto_splitter::ChunkMetadata chunk_metadata;
  bool read_metadata_success = reader.Seek(reader.Size().value()) &&
                               reader.SeekBack() &&
                               reader.ReadRecord(chunk_metadata);
  if (read_metadata_success) return chunk_metadata;
  return reader.status();
}

absl::StatusOr<std::string> ReadChunk(
    riegeli::RecordReaderBase& reader,
    const ::tensorflow::proto_splitter::ChunkInfo& chunk_info) {
  riegeli::Position pos = chunk_info.offset();
  std::string chunk(chunk_info.size(), '\0');
  if (reader.Seek(pos) && reader.ReadRecord(chunk)) return chunk;
  return absl::NotFoundError(
      absl::StrCat("Chunk could not be found at position: ", pos, ".\n",
                   reader.status().ToString()));
}

absl::StatusOr<bool> OnlyContainsPb(absl::string_view prefix) {
  const std::string pb_file = absl::StrCat(prefix, ".pb");
  const std::string cpb_file = absl::StrCat(prefix, ".cpb");
  TF_ASSIGN_OR_RETURN(bool is_pb,
                      internal::FileExists(Env::Default(), pb_file));
  TF_ASSIGN_OR_RETURN(bool is_cpb,
                      internal::FileExists(Env::Default(), cpb_file));
  if (is_pb && !is_cpb) {
    return true;
  } else if (!is_pb && !is_cpb) {
    return absl::NotFoundError(
        absl::StrCat("Could not find SavedModel .pb or .cpb at supplied "
                     "export directory path with prefix: ",
                     prefix,
                     ". Check that "
                     "the directory exists and that you have the right "
                     "permissions for accessing it."));
  }
  LOG(INFO) << "Reading chunked proto from " << cpb_file;
  return false;
}

}  // namespace tools::proto_splitter
}  // namespace tensorflow
