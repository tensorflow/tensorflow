/*
 * Copyright 2015 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flatbuffers/reflection.h"
#include "flatbuffers/util.h"

// Helper functionality for reflection.

namespace flatbuffers {

int64_t GetAnyValueI(reflection::BaseType type, const uint8_t *data) {
  // clang-format off
  #define FLATBUFFERS_GET(T) static_cast<int64_t>(ReadScalar<T>(data))
  switch (type) {
    case reflection::UType:
    case reflection::Bool:
    case reflection::UByte:  return FLATBUFFERS_GET(uint8_t);
    case reflection::Byte:   return FLATBUFFERS_GET(int8_t);
    case reflection::Short:  return FLATBUFFERS_GET(int16_t);
    case reflection::UShort: return FLATBUFFERS_GET(uint16_t);
    case reflection::Int:    return FLATBUFFERS_GET(int32_t);
    case reflection::UInt:   return FLATBUFFERS_GET(uint32_t);
    case reflection::Long:   return FLATBUFFERS_GET(int64_t);
    case reflection::ULong:  return FLATBUFFERS_GET(uint64_t);
    case reflection::Float:  return FLATBUFFERS_GET(float);
    case reflection::Double: return FLATBUFFERS_GET(double);
    case reflection::String: {
      auto s = reinterpret_cast<const String *>(ReadScalar<uoffset_t>(data) +
                                                data);
      return s ? StringToInt(s->c_str()) : 0;
    }
    default: return 0;  // Tables & vectors do not make sense.
  }
  #undef FLATBUFFERS_GET
  // clang-format on
}

double GetAnyValueF(reflection::BaseType type, const uint8_t *data) {
  switch (type) {
    case reflection::Float: return static_cast<double>(ReadScalar<float>(data));
    case reflection::Double: return ReadScalar<double>(data);
    case reflection::String: {
      auto s =
          reinterpret_cast<const String *>(ReadScalar<uoffset_t>(data) + data);
      return s ? strtod(s->c_str(), nullptr) : 0.0;
    }
    default: return static_cast<double>(GetAnyValueI(type, data));
  }
}

std::string GetAnyValueS(reflection::BaseType type, const uint8_t *data,
                         const reflection::Schema *schema, int type_index) {
  switch (type) {
    case reflection::Float:
    case reflection::Double: return NumToString(GetAnyValueF(type, data));
    case reflection::String: {
      auto s =
          reinterpret_cast<const String *>(ReadScalar<uoffset_t>(data) + data);
      return s ? s->c_str() : "";
    }
    case reflection::Obj:
      if (schema) {
        // Convert the table to a string. This is mostly for debugging purposes,
        // and does NOT promise to be JSON compliant.
        // Also prefixes the type.
        auto &objectdef = *schema->objects()->Get(type_index);
        auto s = objectdef.name()->str();
        if (objectdef.is_struct()) {
          s += "(struct)";  // TODO: implement this as well.
        } else {
          auto table_field = reinterpret_cast<const Table *>(
              ReadScalar<uoffset_t>(data) + data);
          s += " { ";
          auto fielddefs = objectdef.fields();
          for (auto it = fielddefs->begin(); it != fielddefs->end(); ++it) {
            auto &fielddef = **it;
            if (!table_field->CheckField(fielddef.offset())) continue;
            auto val = GetAnyFieldS(*table_field, fielddef, schema);
            if (fielddef.type()->base_type() == reflection::String) {
              std::string esc;
              flatbuffers::EscapeString(val.c_str(), val.length(), &esc, true,
                                        false);
              val = esc;
            }
            s += fielddef.name()->str();
            s += ": ";
            s += val;
            s += ", ";
          }
          s += "}";
        }
        return s;
      } else {
        return "(table)";
      }
    case reflection::Vector:
      return "[(elements)]";                   // TODO: implement this as well.
    case reflection::Union: return "(union)";  // TODO: implement this as well.
    default: return NumToString(GetAnyValueI(type, data));
  }
}

void SetAnyValueI(reflection::BaseType type, uint8_t *data, int64_t val) {
  // clang-format off
  #define FLATBUFFERS_SET(T) WriteScalar(data, static_cast<T>(val))
  switch (type) {
    case reflection::UType:
    case reflection::Bool:
    case reflection::UByte:  FLATBUFFERS_SET(uint8_t ); break;
    case reflection::Byte:   FLATBUFFERS_SET(int8_t  ); break;
    case reflection::Short:  FLATBUFFERS_SET(int16_t ); break;
    case reflection::UShort: FLATBUFFERS_SET(uint16_t); break;
    case reflection::Int:    FLATBUFFERS_SET(int32_t ); break;
    case reflection::UInt:   FLATBUFFERS_SET(uint32_t); break;
    case reflection::Long:   FLATBUFFERS_SET(int64_t ); break;
    case reflection::ULong:  FLATBUFFERS_SET(uint64_t); break;
    case reflection::Float:  FLATBUFFERS_SET(float   ); break;
    case reflection::Double: FLATBUFFERS_SET(double  ); break;
    // TODO: support strings
    default: break;
  }
  #undef FLATBUFFERS_SET
  // clang-format on
}

void SetAnyValueF(reflection::BaseType type, uint8_t *data, double val) {
  switch (type) {
    case reflection::Float: WriteScalar(data, static_cast<float>(val)); break;
    case reflection::Double: WriteScalar(data, val); break;
    // TODO: support strings.
    default: SetAnyValueI(type, data, static_cast<int64_t>(val)); break;
  }
}

void SetAnyValueS(reflection::BaseType type, uint8_t *data, const char *val) {
  switch (type) {
    case reflection::Float:
    case reflection::Double:
      SetAnyValueF(type, data, strtod(val, nullptr));
      break;
    // TODO: support strings.
    default: SetAnyValueI(type, data, StringToInt(val)); break;
  }
}

// Resize a FlatBuffer in-place by iterating through all offsets in the buffer
// and adjusting them by "delta" if they straddle the start offset.
// Once that is done, bytes can now be inserted/deleted safely.
// "delta" may be negative (shrinking).
// Unless "delta" is a multiple of the largest alignment, you'll create a small
// amount of garbage space in the buffer (usually 0..7 bytes).
// If your FlatBuffer's root table is not the schema's root table, you should
// pass in your root_table type as well.
class ResizeContext {
 public:
  ResizeContext(const reflection::Schema &schema, uoffset_t start, int delta,
                std::vector<uint8_t> *flatbuf,
                const reflection::Object *root_table = nullptr)
      : schema_(schema),
        startptr_(vector_data(*flatbuf) + start),
        delta_(delta),
        buf_(*flatbuf),
        dag_check_(flatbuf->size() / sizeof(uoffset_t), false) {
    auto mask = static_cast<int>(sizeof(largest_scalar_t) - 1);
    delta_ = (delta_ + mask) & ~mask;
    if (!delta_) return;  // We can't shrink by less than largest_scalar_t.
    // Now change all the offsets by delta_.
    auto root = GetAnyRoot(vector_data(buf_));
    Straddle<uoffset_t, 1>(vector_data(buf_), root, vector_data(buf_));
    ResizeTable(root_table ? *root_table : *schema.root_table(), root);
    // We can now add or remove bytes at start.
    if (delta_ > 0)
      buf_.insert(buf_.begin() + start, delta_, 0);
    else
      buf_.erase(buf_.begin() + start, buf_.begin() + start - delta_);
  }

  // Check if the range between first (lower address) and second straddles
  // the insertion point. If it does, change the offset at offsetloc (of
  // type T, with direction D).
  template<typename T, int D>
  void Straddle(const void *first, const void *second, void *offsetloc) {
    if (first <= startptr_ && second >= startptr_) {
      WriteScalar<T>(offsetloc, ReadScalar<T>(offsetloc) + delta_ * D);
      DagCheck(offsetloc) = true;
    }
  }

  // This returns a boolean that records if the corresponding offset location
  // has been modified already. If so, we can't even read the corresponding
  // offset, since it is pointing to a location that is illegal until the
  // resize actually happens.
  // This must be checked for every offset, since we can't know which offsets
  // will straddle and which won't.
  uint8_t &DagCheck(const void *offsetloc) {
    auto dag_idx = reinterpret_cast<const uoffset_t *>(offsetloc) -
                   reinterpret_cast<const uoffset_t *>(vector_data(buf_));
    return dag_check_[dag_idx];
  }

  void ResizeTable(const reflection::Object &objectdef, Table *table) {
    if (DagCheck(table)) return;  // Table already visited.
    auto vtable = table->GetVTable();
    // Early out: since all fields inside the table must point forwards in
    // memory, if the insertion point is before the table we can stop here.
    auto tableloc = reinterpret_cast<uint8_t *>(table);
    if (startptr_ <= tableloc) {
      // Check if insertion point is between the table and a vtable that
      // precedes it. This can't happen in current construction code, but check
      // just in case we ever change the way flatbuffers are built.
      Straddle<soffset_t, -1>(vtable, table, table);
    } else {
      // Check each field.
      auto fielddefs = objectdef.fields();
      for (auto it = fielddefs->begin(); it != fielddefs->end(); ++it) {
        auto &fielddef = **it;
        auto base_type = fielddef.type()->base_type();
        // Ignore scalars.
        if (base_type <= reflection::Double) continue;
        // Ignore fields that are not stored.
        auto offset = table->GetOptionalFieldOffset(fielddef.offset());
        if (!offset) continue;
        // Ignore structs.
        auto subobjectdef =
            base_type == reflection::Obj
                ? schema_.objects()->Get(fielddef.type()->index())
                : nullptr;
        if (subobjectdef && subobjectdef->is_struct()) continue;
        // Get this fields' offset, and read it if safe.
        auto offsetloc = tableloc + offset;
        if (DagCheck(offsetloc)) continue;  // This offset already visited.
        auto ref = offsetloc + ReadScalar<uoffset_t>(offsetloc);
        Straddle<uoffset_t, 1>(offsetloc, ref, offsetloc);
        // Recurse.
        switch (base_type) {
          case reflection::Obj: {
            ResizeTable(*subobjectdef, reinterpret_cast<Table *>(ref));
            break;
          }
          case reflection::Vector: {
            auto elem_type = fielddef.type()->element();
            if (elem_type != reflection::Obj && elem_type != reflection::String)
              break;
            auto vec = reinterpret_cast<Vector<uoffset_t> *>(ref);
            auto elemobjectdef =
                elem_type == reflection::Obj
                    ? schema_.objects()->Get(fielddef.type()->index())
                    : nullptr;
            if (elemobjectdef && elemobjectdef->is_struct()) break;
            for (uoffset_t i = 0; i < vec->size(); i++) {
              auto loc = vec->Data() + i * sizeof(uoffset_t);
              if (DagCheck(loc)) continue;  // This offset already visited.
              auto dest = loc + vec->Get(i);
              Straddle<uoffset_t, 1>(loc, dest, loc);
              if (elemobjectdef)
                ResizeTable(*elemobjectdef, reinterpret_cast<Table *>(dest));
            }
            break;
          }
          case reflection::Union: {
            ResizeTable(GetUnionType(schema_, objectdef, fielddef, *table),
                        reinterpret_cast<Table *>(ref));
            break;
          }
          case reflection::String: break;
          default: FLATBUFFERS_ASSERT(false);
        }
      }
      // Check if the vtable offset points beyond the insertion point.
      // Must do this last, since GetOptionalFieldOffset above still reads
      // this value.
      Straddle<soffset_t, -1>(table, vtable, table);
    }
  }

  void operator=(const ResizeContext &rc);

 private:
  const reflection::Schema &schema_;
  uint8_t *startptr_;
  int delta_;
  std::vector<uint8_t> &buf_;
  std::vector<uint8_t> dag_check_;
};

void SetString(const reflection::Schema &schema, const std::string &val,
               const String *str, std::vector<uint8_t> *flatbuf,
               const reflection::Object *root_table) {
  auto delta = static_cast<int>(val.size()) - static_cast<int>(str->size());
  auto str_start = static_cast<uoffset_t>(
      reinterpret_cast<const uint8_t *>(str) - vector_data(*flatbuf));
  auto start = str_start + static_cast<uoffset_t>(sizeof(uoffset_t));
  if (delta) {
    // Clear the old string, since we don't want parts of it remaining.
    memset(vector_data(*flatbuf) + start, 0, str->size());
    // Different size, we must expand (or contract).
    ResizeContext(schema, start, delta, flatbuf, root_table);
    // Set the new length.
    WriteScalar(vector_data(*flatbuf) + str_start,
                static_cast<uoffset_t>(val.size()));
  }
  // Copy new data. Safe because we created the right amount of space.
  memcpy(vector_data(*flatbuf) + start, val.c_str(), val.size() + 1);
}

uint8_t *ResizeAnyVector(const reflection::Schema &schema, uoffset_t newsize,
                         const VectorOfAny *vec, uoffset_t num_elems,
                         uoffset_t elem_size, std::vector<uint8_t> *flatbuf,
                         const reflection::Object *root_table) {
  auto delta_elem = static_cast<int>(newsize) - static_cast<int>(num_elems);
  auto delta_bytes = delta_elem * static_cast<int>(elem_size);
  auto vec_start =
      reinterpret_cast<const uint8_t *>(vec) - vector_data(*flatbuf);
  auto start = static_cast<uoffset_t>(vec_start + sizeof(uoffset_t) +
                                      elem_size * num_elems);
  if (delta_bytes) {
    if (delta_elem < 0) {
      // Clear elements we're throwing away, since some might remain in the
      // buffer.
      auto size_clear = -delta_elem * elem_size;
      memset(vector_data(*flatbuf) + start - size_clear, 0, size_clear);
    }
    ResizeContext(schema, start, delta_bytes, flatbuf, root_table);
    WriteScalar(vector_data(*flatbuf) + vec_start, newsize);  // Length field.
    // Set new elements to 0.. this can be overwritten by the caller.
    if (delta_elem > 0) {
      memset(vector_data(*flatbuf) + start, 0, delta_elem * elem_size);
    }
  }
  return vector_data(*flatbuf) + start;
}

const uint8_t *AddFlatBuffer(std::vector<uint8_t> &flatbuf,
                             const uint8_t *newbuf, size_t newlen) {
  // Align to sizeof(uoffset_t) past sizeof(largest_scalar_t) since we're
  // going to chop off the root offset.
  while ((flatbuf.size() & (sizeof(uoffset_t) - 1)) ||
         !(flatbuf.size() & (sizeof(largest_scalar_t) - 1))) {
    flatbuf.push_back(0);
  }
  auto insertion_point = static_cast<uoffset_t>(flatbuf.size());
  // Insert the entire FlatBuffer minus the root pointer.
  flatbuf.insert(flatbuf.end(), newbuf + sizeof(uoffset_t), newbuf + newlen);
  auto root_offset = ReadScalar<uoffset_t>(newbuf) - sizeof(uoffset_t);
  return vector_data(flatbuf) + insertion_point + root_offset;
}

void CopyInline(FlatBufferBuilder &fbb, const reflection::Field &fielddef,
                const Table &table, size_t align, size_t size) {
  fbb.Align(align);
  fbb.PushBytes(table.GetStruct<const uint8_t *>(fielddef.offset()), size);
  fbb.TrackField(fielddef.offset(), fbb.GetSize());
}

Offset<const Table *> CopyTable(FlatBufferBuilder &fbb,
                                const reflection::Schema &schema,
                                const reflection::Object &objectdef,
                                const Table &table, bool use_string_pooling) {
  // Before we can construct the table, we have to first generate any
  // subobjects, and collect their offsets.
  std::vector<uoffset_t> offsets;
  auto fielddefs = objectdef.fields();
  for (auto it = fielddefs->begin(); it != fielddefs->end(); ++it) {
    auto &fielddef = **it;
    // Skip if field is not present in the source.
    if (!table.CheckField(fielddef.offset())) continue;
    uoffset_t offset = 0;
    switch (fielddef.type()->base_type()) {
      case reflection::String: {
        offset = use_string_pooling
                     ? fbb.CreateSharedString(GetFieldS(table, fielddef)).o
                     : fbb.CreateString(GetFieldS(table, fielddef)).o;
        break;
      }
      case reflection::Obj: {
        auto &subobjectdef = *schema.objects()->Get(fielddef.type()->index());
        if (!subobjectdef.is_struct()) {
          offset =
              CopyTable(fbb, schema, subobjectdef, *GetFieldT(table, fielddef))
                  .o;
        }
        break;
      }
      case reflection::Union: {
        auto &subobjectdef = GetUnionType(schema, objectdef, fielddef, table);
        offset =
            CopyTable(fbb, schema, subobjectdef, *GetFieldT(table, fielddef)).o;
        break;
      }
      case reflection::Vector: {
        auto vec =
            table.GetPointer<const Vector<Offset<Table>> *>(fielddef.offset());
        auto element_base_type = fielddef.type()->element();
        auto elemobjectdef =
            element_base_type == reflection::Obj
                ? schema.objects()->Get(fielddef.type()->index())
                : nullptr;
        switch (element_base_type) {
          case reflection::String: {
            std::vector<Offset<const String *>> elements(vec->size());
            auto vec_s = reinterpret_cast<const Vector<Offset<String>> *>(vec);
            for (uoffset_t i = 0; i < vec_s->size(); i++) {
              elements[i] = use_string_pooling
                                ? fbb.CreateSharedString(vec_s->Get(i)).o
                                : fbb.CreateString(vec_s->Get(i)).o;
            }
            offset = fbb.CreateVector(elements).o;
            break;
          }
          case reflection::Obj: {
            if (!elemobjectdef->is_struct()) {
              std::vector<Offset<const Table *>> elements(vec->size());
              for (uoffset_t i = 0; i < vec->size(); i++) {
                elements[i] =
                    CopyTable(fbb, schema, *elemobjectdef, *vec->Get(i));
              }
              offset = fbb.CreateVector(elements).o;
              break;
            }
          }
          FLATBUFFERS_FALLTHROUGH(); // fall thru
          default: {  // Scalars and structs.
            auto element_size = GetTypeSize(element_base_type);
            if (elemobjectdef && elemobjectdef->is_struct())
              element_size = elemobjectdef->bytesize();
            fbb.StartVector(vec->size(), element_size);
            fbb.PushBytes(vec->Data(), element_size * vec->size());
            offset = fbb.EndVector(vec->size());
            break;
          }
        }
        break;
      }
      default:  // Scalars.
        break;
    }
    if (offset) { offsets.push_back(offset); }
  }
  // Now we can build the actual table from either offsets or scalar data.
  auto start = objectdef.is_struct() ? fbb.StartStruct(objectdef.minalign())
                                     : fbb.StartTable();
  size_t offset_idx = 0;
  for (auto it = fielddefs->begin(); it != fielddefs->end(); ++it) {
    auto &fielddef = **it;
    if (!table.CheckField(fielddef.offset())) continue;
    auto base_type = fielddef.type()->base_type();
    switch (base_type) {
      case reflection::Obj: {
        auto &subobjectdef = *schema.objects()->Get(fielddef.type()->index());
        if (subobjectdef.is_struct()) {
          CopyInline(fbb, fielddef, table, subobjectdef.minalign(),
                     subobjectdef.bytesize());
          break;
        }
      }
      FLATBUFFERS_FALLTHROUGH(); // fall thru
      case reflection::Union:
      case reflection::String:
      case reflection::Vector:
        fbb.AddOffset(fielddef.offset(), Offset<void>(offsets[offset_idx++]));
        break;
      default: {  // Scalars.
        auto size = GetTypeSize(base_type);
        CopyInline(fbb, fielddef, table, size, size);
        break;
      }
    }
  }
  FLATBUFFERS_ASSERT(offset_idx == offsets.size());
  if (objectdef.is_struct()) {
    fbb.ClearOffsets();
    return fbb.EndStruct();
  } else {
    return fbb.EndTable(start);
  }
}

bool VerifyStruct(flatbuffers::Verifier &v,
                  const flatbuffers::Table &parent_table,
                  voffset_t field_offset, const reflection::Object &obj,
                  bool required) {
  auto offset = parent_table.GetOptionalFieldOffset(field_offset);
  if (required && !offset) { return false; }

  return !offset ||
         v.Verify(reinterpret_cast<const uint8_t *>(&parent_table), offset,
                  obj.bytesize());
}

bool VerifyVectorOfStructs(flatbuffers::Verifier &v,
                           const flatbuffers::Table &parent_table,
                           voffset_t field_offset,
                           const reflection::Object &obj, bool required) {
  auto p = parent_table.GetPointer<const uint8_t *>(field_offset);
  if (required && !p) { return false; }

  return !p || v.VerifyVectorOrString(p, obj.bytesize());
}

// forward declare to resolve cyclic deps between VerifyObject and VerifyVector
bool VerifyObject(flatbuffers::Verifier &v, const reflection::Schema &schema,
                  const reflection::Object &obj,
                  const flatbuffers::Table *table, bool required);

bool VerifyVector(flatbuffers::Verifier &v, const reflection::Schema &schema,
                  const flatbuffers::Table &table,
                  const reflection::Field &vec_field) {
  FLATBUFFERS_ASSERT(vec_field.type()->base_type() == reflection::Vector);
  if (!table.VerifyField<uoffset_t>(v, vec_field.offset())) return false;

  switch (vec_field.type()->element()) {
    case reflection::None: FLATBUFFERS_ASSERT(false); break;
    case reflection::UType:
      return v.VerifyVector(flatbuffers::GetFieldV<uint8_t>(table, vec_field));
    case reflection::Bool:
    case reflection::Byte:
    case reflection::UByte:
      return v.VerifyVector(flatbuffers::GetFieldV<int8_t>(table, vec_field));
    case reflection::Short:
    case reflection::UShort:
      return v.VerifyVector(flatbuffers::GetFieldV<int16_t>(table, vec_field));
    case reflection::Int:
    case reflection::UInt:
      return v.VerifyVector(flatbuffers::GetFieldV<int32_t>(table, vec_field));
    case reflection::Long:
    case reflection::ULong:
      return v.VerifyVector(flatbuffers::GetFieldV<int64_t>(table, vec_field));
    case reflection::Float:
      return v.VerifyVector(flatbuffers::GetFieldV<float>(table, vec_field));
    case reflection::Double:
      return v.VerifyVector(flatbuffers::GetFieldV<double>(table, vec_field));
    case reflection::String: {
      auto vec_string =
          flatbuffers::GetFieldV<flatbuffers::Offset<flatbuffers::String>>(
              table, vec_field);
      if (v.VerifyVector(vec_string) && v.VerifyVectorOfStrings(vec_string)) {
        return true;
      } else {
        return false;
      }
    }
    case reflection::Vector: FLATBUFFERS_ASSERT(false); break;
    case reflection::Obj: {
      auto obj = schema.objects()->Get(vec_field.type()->index());
      if (obj->is_struct()) {
        if (!VerifyVectorOfStructs(v, table, vec_field.offset(), *obj,
                                   vec_field.required())) {
          return false;
        }
      } else {
        auto vec =
            flatbuffers::GetFieldV<flatbuffers::Offset<flatbuffers::Table>>(
                table, vec_field);
        if (!v.VerifyVector(vec)) return false;
        if (vec) {
          for (uoffset_t j = 0; j < vec->size(); j++) {
            if (!VerifyObject(v, schema, *obj, vec->Get(j), true)) {
              return false;
            }
          }
        }
      }
      return true;
    }
    case reflection::Union: FLATBUFFERS_ASSERT(false); break;
    default: FLATBUFFERS_ASSERT(false); break;
  }

  return false;
}

bool VerifyObject(flatbuffers::Verifier &v, const reflection::Schema &schema,
                  const reflection::Object &obj,
                  const flatbuffers::Table *table, bool required) {
  if (!table) {
    if (!required)
      return true;
    else
      return false;
  }

  if (!table->VerifyTableStart(v)) return false;

  for (uoffset_t i = 0; i < obj.fields()->size(); i++) {
    auto field_def = obj.fields()->Get(i);
    switch (field_def->type()->base_type()) {
      case reflection::None: FLATBUFFERS_ASSERT(false); break;
      case reflection::UType:
        if (!table->VerifyField<uint8_t>(v, field_def->offset())) return false;
        break;
      case reflection::Bool:
      case reflection::Byte:
      case reflection::UByte:
        if (!table->VerifyField<int8_t>(v, field_def->offset())) return false;
        break;
      case reflection::Short:
      case reflection::UShort:
        if (!table->VerifyField<int16_t>(v, field_def->offset())) return false;
        break;
      case reflection::Int:
      case reflection::UInt:
        if (!table->VerifyField<int32_t>(v, field_def->offset())) return false;
        break;
      case reflection::Long:
      case reflection::ULong:
        if (!table->VerifyField<int64_t>(v, field_def->offset())) return false;
        break;
      case reflection::Float:
        if (!table->VerifyField<float>(v, field_def->offset())) return false;
        break;
      case reflection::Double:
        if (!table->VerifyField<double>(v, field_def->offset())) return false;
        break;
      case reflection::String:
        if (!table->VerifyField<uoffset_t>(v, field_def->offset()) ||
            !v.VerifyString(flatbuffers::GetFieldS(*table, *field_def))) {
          return false;
        }
        break;
      case reflection::Vector:
        if (!VerifyVector(v, schema, *table, *field_def)) return false;
        break;
      case reflection::Obj: {
        auto child_obj = schema.objects()->Get(field_def->type()->index());
        if (child_obj->is_struct()) {
          if (!VerifyStruct(v, *table, field_def->offset(), *child_obj,
                            field_def->required())) {
            return false;
          }
        } else {
          if (!VerifyObject(v, schema, *child_obj,
                            flatbuffers::GetFieldT(*table, *field_def),
                            field_def->required())) {
            return false;
          }
        }
        break;
      }
      case reflection::Union: {
        //  get union type from the prev field
        voffset_t utype_offset = field_def->offset() - sizeof(voffset_t);
        auto utype = table->GetField<uint8_t>(utype_offset, 0);
        if (utype != 0) {
          // Means we have this union field present
          auto fb_enum = schema.enums()->Get(field_def->type()->index());
          if (utype >= fb_enum->values()->size()) return false;
          auto child_obj = fb_enum->values()->Get(utype)->object();
          if (!VerifyObject(v, schema, *child_obj,
                            flatbuffers::GetFieldT(*table, *field_def),
                            field_def->required())) {
            return false;
          }
        }
        break;
      }
      default: FLATBUFFERS_ASSERT(false); break;
    }
  }

  if (!v.EndTable()) return false;

  return true;
}

bool Verify(const reflection::Schema &schema, const reflection::Object &root,
            const uint8_t *buf, size_t length) {
  Verifier v(buf, length);
  return VerifyObject(v, schema, root, flatbuffers::GetAnyRoot(buf), true);
}

}  // namespace flatbuffers
