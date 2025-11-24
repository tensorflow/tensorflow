/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_nvtx_ext_payload.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExtPayload.h"

extern "C" CUptiResult CUPTIAPI cuptiActivityGetNvtxExtPayloadAttr(
    uint32_t cupti_domain_id, uint64_t schema_id,
    CUpti_NvtxExtPayloadAttr* payload_attributes);

extern "C" const nvtxPayloadEntryTypeInfo_t*
cuptiActivityGetNvtxExtPayloadEntryTypeInfo();

namespace xla {
namespace profiler {

namespace {

// Define the size as unknown or invalid. For unknown size, dynamic automatic
// detection may be used to determine the size. Invalid size indicates
// the size is not applicable or not supported.
static constexpr size_t kSizeUnknown = 0;
static constexpr size_t kSizeInvalid = std::numeric_limits<size_t>::max();

inline bool AreValidSizes(size_t a, size_t b) {
  return a != kSizeInvalid && b != kSizeInvalid;
}

inline bool IsFixedSize(size_t a) {
  return a != kSizeUnknown && a != kSizeInvalid;
}

inline bool AreFixedSizes(size_t a, size_t b) {
  return IsFixedSize(a) && IsFixedSize(b);
}

inline const char* NullCStringToEmpty(const char* str) {
  return str ? str : "";
}

// Define the attributes of NVTX payload schema or enum, which are similar to
// those maintained by NVTX and CUPTI.
struct NvtxPayloadAttributes {
  uint32_t payload_type = 0;  // CUPTI_NVTX_EXT_PAYLOAD_TYPE_ SCHEMA or ENUM
  uint32_t domain_id = 0;
  uint64_t schema_id = 0;
  std::string name = {};

  bool IsEnum() const {
    return payload_type == CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM;
  }
  bool IsSchema() const {
    return payload_type == CUPTI_NVTX_EXT_PAYLOAD_TYPE_SCHEMA;
  }
};

struct NvtxSchemaEntry {
  uint64_t flags = 0;            // currently only 0 is supported
  uint64_t type = 0;             // predefined type or custom schema ID
  std::string name = {};         // name of field
  std::string description = {};  // description of field
  uint64_t extent = 0;           // string or array length or union selector
  uint64_t offset = 0;           // offset in the structure (in bytes)
};

// Field of schema_type: NVTX_PAYLOAD_SCHEMA_TYPE_*, where * could be
// INVALID(0), STATIC, DYNAMIC, UNION, UNION_WITH_INTERNAL_SELECTOR.
// Currently only STATIC is supported.
struct NvtxPayloadSchema : public NvtxPayloadAttributes {
  uint64_t field_mask = 0;
  uint64_t schema_type = NVTX_PAYLOAD_SCHEMA_TYPE_INVALID;
  uint64_t flags = 0;
  uint64_t payload_static_size = 0;
  uint64_t pack_and_align = 0;
  std::vector<NvtxSchemaEntry> entries = {};
  std::atomic<bool> processed = false;

  NvtxPayloadSchema(uint32_t domain_id, uint64_t schema_id,
                    nvtxPayloadSchemaAttr_t& schema_attr);
};

struct NvtxEnumEntry {
  std::string name = {};
  uint64_t value = 0;
  int8_t is_flag = 0;
};

struct NvtxPayloadEnum : public NvtxPayloadAttributes {
  uint64_t field_mask = 0;
  uint64_t size_of_enum = 0;
  std::vector<NvtxEnumEntry> entries = {};

  NvtxPayloadEnum(uint32_t domain_id, uint64_t schema_id,
                  nvtxPayloadEnumAttr_t& enum_attr);
};

// Maps schema IDs to the corresponding NVTX payload attributes in single
// domain.
class NvtxSchemaIdToSchema {
  absl::Mutex mutex_ = {};
  uint32_t domain_id_ ABSL_GUARDED_BY(mutex_) = 0;
  absl::flat_hash_map<uint64_t, std::unique_ptr<NvtxPayloadAttributes>> schemas_
      ABSL_GUARDED_BY(mutex_) = {};

 public:
  explicit NvtxSchemaIdToSchema(uint32_t domain_id) : domain_id_(domain_id) {}

  // Get the NVTX payload attributes for the given schema ID. If the schema is
  // not found, query CUPTI for the attributes.
  NvtxPayloadAttributes* GetNvtxPayloadAttributes(uint64_t schema_id);
};

class NvtxDomainSchemas {
 public:
  // Get the per-domain NVTX payload schemas for given domain_id.
  static NvtxSchemaIdToSchema& ForDomain(uint32_t p_domain_id) {
    static NvtxDomainSchemas* singleton_instance = new NvtxDomainSchemas();
    return singleton_instance->GetNvtxDomainSchemas(p_domain_id);
  }

 private:
  absl::Mutex mutex_ = {};
  // Maps domain IDs to the corresponding per-domain NVTX payload schemas, note
  // that it is pointer stable for the value type NvtxSchemaIdToSchema.
  absl::flat_hash_map<uint32_t, std::unique_ptr<NvtxSchemaIdToSchema>>
      all_domains_schemas_ ABSL_GUARDED_BY(mutex_) = {};

  NvtxSchemaIdToSchema& GetNvtxDomainSchemas(uint32_t domain_id) {
    absl::MutexLock lock(mutex_);
    auto it = all_domains_schemas_.find(domain_id);
    if (it == all_domains_schemas_.end()) {
      it = all_domains_schemas_
               .insert({domain_id,
                        std::make_unique<NvtxSchemaIdToSchema>(domain_id)})
               .first;
    }
    return *(it->second);
  }
};

NvtxPayloadSchema::NvtxPayloadSchema(uint32_t p_domain_id, uint64_t p_schema_id,
                                     nvtxPayloadSchemaAttr_t& schema_attr)
    : NvtxPayloadAttributes{CUPTI_NVTX_EXT_PAYLOAD_TYPE_SCHEMA, p_domain_id,
                            p_schema_id, NullCStringToEmpty(schema_attr.name)},
      field_mask(schema_attr.fieldMask),
      schema_type(schema_attr.type),
      flags(schema_attr.flags),
      payload_static_size(schema_attr.payloadStaticSize),
      pack_and_align(schema_attr.packAlign),
      processed(false) {
  // Populate the schema's entries from the CUPTI-provided array.
  this->entries.reserve(schema_attr.numEntries);
  for (size_t i = 0; i < schema_attr.numEntries; ++i) {
    const nvtxPayloadSchemaEntry_t& entry = schema_attr.entries[i];
    this->entries.push_back(
        NvtxSchemaEntry{entry.flags, entry.type, NullCStringToEmpty(entry.name),
                        NullCStringToEmpty(entry.description),
                        entry.arrayOrUnionDetail, entry.offset});
  }
  if (schema_attr.entries != nullptr) {
    // Free the CUPTI-allocated payload entries array if it was allocated.
    free(const_cast<void*>(static_cast<const void*>(schema_attr.entries)));
  }
  schema_attr.entries = nullptr;  // Make sure it is marked as released.
}

NvtxPayloadEnum::NvtxPayloadEnum(uint32_t p_domain_id, uint64_t p_schema_id,
                                 nvtxPayloadEnumAttr_t& enum_attr)
    : NvtxPayloadAttributes{CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM, p_domain_id,
                            p_schema_id, NullCStringToEmpty(enum_attr.name)},
      field_mask(enum_attr.fieldMask),
      size_of_enum(enum_attr.sizeOfEnum) {
  this->entries.reserve(enum_attr.numEntries);
  for (size_t i = 0; i < enum_attr.numEntries; ++i) {
    const nvtxPayloadEnum_t& entry = enum_attr.entries[i];
    this->entries.push_back(
        NvtxEnumEntry{entry.name, entry.value, entry.isFlag});
  }
  if (enum_attr.entries != nullptr) {
    // Free the CUPTI-allocated payload entries array if it was allocated.
    free(const_cast<void*>(static_cast<const void*>(enum_attr.entries)));
  }
  enum_attr.entries = nullptr;  // Make sure it is marked as released.
}

NvtxPayloadAttributes* NvtxSchemaIdToSchema::GetNvtxPayloadAttributes(
    uint64_t schema_id) {
  absl::MutexLock lock(mutex_);
  auto schema = schemas_.find(schema_id);
  if (schema != schemas_.end()) {
    return schema->second.get();
  }

  // If the schema is not found, query CUPTI for the attributes.
  CUpti_NvtxExtPayloadAttr cupti_payload_attrs = {0};
  CUptiResult result = cuptiActivityGetNvtxExtPayloadAttr(domain_id_, schema_id,
                                                          &cupti_payload_attrs);
  if (result != CUPTI_SUCCESS) {
    VLOG(1) << "Could not get NVTX payload attributes from CUPTI for schema:"
            << schema_id << " in domain: " << domain_id_;
    return nullptr;
  }
  if (cupti_payload_attrs.attributes == nullptr) {
    VLOG(1) << "Payload schema/enum attribute is null from CUPTI for schema:"
            << schema_id << " in domain: " << domain_id_;
    return nullptr;
  }

  NvtxPayloadAttributes* attrs = nullptr;
  if (cupti_payload_attrs.type == CUPTI_NVTX_EXT_PAYLOAD_TYPE_SCHEMA) {
    auto [it, inserted] = schemas_.insert(
        {schema_id, std::make_unique<NvtxPayloadSchema>(
                        domain_id_, schema_id,
                        *reinterpret_cast<nvtxPayloadSchemaAttr_t*>(
                            cupti_payload_attrs.attributes))});
    attrs = it->second.get();
  } else if (cupti_payload_attrs.type == CUPTI_NVTX_EXT_PAYLOAD_TYPE_ENUM) {
    auto [it, inserted] = schemas_.insert(
        {schema_id, std::make_unique<NvtxPayloadEnum>(
                        domain_id_, schema_id,
                        *reinterpret_cast<nvtxPayloadEnumAttr_t*>(
                            cupti_payload_attrs.attributes))});
    attrs = it->second.get();
  }
  // Free the CUPTI-allocated payload attribute memory by above call to
  // cuptiActivityGetNvtxExtPayloadAttr().
  free(cupti_payload_attrs.attributes);

  return attrs;
}

struct PayloadSizeAndAlign {
  uint16_t size = 0;   // Size of the data type in bytes
  uint16_t align = 0;  // Alignment of the data type in bytes
};

// Get the singleton instance of the predefined payload types from CUPTI.
const std::vector<PayloadSizeAndAlign>& PredefinedPayloadTypes() {
  static std::vector<PayloadSizeAndAlign>* predefined_types = []() {
    auto* global_data = new std::vector<PayloadSizeAndAlign>();
    // Query CUPTI for the NVTX payload entry type information.
    const nvtxPayloadEntryTypeInfo_t* payload_type_info =
        cuptiActivityGetNvtxExtPayloadEntryTypeInfo();
    if (payload_type_info == nullptr) {
      LOG(ERROR) << ("Could not initialize NVTX predefined payload type!");
      return global_data;
    }

    // The first element in fact defines the length of the info array.
    global_data->reserve(payload_type_info->size);
    for (uint16_t i = 0; i < payload_type_info->size; ++i) {
      global_data->push_back(PayloadSizeAndAlign{
          payload_type_info[i].size,
          payload_type_info[i].align,
      });
    }
    VLOG(9) << "Initialized NVTX predefined payload type info with "
            << global_data->size() << " entries.";
    return global_data;
  }();
  return *predefined_types;
}

size_t GetSizeOfFixedSizeTypes(uint64_t type) {
  switch (type) {
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT16:
    case NVTX_PAYLOAD_ENTRY_TYPE_BF16:
    case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF16:
      return 2;
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT32:
    case NVTX_PAYLOAD_ENTRY_TYPE_TF32:
    case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32:
    case NVTX_PAYLOAD_ENTRY_TYPE_CATEGORY:
    case NVTX_PAYLOAD_ENTRY_TYPE_COLOR_ARGB:
    case NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT32:
    case NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32:
      return 4;
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT64:
    case NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT64:
    case NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT64:
    case NVTX_PAYLOAD_ENTRY_TYPE_SCOPE_ID:
      return 8;
    case NVTX_PAYLOAD_ENTRY_TYPE_INT128:
    case NVTX_PAYLOAD_ENTRY_TYPE_UINT128:
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT128:
      return 16;
    case NVTX_PAYLOAD_ENTRY_TYPE_BYTE:
    case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING:
    case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8:
    case NVTX_PAYLOAD_ENTRY_TYPE_UNION_SELECTOR:
      return 1;
    default:
      return kSizeInvalid;
  }
}

/**
 * @brief Returns the size (in bytes) of a predefined NVTX payload entry type.
 *
 * For standard NVTX types, size and alignment data is fetched from
 * g_nvtxData.nvtxPayloadDataTypes. For special cases (not in the standard
 * array), GetSizeOfFixedSizeTypes() is used. Handles special cases for
 * registered string handles and unknown types.
 *
 * @param type The NVTX payload entry type identifier.
 * @return The size in bytes of the type, or InvalidTypeSize (usually 0) if
 * unknown.
 */
size_t GetSizeOfPayloadPredefinedType(uint64_t type) {
  // If the type is within the range of the info array, use the global data
  // types vector.
  if (type < NVTX_PAYLOAD_ENTRY_TYPE_INFO_ARRAY_SIZE) {
    // Check if the type index is valid for the vector.
    if (type >= PredefinedPayloadTypes().size()) {
      VLOG(1) << "NVTX payload entry type:" << type
              << " is not found among pre-defined payload types.";
      return kSizeInvalid;
    }
    return PredefinedPayloadTypes()[type].size;
  }
  if (type < NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE) {
    // If the type is not in the info array, but is less than the string handle
    // type, use the fixed size types.
    return GetSizeOfFixedSizeTypes(type);
  }
  if (type == NVTX_PAYLOAD_ENTRY_TYPE_NVTX_REGISTERED_STRING_HANDLE &&
      NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS < PredefinedPayloadTypes().size()) {
    // If the type is the registered string handle, use the address type's size.
    return PredefinedPayloadTypes()[NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS].size;
  }
  return kSizeInvalid;
}

size_t AlignTo(size_t offset, size_t type_size, size_t alignment) {
  // The entry_offset is treated as a pointer for alignment calculation.
  void* addr_to_align = reinterpret_cast<void*>(offset);

  // The buffer size is not known, so use SIZE_MAX as a placeholder.
  size_t sz = SIZE_MAX;

  // Use std::align to compute the next aligned address.
  // NOTE: This requires C++17 or later.
  void* aligned_addr = std::align(alignment, type_size, addr_to_align, sz);
  return aligned_addr ? reinterpret_cast<size_t>(aligned_addr) : offset;
}

// Align the entry. entry_size and entry_align are valid size. If manual_offset
// is not zero, it must be less than orig_offset, otherwise the entry will be
// used as the starting offset of the entry. If manual_offset is zero, the
// orig_offset will be aligned to the next multiple of entry_align. Note that
// when entry_idx is zero, manual offset should be set to zero.
// Returns false if the manual_offset is not valid, otherwise true.
bool AlignEntryOffset(size_t& orig_offset, size_t manual_offset,
                      size_t entry_size, size_t entry_align, size_t entry_idx) {
  if (manual_offset != 0 && manual_offset < orig_offset) {
    return false;
  }
  orig_offset = manual_offset ? manual_offset
                              : AlignTo(orig_offset, entry_size, entry_align);
  return true;
}

// Declare in advance for recursion with UpdateSizeAndAlignForSchema().
std::pair<size_t, size_t> GetSizeAndAlign(NvtxSchemaIdToSchema& domain_schemas,
                                          const NvtxSchemaEntry& entry,
                                          int depth);

void UpdateSizeAndAlignForSchema(NvtxSchemaIdToSchema& domain_schemas,
                                 NvtxPayloadSchema& schema, int depth) {
  if (schema.processed) {
    return;
  }
  if (schema.schema_type != NVTX_PAYLOAD_SCHEMA_TYPE_STATIC) {
    schema.payload_static_size = kSizeInvalid;
  } else if (AreValidSizes(schema.pack_and_align, schema.payload_static_size)) {
    // Calculate size and alignment for the schema by iterating through its
    // entries. Verify or update the size and alignment of the whole schema.
    // It may recursively update for nested schemas.
    size_t schema_size = 0LL, schema_align = 0LL, entry_idx = 0LL;
    for (const NvtxSchemaEntry& entry : schema.entries) {
      auto [entry_size, entry_align] =
          GetSizeAndAlign(domain_schemas, entry, depth + 1);

      if (!AreFixedSizes(entry_size, entry_align) ||
          !AlignEntryOffset(schema_size, entry_idx ? entry.offset : 0,
                            entry_size, entry_align, entry_idx)) {
        schema.pack_and_align = kSizeInvalid;
        schema.payload_static_size = kSizeInvalid;
        break;
      }
      entry_idx++;

      // Increasing size and update alignment
      schema_size += entry_size;
      schema_align = std::max(schema_align, entry_align);
    }
    if (schema.pack_and_align == kSizeUnknown) {
      schema.pack_and_align = schema_align;
    }
    if (schema.payload_static_size == kSizeUnknown) {
      schema.payload_static_size = schema_size;
    } else if (schema.payload_static_size < schema_size) {
      schema.payload_static_size = kSizeInvalid;
    }
  }
  schema.processed = true;
}

std::pair<size_t, size_t> GetSizeAndAlign(NvtxSchemaIdToSchema& domain_schemas,
                                          const NvtxSchemaEntry& entry,
                                          int depth = 0) {
  if (entry.flags != 0) {  // No support for flags other than zero.
    return {kSizeInvalid, kSizeInvalid};
  }
  // Limit depth of nested schemas, also avoid circular reference.
  if (depth > 5) {
    VLOG(1) << "NVTX payload schema nested too deeply";
    return {kSizeInvalid, kSizeInvalid};
  }

  if (entry.type < NVTX_PAYLOAD_SCHEMA_ID_STATIC_START) {
    size_t type_size = GetSizeOfPayloadPredefinedType(entry.type);
    // Handle fixed size strings.
    bool use_extent = (type_size != kSizeInvalid &&
                       entry.type >= NVTX_PAYLOAD_ENTRY_TYPE_CSTRING &&
                       entry.type <= NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF32);
    return {type_size * (use_extent ? entry.extent : 1), type_size};
  }

  if (NvtxPayloadAttributes* payload_attributes =
          domain_schemas.GetNvtxPayloadAttributes(entry.type);
      payload_attributes != nullptr) {
    if (payload_attributes->IsEnum()) {
      auto& payload_enum = *static_cast<NvtxPayloadEnum*>(payload_attributes);
      return {payload_enum.size_of_enum, payload_enum.size_of_enum};
    }
    if (payload_attributes->IsSchema()) {
      auto& schema = *static_cast<NvtxPayloadSchema*>(payload_attributes);
      UpdateSizeAndAlignForSchema(domain_schemas, schema, depth);
      return {schema.payload_static_size, schema.pack_and_align};
    }
  }
  return {kSizeInvalid, kSizeInvalid};
}

template <typename T>
T ValueOf(const char* payload_data) {
  return *reinterpret_cast<const T*>(payload_data);
}

void ParseValueOfPredefinedType(const NvtxSchemaEntry& entry,
                                const char* payload_base, std::string& output) {
  switch (entry.type) {
    case NVTX_PAYLOAD_ENTRY_TYPE_CHAR:
      absl::StrAppend(&output, absl::string_view(payload_base, 1));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_UCHAR:
      absl::StrAppend(&output, ValueOf<unsigned char>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_SHORT:
      absl::StrAppend(&output, ValueOf<int16_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_USHORT:
      absl::StrAppend(&output, ValueOf<uint16_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_INT:
      absl::StrAppend(&output, ValueOf<int>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_UINT:
      absl::StrAppend(&output, ValueOf<unsigned int>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_LONG:
      absl::StrAppend(&output, ValueOf<int32_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_ULONG:
      absl::StrAppend(&output, ValueOf<uint32_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_LONGLONG:
      absl::StrAppend(&output, ValueOf<int64_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_ULONGLONG:
      absl::StrAppend(&output, ValueOf<uint64_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_INT8:
      absl::StrAppend(&output, ValueOf<int8_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_UINT8:
      absl::StrAppend(&output, ValueOf<uint8_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_INT16:
      absl::StrAppend(&output, ValueOf<int16_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_UINT16:
      absl::StrAppend(&output, ValueOf<uint16_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_INT32:
      absl::StrAppend(&output, ValueOf<int32_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_UINT32:
      absl::StrAppend(&output, ValueOf<uint32_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_INT64:
      absl::StrAppend(&output, ValueOf<int64_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_UINT64:
      absl::StrAppend(&output, ValueOf<uint64_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT:
      absl::StrAppend(&output, ValueOf<float>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_DOUBLE:
      absl::StrAppend(&output, ValueOf<double>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_LONGDOUBLE:
      absl::StrAppend(&output,
                      std::to_string(ValueOf<long double>(payload_base)));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_SIZE:
      absl::StrAppend(&output, ValueOf<size_t>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT32:
      absl::StrAppend(&output, ValueOf<float>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_FLOAT64:
      absl::StrAppend(&output, ValueOf<double>(payload_base));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_ADDRESS:
      absl::StrAppend(&output, absl::Hex(ValueOf<void*>(payload_base)));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING:
    case NVTX_PAYLOAD_ENTRY_TYPE_CSTRING_UTF8:
      absl::StrAppend(&output, absl::string_view(payload_base, entry.extent));
      break;
    case NVTX_PAYLOAD_ENTRY_TYPE_BYTE:
      absl::StrAppend(&output, absl::Hex(ValueOf<unsigned char>(payload_base)));
      break;
    default:
      VLOG(3) << "NVTX payload schema entry type " << entry.type
              << " is not supported as a predefined type.";
      break;
  }
}

size_t ParseValueOfPayloadEnum(const NvtxPayloadEnum* enum_attrs,
                               const char* payload_base, std::string& output) {
  auto size_of_enum = enum_attrs->size_of_enum;
  if (size_of_enum == 8 || size_of_enum == 4) {
    uint64_t enum_value =
        size_of_enum == 8
            ? *reinterpret_cast<const uint64_t*>(payload_base)
            : static_cast<uint64_t>(
                  *reinterpret_cast<const uint32_t*>(payload_base));
    for (const NvtxEnumEntry& entry : enum_attrs->entries) {
      if (entry.value == enum_value) {
        absl::StrAppend(&output, entry.name, "(", enum_value, ")");
        return size_of_enum;
      }
    }
    absl::StrAppend(&output, "UNKNOWN_ENUM_VALUE(", enum_value, ")");
  }
  return size_of_enum;
}

size_t ParseNvtxExtPayloadEntry(NvtxSchemaIdToSchema& domain_schemas,
                                const NvtxSchemaEntry& entry,
                                const char* payload_base, uint64_t payload_size,
                                std::string& result_str, int depth) {
  auto [entry_size, entry_align] =
      GetSizeAndAlign(domain_schemas, entry, depth);
  if (!AreFixedSizes(entry_size, entry_align)) {
    return kSizeInvalid;
  }
  if (payload_size < entry_size) {
    VLOG(1) << "NVTX payload size " << payload_size << " < entry size "
            << entry_size << " for entry " << entry.name << " (" << entry.type
            << ")";
    return kSizeInvalid;
  }

  if (!entry.name.empty()) {
    absl::StrAppend(&result_str, entry.name, " : ");
  }

  if (entry.type < NVTX_PAYLOAD_SCHEMA_ID_STATIC_START) {
    ParseValueOfPredefinedType(entry, payload_base, result_str);
    return entry_size;
  }

  NvtxPayloadAttributes* payload_attributes =
      domain_schemas.GetNvtxPayloadAttributes(entry.type);
  if (payload_attributes == nullptr) {
    return kSizeInvalid;
  }

  if (payload_attributes->IsEnum()) {
    auto& payload_enum = *static_cast<NvtxPayloadEnum*>(payload_attributes);
    ParseValueOfPayloadEnum(&payload_enum, payload_base, result_str);
    return entry_size;
  }

  if (payload_attributes->IsSchema()) {
    auto& schema = *static_cast<NvtxPayloadSchema*>(payload_attributes);
    size_t entry_offset = 0, entry_idx = 0;
    absl::StrAppend(&result_str, depth ? "{" : "");
    for (const NvtxSchemaEntry& entry : schema.entries) {
      auto [entry_size, entry_align] = GetSizeAndAlign(domain_schemas, entry);
      // Align the entry_offset.
      if (!AreFixedSizes(entry_size, entry_align) ||
          !AlignEntryOffset(entry_offset, entry_idx ? entry.offset : 0,
                            entry_size, entry_align, entry_idx) ||
          entry_offset > payload_size) {
        return kSizeInvalid;
      }
      if (entry_idx) {
        absl::StrAppend(&result_str, ", ");
      }
      ParseNvtxExtPayloadEntry(
          domain_schemas, entry, payload_base + entry_offset,
          payload_size - entry_offset, result_str, depth + 1);

      entry_offset += entry_size;
      entry_idx++;
    }
    absl::StrAppend(&result_str, depth ? "}" : "");
  }
  return entry_size;
}

}  // namespace

void CuptiParseNvtxPayload(uint32_t cupti_domain_id,
                           nvtxPayloadData_t* payload_data,
                           std::string& result_str) {
  if (payload_data != nullptr && payload_data->payload != nullptr &&
      payload_data->size > 0) {
    ParseNvtxExtPayloadEntry(
        NvtxDomainSchemas::ForDomain(cupti_domain_id),
        NvtxSchemaEntry{.type = payload_data->schemaId, .name = ""},
        reinterpret_cast<const char*>(payload_data->payload),
        payload_data->size, result_str, /*depth=*/0);
  }
}

}  // namespace profiler
}  // namespace xla
