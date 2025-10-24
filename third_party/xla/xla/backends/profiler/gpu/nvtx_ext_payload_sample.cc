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

#include "xla/backends/profiler/gpu/nvtx_ext_payload_sample.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExtPayload.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

nvtxDomainHandle_t XProfNvtxDomain() {
  static nvtxDomainHandle_t domain = nvtxDomainCreateA("xprof");
  return domain;
}

nvtxStringHandle_t RegisteredMessage(const char* message) {
  return nvtxDomainRegisterStringA(XProfNvtxDomain(), message);
}

class NvtxScopedRangeEx final {
 public:
  explicit NvtxScopedRangeEx(nvtxStringHandle_t msg,
                             nvtxPayloadData_t* p = nullptr) {
    nvtxEventAttributes_t event_attr{0};
    event_attr.version = NVTX_VERSION;
    event_attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event_attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event_attr.message.registered = msg;
    if (p) {
      event_attr.payloadType = NVTX_PAYLOAD_TYPE_EXT;
      event_attr.reserved0 = 1;  // one single binary payload per event
      event_attr.payload.ullValue = NVTX_POINTER_AS_PAYLOAD_ULLVALUE(p);
    }
    nvtxDomainRangePushEx(XProfNvtxDomain(), &event_attr);
  }

  ~NvtxScopedRangeEx() { nvtxDomainRangePop(XProfNvtxDomain()); }
};

constexpr uint64_t REDUCE_OP_ENUM_SCHEMA_ID =
    NVTX_PAYLOAD_SCHEMA_ID_STATIC_START + 1;
constexpr uint64_t REDUCE_OP_AUTO_OFFSET_ENUM_SCHEMA_ID =
    NVTX_PAYLOAD_SCHEMA_ID_STATIC_START + 2;
constexpr uint64_t REDUCE_PARAMS_SCHEMA_ID =
    NVTX_PAYLOAD_SCHEMA_ID_STATIC_START + 3;
constexpr uint64_t REDUCE_PARAMS_AUTO_OFFSET_SCHEMA_ID =
    NVTX_PAYLOAD_SCHEMA_ID_STATIC_START + 4;

}  // namespace

void RegisteredSchemas() {
  // Register the enum for reduction operation, keep the enum name same as the
  // enum type to mark it is used in manuall offset schema.
  static constexpr nvtxPayloadEnum_t NvtxEnumRedSchema[] = {{"Sum", Sum, 0},
                                                            {"Avg", Avg, 0}};
  static constexpr nvtxPayloadEnumAttr_t enum_attr{
      .fieldMask = NVTX_PAYLOAD_ENUM_ATTR_FIELD_ENTRIES |
                   NVTX_PAYLOAD_ENUM_ATTR_FIELD_NUM_ENTRIES |
                   NVTX_PAYLOAD_ENUM_ATTR_FIELD_SIZE |
                   NVTX_PAYLOAD_ENUM_ATTR_FIELD_SCHEMA_ID,
      .name = nullptr,
      .entries = NvtxEnumRedSchema,
      .numEntries = std::extent<decltype(NvtxEnumRedSchema)>::value,
      .sizeOfEnum = sizeof(ReduceOpType),
      .schemaId = REDUCE_OP_ENUM_SCHEMA_ID,
      .extension = nullptr};
  nvtxPayloadEnumRegister(XProfNvtxDomain(), &enum_attr);

  // Register the enum for reduction operation, append the enum name with
  // "Auto" to mark it is used in auto offset schema.
  static constexpr nvtxPayloadEnum_t NvtxEnumRedAutoOffsetSchema[] = {
      {"SumAuto", Sum, 0}, {"AvgAuto", Avg, 0}};
  static constexpr nvtxPayloadEnumAttr_t auto_offset_enum_attr{
      .fieldMask = NVTX_PAYLOAD_ENUM_ATTR_FIELD_ENTRIES |
                   NVTX_PAYLOAD_ENUM_ATTR_FIELD_NUM_ENTRIES |
                   NVTX_PAYLOAD_ENUM_ATTR_FIELD_SIZE |
                   NVTX_PAYLOAD_ENUM_ATTR_FIELD_SCHEMA_ID,
      .name = nullptr,
      .entries = NvtxEnumRedAutoOffsetSchema,
      .numEntries = std::extent<decltype(NvtxEnumRedAutoOffsetSchema)>::value,
      .sizeOfEnum = sizeof(ReduceOpType),
      .schemaId = REDUCE_OP_AUTO_OFFSET_ENUM_SCHEMA_ID,
      .extension = nullptr};
  nvtxPayloadEnumRegister(XProfNvtxDomain(), &auto_offset_enum_attr);

  // Define names of fields in the ReduceParams struct.
  static constexpr char const* kCommunicatorIdName = "CommunicatorId";
  static constexpr char const* kMessageSizeName = "BytesOfMessage";
  static constexpr char const* kRootName = "Root";
  static constexpr char const* kReductionOpName = "ReduceOp";

  // Register the schema for ReduceParams struct with manually given offset.
  static constexpr nvtxPayloadSchemaEntry_t kParamsReduceEntries[] = {
      {.type = NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
       .name = kCommunicatorIdName,
       .offset = offsetof(ReduceParams, comm)},  // Manually give offset.
      {.type = NVTX_PAYLOAD_ENTRY_TYPE_SIZE,
       .name = kMessageSizeName,
       .offset = offsetof(ReduceParams, bytes)},
      {.type = NVTX_PAYLOAD_ENTRY_TYPE_INT,
       .name = kRootName,
       .offset = offsetof(ReduceParams, root)},
      {.type = REDUCE_OP_ENUM_SCHEMA_ID,  // Use manual offset enum.
       .name = kReductionOpName,
       .offset = offsetof(ReduceParams, op)}};
  static constexpr nvtxPayloadSchemaAttr_t reduce_params_schema_attr{
      .fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE |
                   NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
      .name = nullptr, /* schema name is not needed */
      .type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
      .flags = NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
      .entries = kParamsReduceEntries,  // Specify entries manually offset.
      .numEntries = std::extent<decltype(kParamsReduceEntries)>::value,
      .payloadStaticSize = sizeof(ReduceParams),  // Manually give size.
      // Manually set the alignment to largest element size in the ReduceParams.
      .packAlign = sizeof(uint64_t),
      .schemaId = REDUCE_PARAMS_SCHEMA_ID,
      .extension = nullptr};
  nvtxPayloadSchemaRegister(XProfNvtxDomain(), &reduce_params_schema_attr);

  // Register the schema for ReduceParams with auto offset that need tool to
  // automatically calculate. Also the alignment.
  static constexpr nvtxPayloadSchemaEntry_t kParamsReduceAutoEntries[] = {
      {.type = NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
       .name = kCommunicatorIdName,
       .offset = 0},  // Specify tool automatically calculate offset.
      {.type = NVTX_PAYLOAD_ENTRY_TYPE_SIZE,
       .name = kMessageSizeName,
       .offset = 0},
      {.type = NVTX_PAYLOAD_ENTRY_TYPE_INT,
       .name = kRootName,
       .offset = offsetof(ReduceParams, root)},
      {.type = REDUCE_OP_AUTO_OFFSET_ENUM_SCHEMA_ID,  // Use auto offset enum.
       .name = kReductionOpName,
       .offset = 0}};
  static constexpr nvtxPayloadSchemaAttr_t
      reduce_params_auto_offset_schema_attr{
          .fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
                       NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
                       NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
                       NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
          .name = nullptr, /* schema name is not needed */
          .type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
          .flags = NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
          .entries = kParamsReduceAutoEntries,  // Specify entries auto offset.
          .numEntries = std::extent<decltype(kParamsReduceAutoEntries)>::value,
          .payloadStaticSize = 0,  // Ask tool to automatically calculate size.
          .packAlign = 0,
          .schemaId = REDUCE_PARAMS_AUTO_OFFSET_SCHEMA_ID,
          .extension = nullptr};
  nvtxPayloadSchemaRegister(XProfNvtxDomain(),
                            &reduce_params_auto_offset_schema_attr);
}

void ReduceWithManualOffset(ReduceParams params) {
  static nvtxStringHandle_t func_name_nvtx = RegisteredMessage(__func__);
  nvtxPayloadData_t payload_nvtx = {REDUCE_PARAMS_SCHEMA_ID, sizeof(params),
                                    &params};
  NvtxScopedRangeEx range_nvtx(func_name_nvtx, &payload_nvtx);
  absl::SleepFor(absl::Milliseconds(2));
}

void ReduceWithAutoOffset(ReduceParams params) {
  static nvtxStringHandle_t func_name_nvtx = RegisteredMessage(__func__);
  nvtxPayloadData_t payload_nvtx = {REDUCE_PARAMS_AUTO_OFFSET_SCHEMA_ID,
                                    sizeof(params), &params};
  NvtxScopedRangeEx range_nvtx(func_name_nvtx, &payload_nvtx);
  absl::SleepFor(absl::Milliseconds(2));
}

}  // namespace test
}  // namespace profiler
}  // namespace xla
