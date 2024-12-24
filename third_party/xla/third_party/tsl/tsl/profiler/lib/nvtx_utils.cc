/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/nvtx_utils.h"

#ifdef __linux__
#include <sys/syscall.h>
#endif

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#include "nvtx3/nvToolsExtPayload.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace {
// Get the ID of the current thread following the convention for
// nvtxNameOsThreadA:
// https://nvidia.github.io/NVTX/doxygen/group___r_e_s_o_u_r_c_e___n_a_m_i_n_g.html
// This convention may not match the one in tsl::Env::GetCurrentThreadId().
std::optional<uint32_t> MaybeGetCurrentThreadId() {
#ifdef __linux__
  return syscall(SYS_gettid);
#else
  return std::nullopt;
#endif
}
}  // namespace

namespace tsl::profiler {
static_assert(std::is_pointer_v<CUstream>);
static_assert(std::is_pointer_v<nvtxDomainHandle_t>);
static_assert(std::is_pointer_v<nvtxStringHandle_t>);

ProfilerDomainHandle DefaultProfilerDomain() {
  static ProfilerDomainHandle domain =
      reinterpret_cast<ProfilerDomainHandle>(nvtxDomainCreateA("TSL"));
  return domain;
}

void NameCurrentThread(const std::string& thread_name) {
  if (std::optional<uint32_t> tid = MaybeGetCurrentThreadId();
      tid.has_value()) {
    nvtxNameOsThreadA(*tid, thread_name.c_str());
  }
}

void NameDevice(int device_id, const std::string& device_name) {
  nvtxNameCudaDeviceA(device_id, device_name.c_str());
}

void NameStream(StreamHandle stream, const std::string& stream_name) {
  nvtxNameCuStreamA(reinterpret_cast<CUstream>(stream), stream_name.c_str());
}

void RangePop(ProfilerDomainHandle domain) {
  nvtxDomainRangePop(reinterpret_cast<nvtxDomainHandle_t>(domain));
}

void RangePush(ProfilerDomainHandle domain, const char* ascii) {
  nvtxEventAttributes_t attrs{};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attrs.message.ascii = ascii;
  nvtxDomainRangePushEx(reinterpret_cast<nvtxDomainHandle_t>(domain), &attrs);
}

namespace detail {
void RangePush(ProfilerDomainHandle domain, StringHandle title,
               uint64_t schema_id, const void* payload, size_t payload_size) {
  nvtxEventAttributes_t attrs{};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attrs.message.registered = reinterpret_cast<nvtxStringHandle_t>(title);
#ifdef NVTX_VERSION_3_1
  NVTX_PAYLOAD_EVTATTR_SET(&attrs, schema_id, payload, payload_size);
#else
  NVTX_PAYLOAD_EVTATTR_SET(attrs, schema_id, payload, payload_size);
#endif
  nvtxDomainRangePushEx(reinterpret_cast<nvtxDomainHandle_t>(domain), &attrs);
}
}  // namespace detail

uint64_t RegisterSchema(ProfilerDomainHandle domain, const void* schemaAttr) {
  return nvtxPayloadSchemaRegister(
      reinterpret_cast<nvtxDomainHandle_t>(domain),
      static_cast<const nvtxPayloadSchemaAttr_t*>(schemaAttr));
}

StringHandle RegisterString(ProfilerDomainHandle domain,
                            const std::string& str) {
  const auto impl = [domain](const char* c_str) {
    return reinterpret_cast<StringHandle>(nvtxDomainRegisterStringA(
        reinterpret_cast<nvtxDomainHandle_t>(domain), c_str));
  };
  constexpr auto max_length = 65330;
  if (str.size() <= max_length) {
    return impl(str.c_str());
  }
  // nvbugs 4340868
  std::string_view suffix{"\n[truncated]\n"};
  std::string buffer(str.data(), max_length - suffix.size());
  buffer.append(suffix);
  return impl(buffer.c_str());
}
}  // namespace tsl::profiler
