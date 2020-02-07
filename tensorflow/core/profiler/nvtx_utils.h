/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_NVTX_H_
#define TENSORFLOW_CORE_PLATFORM_NVTX_H_

#include "third_party/nvtx3/nvToolsExt.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace nvtx {

class NvtxDomain {
 public:
  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
  ~NvtxDomain() { nvtxDomainDestroy(handle_); }
  operator nvtxDomainHandle_t() const { return handle_; }

 private:
  nvtxDomainHandle_t handle_;
  TF_DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
};

static const NvtxDomain& GetNvtxTensorFlowCoreDomain();

namespace detail {

inline unsigned hash_string(const char* c) {
  enum { M = 33 };
  unsigned hash = 5381;
  while (*c) {
    hash = hash * M + *c++;
  }
  return hash;
}

inline uint32_t get_color(unsigned hash) {
  const uint32_t colors[] = {0x00aedb, 0xa200ff, 0xf47835, 0xd41243, 0x8ec127,
                             0xffb3ba, 0xffdfba, 0xffffba, 0xbaffc9, 0xbae1ff,
                             0xbbcbdb, 0x9ebd9e, 0xdd855c, 0xf1e8ca, 0x745151,
                             0x2e4045, 0x83adb5, 0xc7bbc9, 0x5e3c58, 0xbfb5b2,
                             0xff77aa, 0xaaff77, 0x77aaff, 0xffffff, 0x000000};
  const int ncolor = sizeof(colors) / sizeof(uint32_t);
  return colors[hash % ncolor];
}

inline nvtxRangeId_t nvtxRangeStartHelper(const char* msg,
                                          const char* type,
                                          nvtxDomainHandle_t nvtx_domain,
                                          bool set_category = true) {
  unsigned h = hash_string(type);
  uint32_t color = get_color(h);
  uint32_t category = set_category ? h : 0;

  nvtxEventAttributes_t attrs = {};
  attrs.version = NVTX_VERSION;
  attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attrs.colorType = NVTX_COLOR_ARGB;
  attrs.color = color;
  attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attrs.message.ascii = msg;
  attrs.category = category;

  if (nvtx_domain != NULL)
    return ::nvtxDomainRangeStartEx(nvtx_domain, &attrs);

  return ::nvtxRangeStartEx(&attrs);
}

}  // namespace detail

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
bool NvtxRangesEnabled();

// A helper function to decide whether to enable CUDA NVTX profiling ranges
// with detailed node information.
bool NvtxRangesDetailedEnabled();

string DataTypeToNumpyString(DataType dtype);

// TODO(benbarsdell): This is a bit crude and hacky (and inefficient).
string AttrValueToJson(const AttrValue& attr_value);

nvtxRangeId_t MaybeNvtxDomainRangeStart(string node_op, string node_name);

nvtxRangeId_t MaybeNvtxDomainRangeStartMsg(string msg, string node_op);

void MaybeNvtxDomainRangeEnd(nvtxRangeId_t nvtx_range);

}  // namespace nvtx
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NVTX_H_
