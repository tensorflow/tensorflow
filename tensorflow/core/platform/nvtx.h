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
bool NvtxRangesEnabled() {
  static bool is_enabled = [] {
    bool is_disabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DISABLE_NVTX_RANGES",
                                               /*default_val=*/false,
                                               &is_disabled));
    return !is_disabled;
  }();
  return is_enabled;
}

// A helper function to decide whether to enable CUDA NVTX profiling ranges
// with detailed node information.
bool NvtxRangesDetailedEnabled() {
  static bool is_enabled = [] {
    bool _is_enabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_ENABLE_NVTX_RANGES_DETAILED",
                                               /*default_val=*/false,
                                               &_is_enabled));
    return _is_enabled;
  }();
  return is_enabled;
}

string DataTypeToNumpyString(DataType dtype) {
  int dtype_i = static_cast<int>(dtype);
  bool is_ref = false;
  if (dtype_i > 100) {
    is_ref = true;
    dtype_i -= 100;
  }
  const char* ret = "unknown";
  // clang-format off
  switch(dtype) {
    case DT_INVALID: ret = "unknown"; break;
    case DT_FLOAT: ret = "float32"; break;
    case DT_DOUBLE: ret = "float64"; break;
    case DT_INT32: ret = "int32"; break;
    case DT_UINT8: ret = "uint8"; break;
    case DT_INT16: ret = "int16"; break;
    case DT_INT8: ret = "int8"; break;
    case DT_STRING: ret = "string"; break;
    case DT_COMPLEX64: ret = "complex64"; break;
    case DT_INT64: ret = "int64"; break;
    case DT_BOOL: ret = "bool"; break;
    case DT_QINT8: ret = "qint8"; break;       // Not an actual Numpy type
    case DT_QUINT8: ret = "quint8"; break;     // Not an actual Numpy type
    case DT_QINT32: ret = "qint32"; break;     // Not an actual Numpy type
    case DT_BFLOAT16: ret = "bfloat32";break;  // Not an actual Numpy type
    case DT_QINT16: ret = "qint16"; break;
    case DT_QUINT16: ret = "quint16"; break;
    case DT_UINT16: ret = "uint16"; break;
    case DT_COMPLEX128: ret = "complex128"; break;
    case DT_HALF: ret = "float16"; break;
    case DT_RESOURCE: ret = "object"; break;
    case DT_VARIANT: ret = "object"; break;
    case DT_UINT32: ret = "uint32"; break;
    case DT_UINT64: ret = "uint64"; break;
    default: break;
  }
  // clang-format on
  return is_ref ? strings::StrCat(ret, "&") : ret;
}

// TODO(benbarsdell): This is a bit crude and hacky (and inefficient).
string AttrValueToJson(const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return SummarizeAttrValue(attr_value);
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF:
      return strings::StrCat(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return strings::StrCat("\"", DataTypeToNumpyString(attr_value.type()),
                             "\"");
    case AttrValue::kShape: {
      if (attr_value.shape().unknown_rank()) return "null";
      return PartialTensorShape::DebugString(attr_value.shape());
    }
    case AttrValue::kTensor:
      return strings::StrCat("\"", SummarizeAttrValue(attr_value), "\"");
    case AttrValue::kList: {
      std::vector<string> pieces;
      if (attr_value.list().s_size() > 0) {
        return SummarizeAttrValue(attr_value);
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().i(i)));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().f(i)));
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          pieces.push_back(attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          pieces.push_back(strings::StrCat(
              "\"", DataTypeToNumpyString(attr_value.list().type(i)), "\""));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          pieces.push_back(
              attr_value.list().shape(i).unknown_rank()
                  ? "null"
                  : TensorShape::DebugString(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        return strings::StrCat("\"", SummarizeAttrValue(attr_value), "\"");
      } else if (attr_value.list().func_size() > 0) {
        return strings::StrCat("\"", SummarizeAttrValue(attr_value), "\"");
      }
      // Truncate long lists and indicate with an ending null value.
      constexpr int kMaxListSummarySize = 10;
      if (pieces.size() > kMaxListSummarySize) {
        pieces.erase(pieces.begin() + kMaxListSummarySize, pieces.end());
        pieces.push_back("null");
      }
      return strings::StrCat("[", str_util::Join(pieces, ","), "]");
    }
    case AttrValue::kFunc: {
      return strings::StrCat("\"", SummarizeAttrValue(attr_value), "\"");
    }
    case AttrValue::kPlaceholder:
      return strings::StrCat("\"$", attr_value.placeholder(), "\"");
    case AttrValue::VALUE_NOT_SET:
      return "\"<Unknown AttrValue type>\"";
  }
  return "\"<Unknown AttrValue type>\"";  // Prevent missing return warning
}

class NvtxDomain {
 public:
  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
  ~NvtxDomain() { nvtxDomainDestroy(handle_); }
  operator nvtxDomainHandle_t() const { return handle_; }

 private:
  nvtxDomainHandle_t handle_;
  TF_DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
};

static const NvtxDomain& GetNvtxTensorFlowCoreDomain() {
  // Singleton because we want the same domain for the lifetime of the process.
  static NvtxDomain nvtx_domain("tensorflow-core");
  return nvtx_domain;
}

nvtxRangeId_t MaybeNvtxDomainRangeStart(string node_op, string node_name) {
  nvtxRangeId_t nvtx_range;
  if (NvtxRangesEnabled() || NvtxRangesDetailedEnabled()) {
    string msg;
    msg = node_op + ": " + node_name;
    nvtx_range = detail::nvtxRangeStartHelper(msg.c_str(), node_op.c_str(),
                                              GetNvtxTensorFlowCoreDomain());
  }
  return nvtx_range;
}

nvtxRangeId_t MaybeNvtxDomainRangeStartMsg(string msg, string node_op) {
  nvtxRangeId_t nvtx_range;
  if (NvtxRangesEnabled() || NvtxRangesDetailedEnabled()) {
    nvtx_range = detail::nvtxRangeStartHelper(msg.c_str(), node_op.c_str(),
                                              GetNvtxTensorFlowCoreDomain());
  }
  return nvtx_range;
}

void MaybeNvtxDomainRangeEnd(nvtxRangeId_t nvtx_range) {
  if (NvtxRangesEnabled() || NvtxRangesDetailedEnabled()) {
    ::nvtxDomainRangeEnd(GetNvtxTensorFlowCoreDomain(), nvtx_range);
  }
}

}  // namespace nvtx
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NVTX_H_
