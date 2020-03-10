/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/profiler/nvtx_utils.h"

#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace nvtx {

namespace {

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

inline nvtxRangeId_t nvtxRangeStartHelper(const char* msg, const char* type,
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

  if (nvtx_domain != NULL) return ::nvtxDomainRangeStartEx(nvtx_domain, &attrs);

  return ::nvtxRangeStartEx(&attrs);
}

}  // namespace anonymous

// #############################################################################

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
bool check_nvtx_envvar(const char* envvar_name){
  bool is_enabled = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(envvar_name,
                                             /*default_val=*/false,
                                             &is_enabled));
  return is_enabled;
}

const int8 NVTX_RANGE_INJECTION_STATUS = [] {
  if (check_nvtx_envvar("TF_ENABLE_NVTX_RANGES_DETAILED")){
    return NVTX_RANGE_INJECTION_STATUS_kDetailed;
  }
  else if (check_nvtx_envvar("TF_ENABLE_NVTX_RANGES")) {
    return NVTX_RANGE_INJECTION_STATUS_kBasic;
  }
  else {
    return NVTX_RANGE_INJECTION_STATUS_kDisabled;
  }
}();

// #############################################################################

const NvtxDomain& GetNvtxTensorFlowCoreDomain() {
  // Singleton because we want the same domain for the lifetime of the process.
  static NvtxDomain nvtx_domain("tensorflow-core");
  return nvtx_domain;
}

string DataTypeToNumpyString(DataType dtype) {
  const bool is_ref = (static_cast<int>(dtype) > 100);

  absl::string_view ret(DataType_Name(dtype).c_str());
  absl::ConsumePrefix(&ret, "DT_");
  ret = absl::AsciiStrToLower(ret);

  return is_ref ? strings::StrCat(std::string(ret), "&") : std::string(ret);
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
    case AttrValue::kTensor: {
      const TensorProto& tensor_proto = attr_value.tensor();
      const TensorShapeProto& proto_shape = tensor_proto.tensor_shape();
      if (!TensorShape::IsValid(proto_shape)) {
        return strings::StrCat("\"", tensor_proto.ShortDebugString(), "\"");
      }
      TensorShape shape(proto_shape);
      const int64 N = shape.num_elements();
      if (N > 1024 * 128) {
        return strings::StrCat("\"", tensor_proto.ShortDebugString(), "\"");
      }
      return strings::StrCat("\"", SummarizeAttrValue(attr_value), "\"");
    }
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

string MaybeGetNvtxDomainRangeMessage(
    const OpKernel* kernel, const int num_inputs,
    std::vector<const TensorShape*> input_shape_array) {
  string msg;
  if (IsNvtxRangesEnabled()) {
    msg = kernel->def().op() + ": " + kernel->name();
  } else if (IsNvtxRangesDetailedEnabled()) {
    std::vector<string> args_pieces;
    for (int i = 0; i < num_inputs; ++i) {
      if (i == 10) {
        // Truncate long arg lists and indicate with an ending null value.
        args_pieces.push_back("null");
        break;
      }
      const TensorShape& shape = *(input_shape_array[i]);
      string shape_str = shape.unknown_rank() ? "null" : shape.DebugString();
      args_pieces.push_back(strings::StrCat("{\"name\":\"",
                                            kernel->def().input(i),
                                            "\",\"shape\":", shape_str, "}"));
    }
    std::vector<string> attrs_pieces;
    const auto& attrs = kernel->def().attr();
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
      const string& key = it->first;
      const AttrValue& value = it->second;
      // Exclude types that aren't useful for profiling.
      if (value.value_case() == AttrValue::kFunc ||
          value.value_case() == AttrValue::kPlaceholder ||
          value.value_case() == AttrValue::VALUE_NOT_SET) {
        continue;
      }
      string value_str = AttrValueToJson(value);
      attrs_pieces.push_back(strings::StrCat("\"", key, "\":", value_str));
    }
    msg = strings::StrCat("{\"op\":\"", kernel->def().op(), "\",\"name\":\"",
                          kernel->name(), "\",\"args\":[",
                          str_util::Join(args_pieces, ","), "],\"attrs\":{",
                          str_util::Join(attrs_pieces, ","), "}}");
  }
  return msg;
}

nvtxRangeId_t MaybeNvtxDomainRangeStart(string node_op, string node_name) {
  nvtxRangeId_t nvtx_range;
  if (IsNvtxRangesEnabled() || IsNvtxRangesDetailedEnabled()) {
    string msg;
    msg = node_op + ": " + node_name;
    nvtx_range = nvtxRangeStartHelper(
      msg.c_str(),
      node_op.c_str(),
      GetNvtxTensorFlowCoreDomain()
    );
  }
  return nvtx_range;
}

nvtxRangeId_t MaybeNvtxDomainRangeStartMsg(string msg, string node_op) {
  nvtxRangeId_t nvtx_range;
  if (IsNvtxRangesEnabled()) {
    nvtx_range = nvtxRangeStartHelper(
      msg.c_str(),
      node_op.c_str(),
      GetNvtxTensorFlowCoreDomain()
    );
  }
  return nvtx_range;
}

void MaybeNvtxDomainRangeEnd(nvtxRangeId_t nvtx_range) {
  if (IsNvtxRangesEnabled()) {
    nvtxDomainRangeEnd(GetNvtxTensorFlowCoreDomain(), nvtx_range);
  }
}

namespace hlo {

string NvtxNodeNameString(string cluster_name, string op_name) {
  cluster_name = cluster_name.substr(0, cluster_name.find("__XlaCompile"));
  return cluster_name + "_1/xla_run/" + op_name;
}

}  // namespace hlo

}  // namespace nvtx
}  // namespace tensorflow
