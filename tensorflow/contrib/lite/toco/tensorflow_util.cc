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
#include "tensorflow/contrib/lite/toco/tensorflow_util.h"

#include <string.h>
#include <memory>
#include <set>

#ifdef GOOGLE_PLATFORM
#include "file/logging/log_lines.h"
#endif
#include "google/protobuf/map.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

using tensorflow::AttrValue;
using tensorflow::GraphDef;

void LogDumpGraphDef(int log_level, const string& message,
                     const GraphDef& tf_graph) {
  if (!VLOG_IS_ON(log_level)) {
    return;
  }
  std::set<string> ops;
  for (const auto& node : tf_graph.node()) {
    ops.insert(node.op());
  }
  string dump;
  toco::port::AppendF(&dump, R"MSG(
BEGIN DUMP OF TENSORFLOW GRAPHDEF (%s)
There are %d nodes.
There are %zu different op types:
)MSG", message, tf_graph.node_size(), ops.size());
  for (const auto& op : ops) {
    toco::port::AppendF(&dump, "  %s\n", op);
  }
  dump.append(R"MSG(
PROTO DUMP
)MSG");
  for (const auto& node : tf_graph.node()) {
    toco::port::AppendF(&dump, R"MSG(
BEGIN NODE: name = %s
  op = %s
  inputs = [
)MSG", node.name(), node.op());
    for (const auto& input : node.input()) {
      toco::port::AppendF(&dump, "    %s\n", input);
    }
    dump.append("  ]\n");
    for (const auto& attr : node.attr()) {
      toco::port::AppendF(&dump, "  ATTR: name = %s\n", attr.first);
      if (attr.second.value_case() == AttrValue::kFunc) {
        dump.append("    func\n");
      } else if (attr.second.value_case() == AttrValue::kPlaceholder) {
        toco::port::AppendF(&dump, "    placeholder: %s\n",
                            attr.second.placeholder());
      } else if (attr.second.value_case() == AttrValue::kS) {
        dump.append("    string:\n");
        dump.append(R"MSG(
      BEGIN EMBEDDED STRING
)MSG");
        const auto& lines = absl::StrSplit(attr.second.s(), '\n');
        for (const auto& line : lines) {
          toco::port::AppendF(&dump, "      %s\n", line);
        }
        dump.append(R"MSG(
      END EMBEDDED STRING
)MSG");
      } else if (attr.second.value_case() == AttrValue::kI) {
        toco::port::AppendF(&dump, "    int: %lld\n", attr.second.i());
      } else if (attr.second.value_case() == AttrValue::kF) {
        toco::port::AppendF(&dump, "    float: %g\n", attr.second.f());
      } else if (attr.second.value_case() == AttrValue::kB) {
        toco::port::AppendF(&dump, "    bool: %s\n",
                            attr.second.b() ? "true" : "false");
      } else if (attr.second.value_case() == AttrValue::kType) {
        toco::port::AppendF(&dump, "    type: %s\n",
                            tensorflow::DataType_Name(attr.second.type()));
      } else if (attr.second.value_case() == AttrValue::kShape) {
        dump.append("    shape: [ ");
        const auto& shape = attr.second.shape();
        for (int i = 0; i < shape.dim_size(); i++) {
          toco::port::AppendF(&dump, "%lld ", shape.dim(i).size());
        }
        dump.append("]\n");
      } else if (attr.second.value_case() == AttrValue::kTensor) {
        const auto& tensor = attr.second.tensor();
        dump.append("    TENSOR:\n");
        toco::port::AppendF(&dump, "      type: %s\n",
                            tensorflow::DataType_Name(tensor.dtype()));
        const auto& shape = tensor.tensor_shape();
        dump.append("      shape: [ ");
        for (int i = 0; i < shape.dim_size(); i++) {
          toco::port::AppendF(&dump, "%lld ", shape.dim(i).size());
        }
        dump.append("]\n");
        if (!tensor.tensor_content().empty()) {
          toco::port::AppendF(&dump, "      tensor_content: %zu bytes\n",
                              tensor.tensor_content().size());
        }
        if (tensor.dtype() == tensorflow::DT_INT32) {
          CHECK_EQ(0, tensor.tensor_content().size() % sizeof(int32));
          const int size = tensor.tensor_content().size() / sizeof(int32);
          std::vector<int32> data(size);
          toco::port::CopyToBuffer(tensor.tensor_content(),
                                   reinterpret_cast<char*>(data.data()));
          const int kMaxValsToPrint = 4;
          dump.append("        tensor_content as ints: [ ");
          for (int i = 0; i < kMaxValsToPrint && i < size; i++) {
            toco::port::AppendF(&dump, "%d ", data[i]);
          }
          if (size > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.dtype() == tensorflow::DT_FLOAT) {
          CHECK_EQ(0, tensor.tensor_content().size() % sizeof(float));
          const int size = tensor.tensor_content().size() / sizeof(float);
          std::vector<float> data(size);
          toco::port::CopyToBuffer(tensor.tensor_content(),
                                   reinterpret_cast<char*>(data.data()));
          const int kMaxValsToPrint = 4;
          dump.append("        tensor_content as floats: [ ");
          for (int i = 0; i < kMaxValsToPrint && i < size; i++) {
            toco::port::AppendF(&dump, "%g ", data[i]);
          }
          if (size > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.int_val_size()) {
          toco::port::AppendF(&dump, "      int_val: %d ints: [ ",
                              tensor.int_val_size());
          const int kMaxValsToPrint = 4;
          for (int i = 0; i < kMaxValsToPrint && i < tensor.int_val_size();
               i++) {
            toco::port::AppendF(&dump, "%d ", tensor.int_val(i));
          }
          if (tensor.int_val_size() > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.float_val_size()) {
          toco::port::AppendF(&dump, "      float_val: %d floats: [ ",
                              tensor.float_val_size());
          const int kMaxValsToPrint = 4;
          for (int i = 0; i < kMaxValsToPrint && i < tensor.float_val_size();
               i++) {
            toco::port::AppendF(&dump, "%g ", tensor.float_val(i));
          }
          if (tensor.float_val_size() > kMaxValsToPrint) {
            dump.append("... ");
          }
          dump.append("]\n");
        }
        if (tensor.string_val_size()) {
          toco::port::AppendF(&dump, "      string_val: %d strings\n",
                              tensor.string_val_size());
        }
      } else if (attr.second.value_case() == AttrValue::kList) {
        dump.append("  LIST\n");
      }
    }
    dump.append("END NODE\n");
  }
  toco::port::AppendF(&dump, "END DUMP OF TENSORFLOW GRAPHDEF (%s)\n", message);
#if defined(GOOGLE_PLATFORM)
  VLOG_LINES(log_level, dump);
#else
  VLOG(log_level) << dump;
#endif
}
}  // namespace toco
