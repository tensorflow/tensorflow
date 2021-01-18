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

#include "tensorflow/lite/delegates/gpu/metal/kernels/padding.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {

std::string GetPaddingCode(const PadAttributes& attr) {
  const std::string channels[] = {".x", ".y", ".z", ".w"};
  std::string code = R"(
)";
  if (attr.type == PaddingContentType::REFLECT) {
    code += R"(
    int reflect(int x, int size) {
      return size - 1 - abs(abs(x) - size + 1);
    })";
  }
  code += R"(
    kernel void ComputeFunction($0
                                uint3 gid[[thread_position_in_grid]]) {
      if (static_cast<int>(gid.x) >= args.dst_tensor.Width() ||
          static_cast<int>(gid.y) >= args.dst_tensor.Height()) {
        return;
      }

      FLT4 value = FLT4(0.0f);
      int s_x = static_cast<int>(gid.x) - args.padding_w;
      int s_y = static_cast<int>(gid.y) - args.padding_h;)";
  if (attr.type == PaddingContentType::REFLECT) {
    code += R"(
      s_x = reflect(s_x, args.src_tensor.Width());
      s_y = reflect(s_y, args.src_tensor.Height());
)";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      code += "      value = args.src_tensor.Read(s_x, s_y, gid.z);\n";
    } else {
      code += "      int start_channel = static_cast<int>(gid.z) * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        code += "      {\n";
        code += "        int channel = start_channel + " + std::to_string(i) +
                ";\n";
        code += "        int s_z = channel - args.padding_c;\n";
        // We need additional clamp for z, so that we use alignment for channels
        // and can proceed extra channels that can lead to reading out of
        // resource.
        code +=
            "        s_z = clamp(reflect(s_z, args.src_tensor.Channels()), 0, "
            "args.src_tensor.Channels() - 1);\n";
        code += "        FLT4 t = args.src_tensor.Read(s_x, s_y, s_z / 4);\n";
        code += "        FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        code += "        value" + s + " = t_ar[s_z % 4];\n";
        code += "      }\n";
      }
    }
  } else {
    code += R"(
      bool inside_x = s_x >= 0 && s_x < args.src_tensor.Width();
      bool inside_y = s_y >= 0 && s_y < args.src_tensor.Height();
      if (inside_x && inside_y) {
    )";
    if (attr.prepended.c == 0 && attr.appended.c == 0) {
      // optimized case
      code += "        value = args.src_tensor.Read(s_x, s_y, gid.z);\n";
    } else if (attr.prepended.c % 4 == 0) {
      code += R"(
        int s_z = static_cast<int>(gid.z) - args.padding_c / 4;
        if (s_z >= 0 && s_z < args.src_tensor.Slices()) {
          value = args.src_tensor.Read(s_x, s_y, s_z);
        })";
    } else {
      code += "    int start_channel = static_cast<int>(gid.z) * 4;\n";
      for (int i = 0; i < 4; ++i) {
        const auto& s = channels[i];
        code += "    {\n";
        code +=
            "    int channel = start_channel + " + std::to_string(i) + ";\n";
        code += "    int s_z = channel - args.padding_c;\n";
        code += "    if (s_z >= 0 && s_z < args.src_tensor.Channels()) {\n";
        code += "      FLT4 t = args.src_tensor.Read(s_x, s_y, s_z / 4);\n";
        code += "      FLT t_ar[4] = {t.x, t.y, t.z, t.w};\n";
        code += "      value" + s + " = t_ar[s_z % 4];\n";
        code += "    }\n";
        code += "    }\n";
      }
    }
    code += "  }\n";
  }
  code += "  args.dst_tensor.Write(value, gid.x, gid.y, gid.z);\n";
  code += "}\n";
  return code;
}
}  // namespace

ComputeTaskDescriptor Padding(const OperationDef& definition,
                              const PadAttributes& attr) {
  ComputeTaskDescriptor desc(definition);
  desc.shader_source = GetPaddingCode(attr);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args.AddInt("padding_w", attr.prepended.w);
  desc.args.AddInt("padding_h", attr.prepended.h);
  desc.args.AddInt("padding_c", attr.prepended.c);
  desc.args.AddInt("padding_b", attr.prepended.b);

  desc.resize_function = [attr](const std::vector<BHWC>& src_shapes,
                                const std::vector<BHWC>& dst_shapes) {
    const uint3 groups_size{16, 16, 1};
    const int dst_layers = DivideRoundUp(dst_shapes[0].c, 4);
    int groups_x = DivideRoundUp(dst_shapes[0].w, groups_size.x);
    int groups_y = DivideRoundUp(dst_shapes[0].h, groups_size.y);
    int groups_z = DivideRoundUp(dst_layers, groups_size.z);
    return std::make_pair(groups_size, uint3{groups_x, groups_y, groups_z});
  };

  return desc;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
