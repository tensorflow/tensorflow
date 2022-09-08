/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/util.h"

#include <cfloat>
#include <map>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
std::string GetGlslConversion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size) {
  if (src_type == dst_type) {
    return "";
  }
  bool need_explicit_conversion = true;
  switch (dst_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
      if (gpu_info.IsGlslSupportsExplicitFp16()) {
        if (src_type == dst_type) {
          need_explicit_conversion = false;
        }
      } else {
        if (src_type == DataType::FLOAT32 || src_type == DataType::FLOAT16) {
          need_explicit_conversion = false;
        }
      }
      break;
    case DataType::INT32:
    case DataType::INT16:
    case DataType::INT8:
      if (src_type == DataType::INT32 || src_type == DataType::INT16 ||
          src_type == DataType::INT8) {
        need_explicit_conversion = false;
      }
      break;
    case DataType::UINT32:
    case DataType::UINT16:
    case DataType::UINT8:
      if (src_type == DataType::UINT32 || src_type == DataType::UINT16 ||
          src_type == DataType::UINT8) {
        need_explicit_conversion = false;
      }
      break;
    case DataType::BOOL:
      need_explicit_conversion = true;
      break;
    default:
      break;
  }
  if (need_explicit_conversion) {
    return ToGlslShaderDataType(
        dst_type, vec_size,
        /*add_precision*/ false,
        /*explicit_fp16*/ gpu_info.IsGlslSupportsExplicitFp16());
  } else {
    return "";
  }
}

bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}
}  // namespace

std::string MemoryTypeToCLType(MemoryType type) {
  switch (type) {
    case MemoryType::GLOBAL:
      return "__global";
    case MemoryType::CONSTANT:
      return "__constant";
    case MemoryType::LOCAL:
      return "__local";
  }
  return "";
}

std::string MemoryTypeToMetalType(MemoryType type) {
  switch (type) {
    case MemoryType::GLOBAL:
      return "device";
    case MemoryType::CONSTANT:
      return "constant";
      break;
    case MemoryType::LOCAL:
      return "threadgroup";
  }
  return "";
}

float4 GetMaskForLastPlane(int channels) {
  float4 mask = float4(0.0f);
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int GetRecommendedBlockSizeForConv(const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   int task_size) {
  const float task_size_per_cu =
      task_size / static_cast<float>(gpu_info.GetComputeUnitsCount());
  int block_size = 1;
  float threshold_1 = FLT_MAX;
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (!gpu_info.IsMali()) {
    return 1;
  }
  MaliInfo mali_info = gpu_info.mali_info;
  switch (precision) {
    case CalculationsPrecision::F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
        threshold_4 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 6.0f;
        threshold_4 = 256.0f * 16.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
        threshold_2 = 256.0f * 16.0f;
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 3.0f;
        threshold_4 = 256.0f * 32.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 8.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
      }
      break;
    case CalculationsPrecision::F32:
      if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 128.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen3() || mali_info.IsValhall()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 12.0f;
      } else if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 16.0f;
      }
      break;
  }
  if (task_size_per_cu <= threshold_1) {
    block_size = 1;
  } else if (task_size_per_cu <= threshold_2) {
    block_size = 2;
  } else if (task_size_per_cu <= threshold_4) {
    block_size = 4;
  } else {
    block_size = 8;
  }
  return block_size;
}

int3 GetWorkGroupsCount(const int3& grid_size, const int3& work_group_size) {
  int3 work_groups_count;
  work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
  work_groups_count.y = DivideRoundUp(grid_size.y, work_group_size.y);
  work_groups_count.z = DivideRoundUp(grid_size.z, work_group_size.z);
  return work_groups_count;
}

std::string GetTypeDeclaration(const GpuInfo& gpu_info, DataType data_type,
                               int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return ToCLDataType(data_type, vec_size);
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size);
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, true,
                                gpu_info.IsGlslSupportsExplicitFp16());
  } else {
    return "";
  }
}

std::string GetZeroValue(const GpuInfo& gpu_info, DataType data_type,
                         int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return "(" + ToCLDataType(data_type, vec_size) + ")(0)";
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size) + "(0)";
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, false,
                                gpu_info.IsGlslSupportsExplicitFp16()) +
           "(0)";
  } else {
    return "";
  }
}

std::string GetOneValue(const GpuInfo& gpu_info, DataType data_type,
                        int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return "(" + ToCLDataType(data_type, vec_size) + ")(1)";
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size) + "(1)";
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, false,
                                gpu_info.IsGlslSupportsExplicitFp16()) +
           "(1)";
  } else {
    return "";
  }
}

std::string GetTypeConversion(const GpuInfo& gpu_info, DataType src_type,
                              DataType dst_type, int vec_size) {
  if (src_type != dst_type) {
    if (gpu_info.IsApiOpenCl()) {
      if (dst_type == DataType::BOOL && vec_size != 1) {
        // In OpenCL for bool4 we are using uchar4
        // From OpenCL specification for "Relational and Equality Operators":
        //   "These functions shall return a 0 if the specified relation is
        //   false and a -1 (i.e. all bits set) if the specified relation is
        //   true for vector argument types."
        // (convert_uchar4((value) != 0) & (uchar4)(1))
        return "(convert_" + ToCLDataType(DataType::UINT8, vec_size) +
               "(($0) != " + GetZeroValue(gpu_info, src_type, vec_size) +
               ") & " + GetOneValue(gpu_info, DataType::UINT8, vec_size) + ")";
      } else {
        return "convert_" + ToCLDataType(dst_type, vec_size) + "($0)";
      }
    } else if (gpu_info.IsApiMetal()) {
      return ToMetalDataType(dst_type, vec_size) + "($0)";
    } else if (gpu_info.IsGlsl()) {
      const std::string conversion =
          GetGlslConversion(gpu_info, src_type, dst_type, vec_size);
      if (!conversion.empty()) {
        return conversion + "($0)";
      } else {
        return "$0";
      }
    }
  }
  return "$0";
}

std::string GetNextWord(const std::string& code, size_t first_position) {
  size_t pos = first_position;
  char t = code[pos];
  while (IsWordSymbol(t)) {
    pos++;
    t = code[pos];
  }
  return code.substr(first_position, pos - first_position);
}

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket) {
  const std::map<char, char> brackets = {
      {'(', ')'},
      {'{', '}'},
      {'[', ']'},
      {'<', '>'},
  };
  char b_open = bracket;
  auto it = brackets.find(b_open);
  if (it == brackets.end()) {
    return -1;
  }
  char b_close = it->second;
  size_t pos = first_pos;
  int opened = 1;
  int closed = 0;
  while (opened != closed && pos < text.size()) {
    if (text[pos] == b_open) {
      opened++;
    } else if (text[pos] == b_close) {
      closed++;
    }
    pos++;
  }
  if (opened == closed) {
    return pos;
  } else {
    return -1;
  }
}

absl::Status ParseArgsInsideBrackets(const std::string& text,
                                     size_t open_bracket_pos,
                                     size_t* close_bracket_pos,
                                     std::vector<std::string>* args) {
  *close_bracket_pos =
      FindEnclosingBracket(text, open_bracket_pos + 1, text[open_bracket_pos]);
  if (*close_bracket_pos == -1) {
    return absl::NotFoundError("Not found enclosing bracket");
  }
  std::string str_args = text.substr(open_bracket_pos + 1,
                                     *close_bracket_pos - open_bracket_pos - 2);
  std::vector<absl::string_view> words = absl::StrSplit(str_args, ',');
  args->reserve(words.size());
  for (const auto& word : words) {
    absl::string_view arg = absl::StripAsciiWhitespace(word);
    if (!arg.empty()) {
      args->push_back(std::string(arg));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
