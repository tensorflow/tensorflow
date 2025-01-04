/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/group_normalization.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tflite {
namespace gpu {

namespace {
std::string GetGroupNormalizationCode(const OperationDef& op_def, GroupNormalizationAttributes attr) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";

  // getting the X, Y, S, B values.
  if (op_def.IsBatchSupported()) {
    c += "  int linear_id = GLOBAL_ID_0;\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = GLOBAL_ID_0;\n";
  }
  c += "  int Y = GLOBAL_ID_1;\n";
  c += "  int S = GLOBAL_ID_2;\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";

// calculate the mean and variance of the dst_tensor element
// identified by (X, Y, S) (i am assuming the batch is 1 for 
// this case, or tensorflow handles this case not sure)
  c += "  FLT4 x = args.src_tensor.Read(X, Y, S, B);\n";
  c += "  int group_size = args.src_tensor.Channels()/ args.groups;\n";

  // have to check for divisor = 0 case (generally it should not happen for this layer)
  c += "  int divisor = group_size * args.src_tensor.Height() * args.src_tensor.Width();\n";

// there will be two mean and two variance for a 4 channels packed in a slice
// so calculating mean1, mean2 and var1 and var2
  c += "  int group_id_1 = (S*4) / group_size;\n";
  c += "  int group_id_2 = group_id_1 + 1;\n";
  c += "  float mean_array[4];\n";
  c += "  float var_array[4];\n";

  // calculating the mean for group1
  c += "  int left_index = group_id_1 * group_size;\n";
  c += "  int right_index = left_index + group_size;\n"; 
  c += "  float mean1 = 0.0f;\n";
  c += "  for(int i=left_index; i < right_index; i++){\n";
  c += "    int s_idx = i/4;\n";
  c += "    FLT4 value = args.mean_tensor.Read(0,0,s_idx);\n";
  c += "    mean_array[0] = value.x;\n";
  c += "    mean_array[1] = value.y;\n";
  c += "    mean_array[2] = value.z;\n";
  c += "    mean_array[3] = value.w;\n";
  c += "    mean1 = mean1 + mean_array[i % 4];\n";
  c += "  }\n";
  c += "  mean1 = mean1 / divisor;\n";

  // calculating the mean for group2
  c += "  left_index = group_id_2 * group_size;\n";
  c += "  right_index = left_index + group_size;\n"; 
  c += "  float mean2 = 0.0f;\n";
  c += "  for(int i=left_index; i < right_index; i++){\n";
  c += "    int s_idx = i/4;\n";
  c += "    FLT4 value = args.mean_tensor.Read(0,0,s_idx);\n";
  c += "    mean_array[0] = value.x;\n";
  c += "    mean_array[1] = value.y;\n";
  c += "    mean_array[2] = value.z;\n";
  c += "    mean_array[3] = value.w;\n";
  c += "    mean2 = mean2 + mean_array[i % 4];\n";
  c += "  }\n";
  c += "  mean2 = mean2 / divisor;\n";

  // calculating the var for group1
  c += "  left_index = group_id_1 * group_size;\n";
  c += "  right_index = left_index + group_size;\n"; 
  c += "  float var1 = 0.0f;\n";
  c += "  for(int i=left_index; i < right_index; i++){\n";
  c += "    int s_idx = i/4;\n";
  c += "    FLT4 value = args.var_tensor.Read(0,0,s_idx);\n";
  c += "    var_array[0] = value.x;\n";
  c += "    var_array[1] = value.y;\n";
  c += "    var_array[2] = value.z;\n";
  c += "    var_array[3] = value.w;\n";
  c += "    var1 = var1 + var_array[i % 4];\n";
  c += "  }\n";
  // using population variance 
  c += "  var1 = sqrt(var1 / divisor);\n";

  // calculating the var for group2
  c += "  left_index = group_id_2 * group_size;\n";
  c += "  right_index = left_index + group_size;\n"; 
  c += "  float var2 = 0.0f;\n";
  c += "  for(int i=left_index; i < right_index; i++){\n";
  c += "    int s_idx = i/4;\n";
  c += "    FLT4 value = args.var_tensor.Read(0,0,s_idx);\n";
  c += "    var_array[0] = value.x;\n";
  c += "    var_array[1] = value.y;\n";
  c += "    var_array[2] = value.z;\n";
  c += "    var_array[3] = value.w;\n";
  c += "    var2 = var2 + var_array[i % 4];\n";
  c += "  }\n";
  c += "  var2 = sqrt((var2 / divisor));\n";

  // now i have to calculate the mean for the 4 index;
  c += " float mean_group[2] = {mean1, mean2};\n";
  c += " int idx1 = ((S*4)/group_size) - group_id_1;\n";
  c += " int idx2 = ((S*4+1)/group_size) - group_id_1;\n";
  c += " int idx3 = ((S*4+2)/group_size) - group_id_1;\n";
  c += " int idx4 = ((S*4+3)/group_size) - group_id_1;\n";
  c += " FLT4 mean = (FLT4){mean_group[idx1], mean_group[idx2], mean_group[idx3], mean_group[idx4]};\n";

  // now i have to calculate the var for the 4 index;
  c += " float var_group[2] = {var1, var2};\n";
  c += " idx1 = ((S*4)/group_size) - group_id_1;\n";
  c += " idx2 = ((S*4+1)/group_size) - group_id_1;\n";
  c += " idx3 = ((S*4+2)/group_size) - group_id_1;\n";
  c += " idx4 = ((S*4+3)/group_size) - group_id_1;\n";
  c += " FLT4 var = (FLT4){var_group[idx1], var_group[idx2], var_group[idx3], var_group[idx4]};\n";


  // calculating the var for group1
  c += "  FLT4 epsilon_vec = INIT_FLT4v4(args.epsilon, args.epsilon, args.epsilon, args.epsilon);\n";
  c += "  FLT4 x_cap = (x - mean)/(var + epsilon_vec);\n";

  // read beta and gamma and do gamma * xcap + beta
  if(op_def.precision == CalculationsPrecision::F32){
    c += "  FLT4 gamma = args.gamma.Read(0, 0, S, 0);\n";
    c += "  FLT4 beta = args.beta.Read(0, 0, S, 0);\n";
  }else{
    c += "  FLT4 gamma = convert_half4(args.gamma.Read(0, 0, S, 0));\n";
    c += "  FLT4 beta = convert_half4(args.beta.Read(0, 0, S, 0));\n";
  }

  c += "  FLT4 result = (x_cap*gamma) + beta;\n";
  c += "  args.dst_tensor.Write(result, X, Y, S, B);\n";
  c += "}\n";

  return c;
}
}  // namespace

GPUOperation CreateGroupNormalization(const GpuInfo& gpu_info, 
                                      const OperationDef& op_def,
                                      const GroupNormalizationAttributes& attr) {
  GPUOperation op(op_def);

  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  // this tensor comes from transformation
  op.AddSrcTensor("mean_tensor", op_def.src_tensors[1]);
  op.AddSrcTensor("var_tensor", op_def.src_tensors[2]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  op.code_ = GetGroupNormalizationCode(op_def, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  BHWC gamma_shape = BHWC(1, 1, 1, attr.gamma.shape.v);
  TensorStorageType storage_type = TensorStorageType::TEXTURE_2D;

  TensorDescriptor gamma_tensor = 
    CreateBhwcTensorDescriptor(DataType::FLOAT32, storage_type, gamma_shape);
  gamma_tensor.UploadLinearDataChannels(attr.gamma);
  op.args_.AddObject("gamma", 
    std::make_unique<TensorDescriptor>(std::move(gamma_tensor)));

  BHWC beta_shape = BHWC(1, 1, 1, attr.beta.shape.v);
  TensorDescriptor beta_tensor = 
    CreateBhwcTensorDescriptor(DataType::FLOAT32, storage_type, beta_shape);
  beta_tensor.UploadLinearDataChannels(attr.beta);
  op.args_.AddObject("beta", 
    std::make_unique<TensorDescriptor>(std::move(beta_tensor)));

  op.args_.AddInt("axis", attr.axis);
  op.args_.AddInt("groups", attr.groups);
  op.args_.AddInt("centre", (int)attr.centre);
  op.args_.AddInt("scale", (int)attr.scale);
  op.args_.AddFloat("epsilon", attr.epsilon);

  return op;
}

}  // namespace gpu
}  // namespace tflite