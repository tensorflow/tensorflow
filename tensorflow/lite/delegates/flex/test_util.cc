/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/flex/test_util.h"

#include "absl/memory/memory.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace flex {
namespace testing {

bool FlexModelTest::Invoke() { return interpreter_->Invoke() == kTfLiteOk; }

void FlexModelTest::SetStringValues(int tensor_index,
                                    const std::vector<string>& values) {
  DynamicBuffer dynamic_buffer;
  for (const string& s : values) {
    dynamic_buffer.AddString(s.data(), s.size());
  }
  dynamic_buffer.WriteToTensor(interpreter_->tensor(tensor_index),
                               /*new_shape=*/nullptr);
}

std::vector<string> FlexModelTest::GetStringValues(int tensor_index) const {
  std::vector<string> result;

  TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  auto num_strings = GetStringCount(tensor);
  for (size_t i = 0; i < num_strings; ++i) {
    auto ref = GetString(tensor, i);
    result.push_back(string(ref.str, ref.len));
  }

  return result;
}

void FlexModelTest::SetShape(int tensor_index, const std::vector<int>& values) {
  ASSERT_EQ(interpreter_->ResizeInputTensor(tensor_index, values), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

std::vector<int> FlexModelTest::GetShape(int tensor_index) {
  std::vector<int> result;
  auto* dims = interpreter_->tensor(tensor_index)->dims;
  result.reserve(dims->size);
  for (int i = 0; i < dims->size; ++i) {
    result.push_back(dims->data[i]);
  }
  return result;
}

TfLiteType FlexModelTest::GetType(int tensor_index) {
  return interpreter_->tensor(tensor_index)->type;
}

bool FlexModelTest::IsDynamicTensor(int tensor_index) {
  return interpreter_->tensor(tensor_index)->allocation_type == kTfLiteDynamic;
}

void FlexModelTest::AddTensors(int num_tensors, const std::vector<int>& inputs,
                               const std::vector<int>& outputs, TfLiteType type,
                               const std::vector<int>& dims) {
  interpreter_->AddTensors(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteQuantizationParams quant;
    // Suppress explicit output type specification to ensure type inference
    // works properly.
    if (std::find(outputs.begin(), outputs.end(), i) != outputs.end()) {
      type = kTfLiteFloat32;
    }
    CHECK_EQ(interpreter_->SetTensorParametersReadWrite(i, type,
                                                        /*name=*/"",
                                                        /*dims=*/dims, quant),
             kTfLiteOk);
  }

  CHECK_EQ(interpreter_->SetInputs(inputs), kTfLiteOk);
  CHECK_EQ(interpreter_->SetOutputs(outputs), kTfLiteOk);
}

void FlexModelTest::SetConstTensor(int tensor_index,
                                   const std::vector<int>& values,
                                   TfLiteType type, const char* buffer,
                                   size_t bytes) {
  TfLiteQuantizationParams quant;
  CHECK_EQ(interpreter_->SetTensorParametersReadOnly(tensor_index, type,
                                                     /*name=*/"",
                                                     /*dims=*/values, quant,
                                                     buffer, bytes),
           kTfLiteOk);
}

void FlexModelTest::AddTfLiteMulOp(const std::vector<int>& inputs,
                                   const std::vector<int>& outputs) {
  ++next_op_index_;

  static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.builtin_code = BuiltinOperator_MUL;
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    auto* i0 = &context->tensors[node->inputs->data[0]];
    auto* o = &context->tensors[node->outputs->data[0]];
    return context->ResizeTensor(context, o, TfLiteIntArrayCopy(i0->dims));
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    auto* i0 = &context->tensors[node->inputs->data[0]];
    auto* i1 = &context->tensors[node->inputs->data[1]];
    auto* o = &context->tensors[node->outputs->data[0]];
    for (int i = 0; i < o->bytes / sizeof(float); ++i) {
      o->data.f[i] = i0->data.f[i] * i1->data.f[i];
    }
    return kTfLiteOk;
  };

  CHECK_EQ(interpreter_->AddNodeWithParameters(inputs, outputs, nullptr, 0,
                                               nullptr, &reg),
           kTfLiteOk);
}

void FlexModelTest::AddTfOp(TfOpType op, const std::vector<int>& inputs,
                            const std::vector<int>& outputs) {
  tf_ops_.push_back(next_op_index_);
  ++next_op_index_;

  auto attr = [](const string& key, const string& value) {
    return " attr{ key: '" + key + "' value {" + value + "}}";
  };

  string type_attribute;
  switch (interpreter_->tensor(inputs[0])->type) {
    case kTfLiteInt32:
      type_attribute = attr("T", "type: DT_INT32");
      break;
    case kTfLiteFloat32:
      type_attribute = attr("T", "type: DT_FLOAT");
      break;
    case kTfLiteString:
      type_attribute = attr("T", "type: DT_STRING");
      break;
    default:
      // TODO(b/113613439): Use nodedef string utilities to properly handle all
      // types.
      LOG(FATAL) << "Type not supported";
      break;
  }

  if (op == kUnpack) {
    string attributes =
        type_attribute + attr("num", "i: 2") + attr("axis", "i: 0");
    AddTfOp("FlexUnpack", "Unpack", attributes, inputs, outputs);
  } else if (op == kIdentity) {
    string attributes = type_attribute;
    AddTfOp("FlexIdentity", "Identity", attributes, inputs, outputs);
  } else if (op == kAdd) {
    string attributes = type_attribute;
    AddTfOp("FlexAdd", "Add", attributes, inputs, outputs);
  } else if (op == kMul) {
    string attributes = type_attribute;
    AddTfOp("FlexMul", "Mul", attributes, inputs, outputs);
  } else if (op == kRfft) {
    AddTfOp("FlexRFFT", "RFFT", "", inputs, outputs);
  } else if (op == kImag) {
    AddTfOp("FlexImag", "Imag", "", inputs, outputs);
  } else if (op == kNonExistent) {
    AddTfOp("NonExistentOp", "NonExistentOp", "", inputs, outputs);
  } else if (op == kIncompatibleNodeDef) {
    // "Cast" op is created without attributes - making it incompatible.
    AddTfOp("FlexCast", "Cast", "", inputs, outputs);
  }
}

void FlexModelTest::AddTfOp(const char* tflite_name, const string& tf_name,
                            const string& nodedef_str,
                            const std::vector<int>& inputs,
                            const std::vector<int>& outputs) {
  static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.builtin_code = BuiltinOperator_CUSTOM;
  reg.custom_name = tflite_name;

  tensorflow::NodeDef nodedef;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      nodedef_str + " op: '" + tf_name + "'", &nodedef));
  string serialized_nodedef;
  CHECK(nodedef.SerializeToString(&serialized_nodedef));
  flexbuffers::Builder fbb;
  fbb.Vector([&]() {
    fbb.String(nodedef.op());
    fbb.String(serialized_nodedef);
  });
  fbb.Finish();

  flexbuffers_.push_back(fbb.GetBuffer());
  auto& buffer = flexbuffers_.back();
  CHECK_EQ(interpreter_->AddNodeWithParameters(
               inputs, outputs, reinterpret_cast<const char*>(buffer.data()),
               buffer.size(), nullptr, &reg),
           kTfLiteOk);
}

}  // namespace testing
}  // namespace flex
}  // namespace tflite
