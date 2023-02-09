# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions that support testing.

All functions that can be commonly used by various tests.
"""

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

TFLITE_SCHEMA_VERSION = 3


def build_mock_flatbuffer_model():
  """Creates a flatbuffer containing an example model."""
  builder = flatbuffers.Builder(1024)

  schema_fb.BufferStart(builder)
  buffer0_offset = schema_fb.BufferEnd(builder)

  schema_fb.BufferStartDataVector(builder, 12)
  builder.PrependUint8(11)
  builder.PrependUint8(10)
  builder.PrependUint8(9)
  builder.PrependUint8(8)
  builder.PrependUint8(7)
  builder.PrependUint8(6)
  builder.PrependUint8(5)
  builder.PrependUint8(4)
  builder.PrependUint8(3)
  builder.PrependUint8(2)
  builder.PrependUint8(1)
  builder.PrependUint8(0)
  buffer1_data_offset = builder.EndVector()
  schema_fb.BufferStart(builder)
  schema_fb.BufferAddData(builder, buffer1_data_offset)
  buffer1_offset = schema_fb.BufferEnd(builder)

  schema_fb.BufferStart(builder)
  buffer2_offset = schema_fb.BufferEnd(builder)

  schema_fb.ModelStartBuffersVector(builder, 3)
  builder.PrependUOffsetTRelative(buffer2_offset)
  builder.PrependUOffsetTRelative(buffer1_offset)
  builder.PrependUOffsetTRelative(buffer0_offset)
  buffers_offset = builder.EndVector()

  string0_offset = builder.CreateString('input_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape0_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string0_offset)
  schema_fb.TensorAddShape(builder, shape0_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 0)
  tensor0_offset = schema_fb.TensorEnd(builder)

  schema_fb.QuantizationParametersStartMinVector(builder, 5)
  builder.PrependFloat32(0.5)
  builder.PrependFloat32(2.0)
  builder.PrependFloat32(5.0)
  builder.PrependFloat32(10.0)
  builder.PrependFloat32(20.0)
  quant1_min_offset = builder.EndVector()

  schema_fb.QuantizationParametersStartMaxVector(builder, 5)
  builder.PrependFloat32(10.0)
  builder.PrependFloat32(20.0)
  builder.PrependFloat32(-50.0)
  builder.PrependFloat32(1.0)
  builder.PrependFloat32(2.0)
  quant1_max_offset = builder.EndVector()

  schema_fb.QuantizationParametersStartScaleVector(builder, 5)
  builder.PrependFloat32(3.0)
  builder.PrependFloat32(4.0)
  builder.PrependFloat32(5.0)
  builder.PrependFloat32(6.0)
  builder.PrependFloat32(7.0)
  quant1_scale_offset = builder.EndVector()

  schema_fb.QuantizationParametersStartZeroPointVector(builder, 5)
  builder.PrependInt64(1)
  builder.PrependInt64(2)
  builder.PrependInt64(3)
  builder.PrependInt64(-1)
  builder.PrependInt64(-2)
  quant1_zero_point_offset = builder.EndVector()

  schema_fb.QuantizationParametersStart(builder)
  schema_fb.QuantizationParametersAddMin(builder, quant1_min_offset)
  schema_fb.QuantizationParametersAddMax(builder, quant1_max_offset)
  schema_fb.QuantizationParametersAddScale(builder, quant1_scale_offset)
  schema_fb.QuantizationParametersAddZeroPoint(builder,
                                               quant1_zero_point_offset)
  quantization1_offset = schema_fb.QuantizationParametersEnd(builder)

  string1_offset = builder.CreateString('constant_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape1_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string1_offset)
  schema_fb.TensorAddShape(builder, shape1_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 1)
  schema_fb.TensorAddQuantization(builder, quantization1_offset)
  tensor1_offset = schema_fb.TensorEnd(builder)

  string2_offset = builder.CreateString('output_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape2_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, string2_offset)
  schema_fb.TensorAddShape(builder, shape2_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 2)
  tensor2_offset = schema_fb.TensorEnd(builder)

  schema_fb.SubGraphStartTensorsVector(builder, 3)
  builder.PrependUOffsetTRelative(tensor2_offset)
  builder.PrependUOffsetTRelative(tensor1_offset)
  builder.PrependUOffsetTRelative(tensor0_offset)
  tensors_offset = builder.EndVector()

  schema_fb.SubGraphStartInputsVector(builder, 1)
  builder.PrependInt32(0)
  inputs_offset = builder.EndVector()

  schema_fb.SubGraphStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  outputs_offset = builder.EndVector()

  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(builder, schema_fb.BuiltinOperator.ADD)
  schema_fb.OperatorCodeAddDeprecatedBuiltinCode(builder,
                                                 schema_fb.BuiltinOperator.ADD)
  schema_fb.OperatorCodeAddVersion(builder, 1)
  code0_offset = schema_fb.OperatorCodeEnd(builder)

  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(builder,
                                       schema_fb.BuiltinOperator.VAR_HANDLE)
  schema_fb.OperatorCodeAddDeprecatedBuiltinCode(
      builder, schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES)
  schema_fb.OperatorCodeAddVersion(builder, 1)
  code1_offset = schema_fb.OperatorCodeEnd(builder)

  schema_fb.ModelStartOperatorCodesVector(builder, 2)
  builder.PrependUOffsetTRelative(code1_offset)
  builder.PrependUOffsetTRelative(code0_offset)
  codes_offset = builder.EndVector()

  schema_fb.OperatorStartInputsVector(builder, 2)
  builder.PrependInt32(0)
  builder.PrependInt32(1)
  op_inputs_offset = builder.EndVector()

  schema_fb.OperatorStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  op_outputs_offset = builder.EndVector()

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddOpcodeIndex(builder, 0)
  schema_fb.OperatorAddInputs(builder, op_inputs_offset)
  schema_fb.OperatorAddOutputs(builder, op_outputs_offset)
  op0_offset = schema_fb.OperatorEnd(builder)

  shared_name = builder.CreateString('var')
  schema_fb.VarHandleOptionsStart(builder)
  schema_fb.VarHandleOptionsAddSharedName(builder, shared_name)
  var_handle_options_offset = schema_fb.VarHandleOptionsEnd(builder)

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddOpcodeIndex(builder, 1)
  schema_fb.OperatorAddBuiltinOptionsType(
      builder, schema_fb.BuiltinOptions.VarHandleOptions)
  schema_fb.OperatorAddBuiltinOptions(builder, var_handle_options_offset)
  op1_offset = schema_fb.OperatorEnd(builder)

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddBuiltinOptionsType(
      builder, schema_fb.BuiltinOptions.VarHandleOptions)
  schema_fb.OperatorAddBuiltinOptions(builder, var_handle_options_offset)
  op2_offset = schema_fb.OperatorEnd(builder)

  schema_fb.SubGraphStartOperatorsVector(builder, 3)
  builder.PrependUOffsetTRelative(op2_offset)
  builder.PrependUOffsetTRelative(op1_offset)
  builder.PrependUOffsetTRelative(op0_offset)
  ops_offset = builder.EndVector()

  string3_offset = builder.CreateString('subgraph_name')
  schema_fb.SubGraphStart(builder)
  schema_fb.SubGraphAddName(builder, string3_offset)
  schema_fb.SubGraphAddTensors(builder, tensors_offset)
  schema_fb.SubGraphAddInputs(builder, inputs_offset)
  schema_fb.SubGraphAddOutputs(builder, outputs_offset)
  schema_fb.SubGraphAddOperators(builder, ops_offset)
  subgraph_offset = schema_fb.SubGraphEnd(builder)

  schema_fb.ModelStartSubgraphsVector(builder, 1)
  builder.PrependUOffsetTRelative(subgraph_offset)
  subgraphs_offset = builder.EndVector()

  signature_key = builder.CreateString('my_key')
  input_tensor_string = builder.CreateString('input_tensor')
  output_tensor_string = builder.CreateString('output_tensor')

  # Signature Inputs
  schema_fb.TensorMapStart(builder)
  schema_fb.TensorMapAddName(builder, input_tensor_string)
  schema_fb.TensorMapAddTensorIndex(builder, 1)
  input_tensor = schema_fb.TensorMapEnd(builder)

  # Signature Outputs
  schema_fb.TensorMapStart(builder)
  schema_fb.TensorMapAddName(builder, output_tensor_string)
  schema_fb.TensorMapAddTensorIndex(builder, 2)
  output_tensor = schema_fb.TensorMapEnd(builder)

  schema_fb.SignatureDefStartInputsVector(builder, 1)
  builder.PrependUOffsetTRelative(input_tensor)
  signature_inputs_offset = builder.EndVector()
  schema_fb.SignatureDefStartOutputsVector(builder, 1)
  builder.PrependUOffsetTRelative(output_tensor)
  signature_outputs_offset = builder.EndVector()

  schema_fb.SignatureDefStart(builder)
  schema_fb.SignatureDefAddSignatureKey(builder, signature_key)
  schema_fb.SignatureDefAddInputs(builder, signature_inputs_offset)
  schema_fb.SignatureDefAddOutputs(builder, signature_outputs_offset)
  signature_offset = schema_fb.SignatureDefEnd(builder)
  schema_fb.ModelStartSignatureDefsVector(builder, 1)
  builder.PrependUOffsetTRelative(signature_offset)
  signature_defs_offset = builder.EndVector()

  string4_offset = builder.CreateString('model_description')
  schema_fb.ModelStart(builder)
  schema_fb.ModelAddVersion(builder, TFLITE_SCHEMA_VERSION)
  schema_fb.ModelAddOperatorCodes(builder, codes_offset)
  schema_fb.ModelAddSubgraphs(builder, subgraphs_offset)
  schema_fb.ModelAddDescription(builder, string4_offset)
  schema_fb.ModelAddBuffers(builder, buffers_offset)
  schema_fb.ModelAddSignatureDefs(builder, signature_defs_offset)
  model_offset = schema_fb.ModelEnd(builder)
  builder.Finish(model_offset)
  model = builder.Output()

  return model


def load_model_from_flatbuffer(flatbuffer_model):
  """Loads a model as a python object from a flatbuffer model."""
  model = schema_fb.Model.GetRootAsModel(flatbuffer_model, 0)
  model = schema_fb.ModelT.InitFromObj(model)
  return model


def build_mock_model():
  """Creates an object containing an example model."""
  model = build_mock_flatbuffer_model()
  return load_model_from_flatbuffer(model)
