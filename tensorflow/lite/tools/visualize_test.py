# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite Python Interface: Sanity check."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from flatbuffers.python import flatbuffers
from tensorflow.lite.tools import schema_py_generated as schema_fb
from tensorflow.lite.tools import visualize
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def BuildMockModel():
  """Creates a flatbuffer object containing an example model."""
  builder = flatbuffers.Builder(1024)

  schema_fb.BufferStart(builder)
  buffer0_offset = schema_fb.BufferEnd(builder)

  schema_fb.BufferStartDataVector(builder, 10)
  builder.PrependUint8(0)
  builder.PrependUint8(1)
  builder.PrependUint8(2)
  builder.PrependUint8(3)
  builder.PrependUint8(4)
  builder.PrependUint8(5)
  builder.PrependUint8(6)
  builder.PrependUint8(7)
  builder.PrependUint8(8)
  builder.PrependUint8(9)
  buffer1_data_offset = builder.EndVector(10)

  schema_fb.BufferStart(builder)
  schema_fb.BufferAddData(builder, buffer1_data_offset)
  buffer1_offset = schema_fb.BufferEnd(builder)

  schema_fb.BufferStart(builder)
  buffer2_offset = schema_fb.BufferEnd(builder)

  schema_fb.ModelStartBuffersVector(builder, 3)
  builder.PrependUOffsetTRelative(buffer2_offset)
  builder.PrependUOffsetTRelative(buffer1_offset)
  builder.PrependUOffsetTRelative(buffer0_offset)
  buffers_offset = builder.EndVector(3)

  name0_offset = builder.CreateString('input_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape0_offset = builder.EndVector(3)
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, name0_offset)
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
  quant1_min_offset = builder.EndVector(5)

  schema_fb.QuantizationParametersStartMaxVector(builder, 5)
  builder.PrependFloat32(10.0)
  builder.PrependFloat32(20.0)
  builder.PrependFloat32(-50.0)
  builder.PrependFloat32(1.0)
  builder.PrependFloat32(2.0)
  quant1_max_offset = builder.EndVector(5)

  schema_fb.QuantizationParametersStartScaleVector(builder, 5)
  builder.PrependFloat32(3.0)
  builder.PrependFloat32(4.0)
  builder.PrependFloat32(5.0)
  builder.PrependFloat32(6.0)
  builder.PrependFloat32(7.0)
  quant1_scale_offset = builder.EndVector(5)

  schema_fb.QuantizationParametersStartZeroPointVector(builder, 5)
  builder.PrependInt64(1)
  builder.PrependInt64(2)
  builder.PrependInt64(3)
  builder.PrependInt64(-1)
  builder.PrependInt64(-2)
  quant1_zero_point_offset = builder.EndVector(5)

  schema_fb.QuantizationParametersStart(builder)
  schema_fb.QuantizationParametersAddMin(builder, quant1_min_offset)
  schema_fb.QuantizationParametersAddMax(builder, quant1_max_offset)
  schema_fb.QuantizationParametersAddScale(builder, quant1_scale_offset)
  schema_fb.QuantizationParametersAddZeroPoint(builder,
                                               quant1_zero_point_offset)
  quantization1_offset = schema_fb.QuantizationParametersEnd(builder)

  name1_offset = builder.CreateString('constant_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape1_offset = builder.EndVector(3)
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, name1_offset)
  schema_fb.TensorAddShape(builder, shape1_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 1)
  schema_fb.TensorAddQuantization(builder, quantization1_offset)
  tensor1_offset = schema_fb.TensorEnd(builder)

  name2_offset = builder.CreateString('output_tensor')
  schema_fb.TensorStartShapeVector(builder, 3)
  builder.PrependInt32(1)
  builder.PrependInt32(2)
  builder.PrependInt32(5)
  shape2_offset = builder.EndVector(3)
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, name2_offset)
  schema_fb.TensorAddShape(builder, shape2_offset)
  schema_fb.TensorAddType(builder, 0)
  schema_fb.TensorAddBuffer(builder, 2)
  tensor2_offset = schema_fb.TensorEnd(builder)

  schema_fb.SubGraphStartTensorsVector(builder, 3)
  builder.PrependUOffsetTRelative(tensor2_offset)
  builder.PrependUOffsetTRelative(tensor1_offset)
  builder.PrependUOffsetTRelative(tensor0_offset)
  tensors_offset = builder.EndVector(3)

  schema_fb.SubGraphStartInputsVector(builder, 1)
  builder.PrependInt32(0)
  inputs_offset = builder.EndVector(1)

  schema_fb.SubGraphStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  outputs_offset = builder.EndVector(1)

  schema_fb.OperatorCodeStart(builder)
  schema_fb.OperatorCodeAddBuiltinCode(builder, schema_fb.BuiltinOperator.ADD)
  schema_fb.OperatorCodeAddVersion(builder, 1)
  code_offset = schema_fb.OperatorCodeEnd(builder)

  schema_fb.ModelStartOperatorCodesVector(builder, 1)
  builder.PrependUOffsetTRelative(code_offset)
  codes_offset = builder.EndVector(1)

  schema_fb.OperatorStartInputsVector(builder, 2)
  builder.PrependInt32(0)
  builder.PrependInt32(1)
  op_inputs_offset = builder.EndVector(2)

  schema_fb.OperatorStartOutputsVector(builder, 1)
  builder.PrependInt32(2)
  op_outputs_offset = builder.EndVector(1)

  schema_fb.OperatorStart(builder)
  schema_fb.OperatorAddOpcodeIndex(builder, 0)
  schema_fb.OperatorAddInputs(builder, op_inputs_offset)
  schema_fb.OperatorAddOutputs(builder, op_outputs_offset)
  op_offset = schema_fb.OperatorEnd(builder)

  schema_fb.SubGraphStartOperatorsVector(builder, 1)
  builder.PrependUOffsetTRelative(op_offset)
  ops_offset = builder.EndVector(1)

  schema_fb.SubGraphStart(builder)
  schema_fb.SubGraphAddTensors(builder, tensors_offset)
  schema_fb.SubGraphAddInputs(builder, inputs_offset)
  schema_fb.SubGraphAddOutputs(builder, outputs_offset)
  schema_fb.SubGraphAddOperators(builder, ops_offset)
  subgraph_offset = schema_fb.SubGraphEnd(builder)

  schema_fb.ModelStartSubgraphsVector(builder, 1)
  builder.PrependUOffsetTRelative(subgraph_offset)
  subgraphs_offset = builder.EndVector(1)

  schema_fb.ModelStart(builder)
  schema_fb.ModelAddBuffers(builder, buffers_offset)
  schema_fb.ModelAddSubgraphs(builder, subgraphs_offset)
  schema_fb.ModelAddOperatorCodes(builder, codes_offset)
  model_offset = schema_fb.ModelEnd(builder)

  builder.Finish(model_offset)
  model_data = builder.Output()

  return model_data


class VisualizeTest(test_util.TensorFlowTestCase):

  def testTensorTypeToName(self):
    self.assertEqual('FLOAT32', visualize.TensorTypeToName(0))

  def testBuiltinCodeToName(self):
    self.assertEqual('HASHTABLE_LOOKUP', visualize.BuiltinCodeToName(10))

  def testFlatbufferToDict(self):
    model_data = BuildMockModel()
    model_dict = visualize.CreateDictFromFlatbuffer(model_data)
    self.assertEqual(0, model_dict['version'])
    self.assertEqual(1, len(model_dict['subgraphs']))
    self.assertEqual(1, len(model_dict['operator_codes']))
    self.assertEqual(3, len(model_dict['buffers']))
    self.assertEqual(3, len(model_dict['subgraphs'][0]['tensors']))
    self.assertEqual(0, model_dict['subgraphs'][0]['tensors'][0]['buffer'])

  def testVisualize(self):
    model_data = BuildMockModel()

    tmp_dir = self.get_temp_dir()
    model_filename = os.path.join(tmp_dir, 'model.tflite')
    with open(model_filename, 'wb') as model_file:
      model_file.write(model_data)
    html_filename = os.path.join(tmp_dir, 'visualization.html')

    visualize.CreateHtmlFile(model_filename, html_filename)

    with open(html_filename, 'r') as html_file:
      html_text = html_file.read()

    # It's hard to test debug output without doing a full HTML parse,
    # but at least sanity check that expected identifiers are present.
    self.assertRegex(
        html_text, re.compile(r'%s' % model_filename, re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text,
                     re.compile(r'input_tensor', re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text,
                     re.compile(r'constant_tensor', re.MULTILINE | re.DOTALL))
    self.assertRegex(html_text, re.compile(r'ADD', re.MULTILINE | re.DOTALL))


if __name__ == '__main__':
  test.main()
