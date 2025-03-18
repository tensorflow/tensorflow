// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/param_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

TEST(ScalarParamWrapperTest, BoolParamTest) {
  ScalarParamWrapper bool_param{"bool_param", true, false};
  Qnn_Param_t bool_qnn_param = QNN_PARAM_INIT;
  bool_param.CloneTo(bool_qnn_param);
  EXPECT_EQ(bool_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(bool_qnn_param.name, "bool_param");
  EXPECT_EQ(bool_qnn_param.scalarParam.dataType, QNN_DATATYPE_BOOL_8);
  EXPECT_EQ(bool_qnn_param.scalarParam.bool8Value, 1);
}

TEST(ScalarParamWrapperTest, Uint8ParamTest) {
  constexpr std::uint8_t value = 255;
  ScalarParamWrapper uint8_param{"uint8_param", value, false};
  Qnn_Param_t uint8_qnn_param = QNN_PARAM_INIT;
  uint8_param.CloneTo(uint8_qnn_param);
  EXPECT_EQ(uint8_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint8_qnn_param.name, "uint8_param");
  EXPECT_EQ(uint8_qnn_param.scalarParam.dataType, QNN_DATATYPE_UINT_8);
  EXPECT_EQ(uint8_qnn_param.scalarParam.uint8Value, value);
}

TEST(ScalarParamWrapperTest, Int8ParamTest) {
  constexpr std::int8_t value = -128;
  ScalarParamWrapper int8_param{"int8_param", value, false};
  Qnn_Param_t int8_qnn_param = QNN_PARAM_INIT;
  int8_param.CloneTo(int8_qnn_param);
  EXPECT_EQ(int8_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int8_qnn_param.name, "int8_param");
  EXPECT_EQ(int8_qnn_param.scalarParam.dataType, QNN_DATATYPE_INT_8);
  EXPECT_EQ(int8_qnn_param.scalarParam.int8Value, value);
}

TEST(ScalarParamWrapperTest, Uint16ParamTest) {
  constexpr std::uint16_t value = 65535;
  ScalarParamWrapper uint16_param{"uint16_param", value, false};
  Qnn_Param_t uint16_qnn_param = QNN_PARAM_INIT;
  uint16_param.CloneTo(uint16_qnn_param);
  EXPECT_EQ(uint16_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint16_qnn_param.name, "uint16_param");
  EXPECT_EQ(uint16_qnn_param.scalarParam.dataType, QNN_DATATYPE_UINT_16);
  EXPECT_EQ(uint16_qnn_param.scalarParam.uint16Value, value);
}

TEST(ScalarParamWrapperTest, Int16ParamTest) {
  constexpr std::int16_t value = -32768;
  ScalarParamWrapper int16_param{"int16_param", value, false};
  Qnn_Param_t int16_qnn_param = QNN_PARAM_INIT;
  int16_param.CloneTo(int16_qnn_param);
  EXPECT_EQ(int16_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int16_qnn_param.name, "int16_param");
  EXPECT_EQ(int16_qnn_param.scalarParam.dataType, QNN_DATATYPE_INT_16);
  EXPECT_EQ(int16_qnn_param.scalarParam.int16Value, value);
}

TEST(ScalarParamWrapperTest, Uint32ParamTest) {
  constexpr std::uint32_t value = 4294967295;
  ScalarParamWrapper uint32_param{"uint32_param", value, false};
  Qnn_Param_t uint32_qnn_param = QNN_PARAM_INIT;
  uint32_param.CloneTo(uint32_qnn_param);
  EXPECT_EQ(uint32_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint32_qnn_param.name, "uint32_param");
  EXPECT_EQ(uint32_qnn_param.scalarParam.dataType, QNN_DATATYPE_UINT_32);
  EXPECT_EQ(uint32_qnn_param.scalarParam.uint32Value, value);
}

TEST(ScalarParamWrapperTest, Int32ParamTest) {
  constexpr std::int32_t value = -2147483648;
  ScalarParamWrapper int32_param{"int32_param", value, false};
  Qnn_Param_t int32_qnn_param = QNN_PARAM_INIT;
  int32_param.CloneTo(int32_qnn_param);
  EXPECT_EQ(int32_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int32_qnn_param.name, "int32_param");
  EXPECT_EQ(int32_qnn_param.scalarParam.dataType, QNN_DATATYPE_INT_32);
  EXPECT_EQ(int32_qnn_param.scalarParam.int32Value, value);
}

TEST(ScalarParamWrapperTest, FloatParamTest) {
  constexpr float value = 3.14f;
  ScalarParamWrapper float_param{"float_param", value, false};
  Qnn_Param_t float_qnn_param = QNN_PARAM_INIT;
  float_param.CloneTo(float_qnn_param);
  EXPECT_EQ(float_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(float_qnn_param.name, "float_param");
  EXPECT_EQ(float_qnn_param.scalarParam.dataType, QNN_DATATYPE_FLOAT_32);
  EXPECT_FLOAT_EQ(float_qnn_param.scalarParam.floatValue, value);
}

TEST(ScalarParamWrapperTest, QuantizedBoolParamTest) {
  ScalarParamWrapper bool_quant_param{"bool_quant_param", true, true};
  Qnn_Param_t bool_quant_qnn_param = QNN_PARAM_INIT;
  bool_quant_param.CloneTo(bool_quant_qnn_param);
  EXPECT_EQ(bool_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(bool_quant_qnn_param.name, "bool_quant_param");
  EXPECT_EQ(bool_quant_qnn_param.scalarParam.dataType, QNN_DATATYPE_BOOL_8);
  EXPECT_EQ(bool_quant_qnn_param.scalarParam.bool8Value, 1);
}

TEST(ScalarParamWrapperTest, QuantizedUint8ParamTest) {
  constexpr std::uint8_t value = 255;
  ScalarParamWrapper uint8_quant_param{"uint8_quant_param", value, true};
  Qnn_Param_t uint8_quant_qnn_param = QNN_PARAM_INIT;
  uint8_quant_param.CloneTo(uint8_quant_qnn_param);
  EXPECT_EQ(uint8_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint8_quant_qnn_param.name, "uint8_quant_param");
  EXPECT_EQ(uint8_quant_qnn_param.scalarParam.dataType,
            QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(uint8_quant_qnn_param.scalarParam.uint8Value, value);
}

TEST(ScalarParamWrapperTest, QuantizedInt8ParamTest) {
  constexpr std::int8_t value = -128;
  ScalarParamWrapper int8_quant_param{"int8_quant_param", value, true};
  Qnn_Param_t int8_quant_qnn_param = QNN_PARAM_INIT;
  int8_quant_param.CloneTo(int8_quant_qnn_param);
  EXPECT_EQ(int8_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int8_quant_qnn_param.name, "int8_quant_param");
  EXPECT_EQ(int8_quant_qnn_param.scalarParam.dataType,
            QNN_DATATYPE_SFIXED_POINT_8);
  EXPECT_EQ(int8_quant_qnn_param.scalarParam.int8Value, value);
}

TEST(ScalarParamWrapperTest, QuantizedUint16ParamTest) {
  constexpr std::uint16_t value = 65535;
  ScalarParamWrapper uint16_quant_param{"uint16_quant_param", value, true};
  Qnn_Param_t uint16_quant_qnn_param = QNN_PARAM_INIT;
  uint16_quant_param.CloneTo(uint16_quant_qnn_param);
  EXPECT_EQ(uint16_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint16_quant_qnn_param.name, "uint16_quant_param");
  EXPECT_EQ(uint16_quant_qnn_param.scalarParam.dataType,
            QNN_DATATYPE_UFIXED_POINT_16);
  EXPECT_EQ(uint16_quant_qnn_param.scalarParam.uint16Value, value);
}

TEST(ScalarParamWrapperTest, QuantizedInt16ParamTest) {
  constexpr std::int16_t value = -32768;
  ScalarParamWrapper int16_quant_param{"int16_quant_param", value, true};
  Qnn_Param_t int16_quant_qnn_param = QNN_PARAM_INIT;
  int16_quant_param.CloneTo(int16_quant_qnn_param);
  EXPECT_EQ(int16_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int16_quant_qnn_param.name, "int16_quant_param");
  EXPECT_EQ(int16_quant_qnn_param.scalarParam.dataType,
            QNN_DATATYPE_SFIXED_POINT_16);
  EXPECT_EQ(int16_quant_qnn_param.scalarParam.int16Value, value);
}

TEST(ScalarParamWrapperTest, QuantizedUint32ParamTest) {
  constexpr std::uint32_t value = 4294967295;
  ScalarParamWrapper uint32_quant_param{"uint32_quant_param", value, true};
  Qnn_Param_t uint32_quant_qnn_param = QNN_PARAM_INIT;
  uint32_quant_param.CloneTo(uint32_quant_qnn_param);
  EXPECT_EQ(uint32_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(uint32_quant_qnn_param.name, "uint32_quant_param");
  EXPECT_EQ(uint32_quant_qnn_param.scalarParam.dataType,
            QNN_DATATYPE_UFIXED_POINT_32);
  EXPECT_EQ(uint32_quant_qnn_param.scalarParam.uint32Value, value);
}

TEST(ScalarParamWrapperTest, QuantizedInt32ParamTest) {
  constexpr std::int32_t value = -2147483648;
  ScalarParamWrapper int32_quant_param{"int32_quant_param", value, true};
  Qnn_Param_t int32_quant_qnn_param = QNN_PARAM_INIT;
  int32_quant_param.CloneTo(int32_quant_qnn_param);
  EXPECT_EQ(int32_quant_qnn_param.paramType, QNN_PARAMTYPE_SCALAR);
  EXPECT_EQ(int32_quant_qnn_param.name, "int32_quant_param");
  EXPECT_EQ(int32_quant_qnn_param.scalarParam.dataType,
            QNN_DATATYPE_SFIXED_POINT_32);
  EXPECT_EQ(int32_quant_qnn_param.scalarParam.int32Value, value);
}

TEST(ParamWrapperTest, TensorParamTest) {
  std::vector<std::uint32_t> dummy_dims = {1, 1, 3};
  std::vector<std::uint8_t> data = {1, 2, 3};
  void* data_ptr = reinterpret_cast<void*>(data.data());

  const auto data_size =
      std::accumulate(dummy_dims.begin(), dummy_dims.end(),
                      sizeof(decltype(data)::value_type), std::multiplies<>());

  TensorWrapper tensor_wrapper{0,
                               QNN_TENSOR_TYPE_STATIC,
                               QNN_DATATYPE_UFIXED_POINT_8,
                               QuantizeParamsWrapperVariant(),
                               dummy_dims,
                               static_cast<uint32_t>(data_size),
                               data_ptr};

  TensorParamWrapper tensor_param{"tensor_param", tensor_wrapper};

  Qnn_Param_t qnn_tensor_param = QNN_PARAM_INIT;
  tensor_param.CloneTo(qnn_tensor_param);
  EXPECT_EQ(qnn_tensor_param.paramType, QNN_PARAMTYPE_TENSOR);
  EXPECT_EQ(qnn_tensor_param.name, "tensor_param");

  Qnn_Tensor_t& ref = qnn_tensor_param.tensorParam;
  EXPECT_EQ(ref.v2.id, 0);
  EXPECT_EQ(ref.v2.type, QNN_TENSOR_TYPE_STATIC);
  EXPECT_EQ(ref.v2.dataFormat, QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER);
  EXPECT_EQ(ref.v2.dataType, QNN_DATATYPE_UFIXED_POINT_8);
  EXPECT_EQ(ref.v2.quantizeParams.encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(ref.v2.rank, dummy_dims.size());
  for (size_t i = 0; i < ref.v2.rank; i++) {
    EXPECT_EQ(ref.v2.dimensions[i], dummy_dims[i]);
  }
  EXPECT_EQ(ref.v2.memType, QNN_TENSORMEMTYPE_RAW);
  EXPECT_EQ(ref.v2.clientBuf.dataSize, data_size);
  const auto* ref_data =
      reinterpret_cast<const std::uint8_t*>(ref.v2.clientBuf.data);
  for (size_t i = 0; i < data.size(); i++) {
    EXPECT_EQ(ref_data[i], data[i]);
  }
}
}  // namespace
}  // namespace qnn
