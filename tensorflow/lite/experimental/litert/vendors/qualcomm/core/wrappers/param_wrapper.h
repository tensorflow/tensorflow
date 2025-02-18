//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_PARAM_WRAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_PARAM_WRAPPER_H_

#include <type_traits>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

class ScalarParamWrapper {
 public:
  template <typename T>
  explicit ScalarParamWrapper(const char* name, const T data,
                              const bool is_quant) {
    qnn_param_.name = name;
    qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
    if constexpr (std::is_same_v<T, bool>) {
      qnn_param_.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
      qnn_param_.scalarParam.bool8Value = data;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      qnn_param_.scalarParam.dataType =
          is_quant ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_UINT_8;
      qnn_param_.scalarParam.uint8Value = data;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
      qnn_param_.scalarParam.dataType =
          is_quant ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_INT_8;
      qnn_param_.scalarParam.int8Value = data;
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
      qnn_param_.scalarParam.dataType =
          is_quant ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UINT_16;
      qnn_param_.scalarParam.uint16Value = data;
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
      qnn_param_.scalarParam.dataType =
          is_quant ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_INT_16;
      qnn_param_.scalarParam.int16Value = data;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      qnn_param_.scalarParam.dataType =
          is_quant ? QNN_DATATYPE_UFIXED_POINT_32 : QNN_DATATYPE_UINT_32;
      qnn_param_.scalarParam.uint32Value = data;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      qnn_param_.scalarParam.dataType =
          is_quant ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_INT_32;
      qnn_param_.scalarParam.int32Value = data;
    } else if constexpr (std::is_same_v<T, float>) {
      qnn_param_.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
      qnn_param_.scalarParam.floatValue = data;
    } else {
      // TODO: error log
    }
  }

  void CloneTo(Qnn_Param_t& dst) const;

 private:
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

class TensorParamWrapper {
 public:
  explicit TensorParamWrapper(const char* name, const TensorWrapper& tensor);

  void CloneTo(Qnn_Param_t& dst) const;

 private:
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_PARAM_WRAPPER_H_
