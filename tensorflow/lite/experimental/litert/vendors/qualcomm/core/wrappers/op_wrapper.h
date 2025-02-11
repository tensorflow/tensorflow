//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_OP_WRAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_OP_WRAPPER_H_

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/param_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

class OpWrapper final {
 public:
  explicit OpWrapper(std::string name, const char* op_type);

  OpWrapper(const OpWrapper& other);

  OpWrapper(OpWrapper&& other);

  ~OpWrapper();

  void AddInputTensor(const TensorWrapper& tensor);

  void AddOutputTensor(const TensorWrapper& tensor);

  template <typename T>
  void AddScalarParam(const char* name, const T data,
                      const bool is_quant = false) {
    ScalarParamWrapper param_wrapper(name, data, is_quant);

    auto& back = params_.emplace_back();
    param_wrapper.CloneTo(back);

    qnn_op_.v1.numOfParams = params_.size();
    qnn_op_.v1.params = params_.data();
  }

  void AddTensorParam(const char* name, const TensorWrapper& tensor);

  const Qnn_OpConfig_t& GetOpConfig() const;

 private:
  Qnn_OpConfig_t qnn_op_ = QNN_OPCONFIG_INIT;
  std::string name_{};  // human readable name
  std::vector<Qnn_Param_t> params_{};
  std::vector<Qnn_Tensor_t> input_tensors_{};
  std::vector<Qnn_Tensor_t> output_tensors_{};
};

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_OP_WRAPPER_H_
