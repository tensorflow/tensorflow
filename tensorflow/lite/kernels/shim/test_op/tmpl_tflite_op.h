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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TFLITE_OP_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TFLITE_OP_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/test_op/tmpl_op.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace custom {
namespace tmpl_tflite_op {

using ::tensorflow::DT_FLOAT;
using ::tensorflow::DT_INT32;
using ::tensorflow::DT_INT64;
using ::tflite::shim::OpKernelShim;
using ::tflite::shim::Runtime;
using ::tflite::shim::TmplOp;

template <Runtime Rt>
class OpWrapper : public OpKernelShim<OpWrapper, Rt> {
  // Atype: int32 or float
  // Btype: int32 or int64
  using TmplOpType =
      std::variant<TmplOp<Rt, int32_t, int32_t>, TmplOp<Rt, int32_t, int64_t>,
                   TmplOp<Rt, float, int32_t>, TmplOp<Rt, float, int64_t>>;
  using TmplOpType0 = typename std::variant_alternative<0, TmplOpType>::type;

 public:
  using typename OpKernelShim<OpWrapper, Rt>::InitContext;
  using typename OpKernelShim<OpWrapper, Rt>::InvokeContext;
  using typename OpKernelShim<OpWrapper, Rt>::ShapeInferenceContext;
  OpWrapper() = default;

  // These two char*s should be copied from the wrapped op.
  static constexpr char kOpName[] = "TemplatizedOperation";
  static constexpr char kDoc[] = R"doc(
      Description:
        Templatized op for testing and demonstration purposes.

      Attrs
        AType: The type for input0
        BType: The type for input1
      Inputs
        in0: AType, shape=[] - A scalar input
        in1: BType, shape=[] - A scalar input
      Outputs
        out0: int, shape=[] - first output
      )doc";

  // For the static methods, they shouldn't change based on the types.
  static std::vector<std::string> Attrs() { return TmplOpType0::Attrs(); }
  static std::vector<std::string> Inputs() { return TmplOpType0::Inputs(); }
  static std::vector<std::string> Outputs() { return TmplOpType0::Outputs(); }
  static absl::Status ShapeInference(ShapeInferenceContext* context) {
    return TmplOpType0::ShapeInference(context);
  }

  // Init should create the correctly typed wrapped object.
  absl::Status Init(InitContext* context) {
    int64_t datatype_a, datatype_b;
    SH_RETURN_IF_ERROR(context->GetAttr("AType", &datatype_a));
    SH_RETURN_IF_ERROR(context->GetAttr("BType", &datatype_b));
    if (datatype_a == DT_INT32 && datatype_b == DT_INT32) {
      op_ = std::make_unique<TmplOpType>(TmplOp<Rt, int32_t, int32_t>());
      type_num_ = 0;
      return std::get<0>(*op_).Init(context);
    } else if (datatype_a == DT_INT32 && datatype_b == DT_INT64) {
      op_ = std::make_unique<TmplOpType>(TmplOp<Rt, int32_t, int64_t>());
      type_num_ = 1;
      return std::get<1>(*op_).Init(context);
    } else if (datatype_a == DT_FLOAT && datatype_b == DT_INT32) {
      op_ = std::make_unique<TmplOpType>(TmplOp<Rt, float, int32_t>());
      type_num_ = 2;
      return std::get<2>(*op_).Init(context);
    } else if (datatype_a == DT_FLOAT && datatype_b == DT_INT64) {
      op_ = std::make_unique<TmplOpType>(TmplOp<Rt, float, int64_t>());
      type_num_ = 3;
      return std::get<3>(*op_).Init(context);
    }
    return absl::InvalidArgumentError("Attribute is of wrong type.");
  }

  // Call invoke on the created wrapped object.
  absl::Status Invoke(InvokeContext* context) {
    if (type_num_ == 0) {
      return std::get<0>(*op_).Invoke(context);
    } else if (type_num_ == 1) {
      return std::get<1>(*op_).Invoke(context);
    } else if (type_num_ == 2) {
      return std::get<2>(*op_).Invoke(context);
    } else if (type_num_ == 3) {
      return std::get<3>(*op_).Invoke(context);
    }
    return absl::InternalError("Unknown type.");
  }

 protected:
  std::unique_ptr<TmplOpType> op_;
  int type_num_;
};

}  // namespace tmpl_tflite_op

// Add TmplOp to the resolver
void AddTmplOp(MutableOpResolver* resolver);

// Creates and returns the op kernel
TfLiteRegistration* Register_TMPL_OP();

// The name of the op
const char* OpName_TMPL_OP();

}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TEST_OP_TMPL_TFLITE_OP_H_
