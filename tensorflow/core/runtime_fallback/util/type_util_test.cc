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

#include "tensorflow/core/runtime_fallback/util/type_util.h"

#include "tensorflow/core/platform/test.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
namespace {

// Mapping between TFRT DType and TF DataType.
struct TfrtDTypeAndTfDataType {
  tfrt::DType tfrt_dtype;
  DataType tf_dtype;
};

class GetDataTypeTest
    : public ::testing::TestWithParam<TfrtDTypeAndTfDataType> {};

INSTANTIATE_TEST_SUITE_P(
    AllDTypes, GetDataTypeTest,
    ::testing::Values(
        TfrtDTypeAndTfDataType{tfrt::DType::UI8, DataType::DT_UINT8},
        TfrtDTypeAndTfDataType{tfrt::DType::UI16, DataType::DT_UINT16},
        TfrtDTypeAndTfDataType{tfrt::DType::UI32, DataType::DT_UINT32},
        TfrtDTypeAndTfDataType{tfrt::DType::UI64, DataType::DT_UINT64},
        TfrtDTypeAndTfDataType{tfrt::DType::I1, DataType::DT_BOOL},
        TfrtDTypeAndTfDataType{tfrt::DType::I8, DataType::DT_INT8},
        TfrtDTypeAndTfDataType{tfrt::DType::I16, DataType::DT_INT16},
        TfrtDTypeAndTfDataType{tfrt::DType::I32, DataType::DT_INT32},
        TfrtDTypeAndTfDataType{tfrt::DType::I64, DataType::DT_INT64},
        TfrtDTypeAndTfDataType{tfrt::DType::F16, DataType::DT_HALF},
        TfrtDTypeAndTfDataType{tfrt::DType::BF16, DataType::DT_BFLOAT16},
        TfrtDTypeAndTfDataType{tfrt::DType::F32, DataType::DT_FLOAT},
        TfrtDTypeAndTfDataType{tfrt::DType::F64, DataType::DT_DOUBLE},
        TfrtDTypeAndTfDataType{tfrt::DType::String, DataType::DT_STRING},
        TfrtDTypeAndTfDataType{tfrt::DType::Complex64, DataType::DT_COMPLEX64},
        TfrtDTypeAndTfDataType{tfrt::DType::Complex128,
                               DataType::DT_COMPLEX128},
        TfrtDTypeAndTfDataType{tfrt::DType::Variant, DataType::DT_VARIANT},
        TfrtDTypeAndTfDataType{tfrt::DType::QUI8, DataType::DT_QUINT8},
        TfrtDTypeAndTfDataType{tfrt::DType::QUI16, DataType::DT_QUINT16},
        TfrtDTypeAndTfDataType{tfrt::DType::QI8, DataType::DT_QINT8},
        TfrtDTypeAndTfDataType{tfrt::DType::QI16, DataType::DT_QINT16},
        TfrtDTypeAndTfDataType{tfrt::DType::QI32, DataType::DT_QINT32}));

TEST_P(GetDataTypeTest, GetTfDataTypeOk) {
  EXPECT_EQ(GetTfDataType(tfrt::DType(GetParam().tfrt_dtype)),
            GetParam().tf_dtype);
}

TEST_P(GetDataTypeTest, GetTfrtDataTypeOk) {
  EXPECT_EQ(GetTfrtDtype(GetParam().tf_dtype),
            tfrt::DType(GetParam().tfrt_dtype));
}

}  // namespace
}  // namespace tfd
}  // namespace tensorflow
