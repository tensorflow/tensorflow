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
#include "tensorflow/core/runtime_fallback/util/attr_util.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tfrt/bef/bef_encoding.h"  // from @tf_runtime
#include "tfrt/bef_converter/bef_attr_encoder.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::EqualsProto;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

std::unique_ptr<tfrt::HostContext> CreateTestHostContext() {
  return std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic&) {}, tfrt::CreateMallocAllocator(),
      tfrt::CreateSingleThreadedWorkQueue());
}

// Mapping between DType and string.
struct DataTypeAndString {
  std::string str_val;
  DataType dtype;
};

class ParseTfDataTypeTest : public ::testing::TestWithParam<DataTypeAndString> {
};

INSTANTIATE_TEST_SUITE_P(
    AllDTypes, ParseTfDataTypeTest,
    ::testing::Values(DataTypeAndString{"DT_INT8", DataType::DT_INT8},
                      DataTypeAndString{"DT_INT32", DataType::DT_INT32},
                      DataTypeAndString{"DT_INT64", DataType::DT_INT64},
                      DataTypeAndString{"DT_HALF", DataType::DT_HALF},
                      DataTypeAndString{"DT_FLOAT", DataType::DT_FLOAT},
                      DataTypeAndString{"DT_DOUBLE", DataType::DT_DOUBLE}));

TEST_P(ParseTfDataTypeTest, Ok) {
  DataType data_type;
  ASSERT_EQ(ParseTfDataType(GetParam().str_val, &data_type), OkStatus());
  EXPECT_EQ(data_type, GetParam().dtype);
}

TEST(ParseTfDataTypeTest, ReturnsInvalidArgument) {
  DataType data_type;
  EXPECT_EQ(ParseTfDataType("DT_BFLOAT16_REF", &data_type),
            errors::InvalidArgument(
                "Unsupported dtype, DT_BFLOAT16_REF in ParseTfDataType."));
}

TEST(UtilsTest, ToAbslStringViewOk) {
  std::string str("Tensorflow Runtime");
  tfrt::string_view str_view(str);
  EXPECT_EQ(ToAbslStringView(str_view), str);
}

// Mapping between OpAttrType and DType
struct OpAttrTypeAndDType {
  tfrt::OpAttrType op_attr_type;
  DataType dtype;
};

class OpAttrTypeDTypeTest
    : public ::testing::TestWithParam<OpAttrTypeAndDType> {};

INSTANTIATE_TEST_SUITE_P(
    AllDTypes, OpAttrTypeDTypeTest,
    ::testing::Values(
        OpAttrTypeAndDType{tfrt::OpAttrType::BOOL, DataType::DT_BOOL},
        OpAttrTypeAndDType{tfrt::OpAttrType::UI8, DataType::DT_UINT8},
        OpAttrTypeAndDType{tfrt::OpAttrType::I8, DataType::DT_INT8},
        OpAttrTypeAndDType{tfrt::OpAttrType::I16, DataType::DT_INT16},
        OpAttrTypeAndDType{tfrt::OpAttrType::UI16, DataType::DT_UINT16},
        OpAttrTypeAndDType{tfrt::OpAttrType::I32, DataType::DT_INT32},
        OpAttrTypeAndDType{tfrt::OpAttrType::UI32, DataType::DT_UINT32},
        OpAttrTypeAndDType{tfrt::OpAttrType::I64, DataType::DT_INT64},
        OpAttrTypeAndDType{tfrt::OpAttrType::UI64, DataType::DT_UINT64},
        OpAttrTypeAndDType{tfrt::OpAttrType::BF16, DataType::DT_BFLOAT16},
        OpAttrTypeAndDType{tfrt::OpAttrType::F16, DataType::DT_HALF},
        OpAttrTypeAndDType{tfrt::OpAttrType::F32, DataType::DT_FLOAT},
        OpAttrTypeAndDType{tfrt::OpAttrType::F64, DataType::DT_DOUBLE},
        OpAttrTypeAndDType{tfrt::OpAttrType::COMPLEX64, DataType::DT_COMPLEX64},
        OpAttrTypeAndDType{tfrt::OpAttrType::COMPLEX128,
                           DataType::DT_COMPLEX128},
        OpAttrTypeAndDType{tfrt::OpAttrType::CHAR, DataType::DT_STRING},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_QUI8,
                           DataType::DT_QUINT8},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_QUI16,
                           DataType::DT_QUINT16},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_QI8,
                           DataType::DT_QINT8},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_QI16,
                           DataType::DT_QINT16},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_QI32,
                           DataType::DT_QINT32},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_RESOURCE,
                           DataType::DT_RESOURCE},
        OpAttrTypeAndDType{tfrt::OpAttrType::UNSUPPORTED_VARIANT,
                           DataType::DT_VARIANT}));

TEST_P(OpAttrTypeDTypeTest, ToTfDataTypeOk) {
  EXPECT_EQ(ConvertToTfDataType(GetParam().op_attr_type), GetParam().dtype);
}

TEST_P(OpAttrTypeDTypeTest, FromTfDataTypeOk) {
  EXPECT_EQ(ConvertFromTfDataType(GetParam().dtype), GetParam().op_attr_type);
}

TEST(OpAttrTypeDTypeTest, DeathUnsupportedDType) {
  EXPECT_DEATH(ConvertFromTfDataType(DataType::DT_RESOURCE_REF), "");
}

// Mapping between tfrt::DType and tensorflow::DataType
struct TfrtDTypeAndTensorflowDType {
  tfrt::DType tfrt_dtype;
  DataType dtype;
};

class TfrtToTensorflowDTypeTest
    : public ::testing::TestWithParam<TfrtDTypeAndTensorflowDType> {};

INSTANTIATE_TEST_SUITE_P(
    AllDTypes, TfrtToTensorflowDTypeTest,
    ::testing::Values(
        TfrtDTypeAndTensorflowDType{tfrt::DType::I1, DataType::DT_BOOL},
        TfrtDTypeAndTensorflowDType{tfrt::DType::UI8, DataType::DT_UINT8},
        TfrtDTypeAndTensorflowDType{tfrt::DType::I8, DataType::DT_INT8},
        TfrtDTypeAndTensorflowDType{tfrt::DType::I16, DataType::DT_INT16},
        TfrtDTypeAndTensorflowDType{tfrt::DType::UI16, DataType::DT_UINT16},
        TfrtDTypeAndTensorflowDType{tfrt::DType::I32, DataType::DT_INT32},
        TfrtDTypeAndTensorflowDType{tfrt::DType::UI32, DataType::DT_UINT32},
        TfrtDTypeAndTensorflowDType{tfrt::DType::I64, DataType::DT_INT64},
        TfrtDTypeAndTensorflowDType{tfrt::DType::UI64, DataType::DT_UINT64},
        TfrtDTypeAndTensorflowDType{tfrt::DType::BF16, DataType::DT_BFLOAT16},
        TfrtDTypeAndTensorflowDType{tfrt::DType::F16, DataType::DT_HALF},
        TfrtDTypeAndTensorflowDType{tfrt::DType::F32, DataType::DT_FLOAT},
        TfrtDTypeAndTensorflowDType{tfrt::DType::F64, DataType::DT_DOUBLE},
        TfrtDTypeAndTensorflowDType{tfrt::DType::Complex64,
                                    DataType::DT_COMPLEX64},
        TfrtDTypeAndTensorflowDType{tfrt::DType::Complex128,
                                    DataType::DT_COMPLEX128},
        TfrtDTypeAndTensorflowDType{tfrt::DType::String, DataType::DT_STRING},
        TfrtDTypeAndTensorflowDType{tfrt::DType::QUI8, DataType::DT_QUINT8},
        TfrtDTypeAndTensorflowDType{tfrt::DType::QUI16, DataType::DT_QUINT16},
        TfrtDTypeAndTensorflowDType{tfrt::DType::QI8, DataType::DT_QINT8},
        TfrtDTypeAndTensorflowDType{tfrt::DType::QI16, DataType::DT_QINT16},
        TfrtDTypeAndTensorflowDType{tfrt::DType::QI32, DataType::DT_QINT32},
        TfrtDTypeAndTensorflowDType{tfrt::DType::Resource,
                                    DataType::DT_RESOURCE},
        TfrtDTypeAndTensorflowDType{tfrt::DType::Variant,
                                    DataType::DT_VARIANT}));

TEST_P(TfrtToTensorflowDTypeTest, BefAttrTypeToTfDataTypeOk) {
  EXPECT_EQ(ConvertBefAttrTypeToTfDataType(GetParam().tfrt_dtype),
            GetParam().dtype);
}

TEST_P(TfrtToTensorflowDTypeTest, TfDataTypeTpBefAttrTypeOk) {
  EXPECT_EQ(ConvertTfDataTypeToBefAttrType(GetParam().dtype),
            GetParam().tfrt_dtype);
}

TEST(TfrtToTensorflowDTypeTest, DeathUnsupportedDType) {
  EXPECT_DEATH(ConvertTfDataTypeToBefAttrType(DataType::DT_RESOURCE_REF), "");
}

TEST(UtilsTest, ParseTensorAttrValueOk) {
  tensorflow::Tensor tensor;
  std::string tensor_str = R"pb(dtype: DT_INT32
                                tensor_shape {
                                  dim { size: 2 }
                                  dim { size: 2 }
                                }
                                int_val: 1
                                int_val: 1
                                int_val: 1
                                int_val: 1)pb";
  ASSERT_EQ(ParseTensorAttrValue(tensor_str, &tensor), OkStatus());
  EXPECT_EQ(tensor.dtype(), DT_INT32);
  EXPECT_EQ(tensor.NumElements(), 4);
}

TEST(UtilsTest, ParseTensorAttrValueReturnsInvalidArgument) {
  tensorflow::Tensor tensor;
  std::string tensor_str = R"pb(foobar)pb";
  EXPECT_EQ(
      ParseTensorAttrValue(tensor_str, &tensor),
      errors::InvalidArgument("Could not parse tensor value from \"foobar\""));
}

TEST(UtilsTest, ParseTensorShapeAttrValueOk) {
  std::vector<int64_t> dims;
  ASSERT_THAT(ParseTensorShapeAttrValue("[1,2,3]", &dims), OkStatus());
  EXPECT_THAT(dims, ElementsAre(Eq(1), Eq(2), Eq(3)));
}

TEST(UtilsTest, ParseTensorShapeAttrValueInvalidArgument) {
  std::vector<int64_t> dims;
  EXPECT_EQ(
      ParseTensorShapeAttrValue("foobar", &dims),
      errors::InvalidArgument("Tensor shape attribute must be a string of the "
                              "form [1,2...], instead got \"foobar\""));
}

TEST(UtilsTest, ParseTensorShapeAttrValueInvalidArgumentEmptyString) {
  std::vector<int64_t> dims;
  EXPECT_EQ(ParseTensorShapeAttrValue("", &dims),
            errors::InvalidArgument("Tensor shape attribute must be a string "
                                    "of the form [1,2...], instead got \"\""));
}

TEST(UtilsTest, ParseBoolAttrValueOk) {
  bool bool_val;
  ASSERT_THAT(ParseBoolAttrValue("false", &bool_val), OkStatus());
  EXPECT_FALSE(bool_val);

  ASSERT_THAT(ParseBoolAttrValue("true", &bool_val), OkStatus());
  EXPECT_TRUE(bool_val);
}

TEST(UtilsTest, ParseBoolAttrValueInvalidArgument) {
  bool bool_val;
  EXPECT_EQ(ParseBoolAttrValue("foobar", &bool_val),
            errors::InvalidArgument("Could not parse bool from \"foobar\""));
}

TEST(UtilsTest, ParseIntAttrValueOk) {
  int64_t int_val;
  ASSERT_THAT(ParseIntAttrValue("42", &int_val), OkStatus());
  EXPECT_EQ(int_val, 42);
}

TEST(UtilsTest, ParseIntAttrValueInvalidArgument) {
  int64_t int_val;
  EXPECT_EQ(ParseIntAttrValue("foobar", &int_val),
            errors::InvalidArgument("Could not parse int from \"foobar\""));
}

TEST(UtilsTest, IsUnusedAttributeOk) {
  EXPECT_TRUE(IsUnusedAttribute("result_segment_sizes"));
  EXPECT_TRUE(IsUnusedAttribute("operand_segment_sizes"));
  EXPECT_TRUE(IsUnusedAttribute("_tf_data_function"));
  EXPECT_FALSE(IsUnusedAttribute("device"));
}

TEST(UtilsTest, FillAttrValueMapOk) {
  tfrt::OpAttrs attrs;
  attrs.SetArray("shape", tfrt::ArrayRef<int64_t>{2, 2});
  attrs.SetArray("values", tfrt::ArrayRef<float>{2});
  attrs.SetArray("flags", tfrt::ArrayRef<bool>{false, true});

  attrs.Set<bool>("transpose_a", false);
  attrs.Set<bool>("transpose_b", true);
  attrs.Set<int64_t>("result_segment_sizes", 2);  // unused
  attrs.Set<float>("foo", 2);
  attrs.Set<int64_t>("bar", 2);

  AttrValueMap map;
  auto host_context = CreateTestHostContext();

  // False implies success for errorToBool.
  ASSERT_FALSE(llvm::errorToBool(
      FillAttrValueMap(attrs.freeze(), host_context.get(), &map)));

  EXPECT_THAT(
      map,
      UnorderedElementsAre(
          Pair(Eq("shape"), EqualsProto(R"pb(list { i: 2 i: 2 })pb")),
          Pair(Eq("values"), EqualsProto(R"pb(list { f: 2 })pb")),
          Pair(Eq("flags"), EqualsProto(R"pb(list { b: false b: true })pb")),
          Pair(Eq("transpose_a"), EqualsProto(R"pb(b: false)pb")),
          Pair(Eq("transpose_b"), EqualsProto(R"pb(b: true)pb")),
          Pair(Eq("foo"), EqualsProto(R"pb(f: 2)pb")),
          Pair(Eq("bar"), EqualsProto(R"pb(i: 2)pb"))));
}

}  // namespace
}  // namespace tfd
}  // namespace tensorflow
