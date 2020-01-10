#include "tensorflow/core/framework/types.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {

TEST(TypesTest, DeviceTypeName) {
  EXPECT_EQ("CPU", DeviceTypeString(DeviceType(DEVICE_CPU)));
  EXPECT_EQ("GPU", DeviceTypeString(DeviceType(DEVICE_GPU)));
}

TEST(TypesTest, kDataTypeRefOffset) {
  // Basic sanity check
  EXPECT_EQ(DT_FLOAT + kDataTypeRefOffset, DT_FLOAT_REF);

  // Use the meta-data provided by proto2 to iterate through the basic
  // types and validate that adding kDataTypeRefOffset gives the
  // corresponding reference type.
  const auto* enum_descriptor = DataType_descriptor();
  int e = DataType_MIN;
  if (e == DT_INVALID) ++e;
  int e_ref = e + kDataTypeRefOffset;
  EXPECT_FALSE(DataType_IsValid(e_ref - 1))
      << "Reference enum "
      << enum_descriptor->FindValueByNumber(e_ref - 1)->name()
      << " without corresponding base enum with value " << e - 1;
  for (;
       DataType_IsValid(e) && DataType_IsValid(e_ref) && e_ref <= DataType_MAX;
       ++e, ++e_ref) {
    string enum_name = enum_descriptor->FindValueByNumber(e)->name();
    string enum_ref_name = enum_descriptor->FindValueByNumber(e_ref)->name();
    EXPECT_EQ(enum_name + "_REF", enum_ref_name)
        << enum_name << "_REF should have value " << e_ref << " not "
        << enum_ref_name;
    // Validate DataTypeString() as well.
    DataType dt_e = static_cast<DataType>(e);
    DataType dt_e_ref = static_cast<DataType>(e_ref);
    EXPECT_EQ(DataTypeString(dt_e) + "_ref", DataTypeString(dt_e_ref));

    // Test DataTypeFromString reverse conversion
    DataType dt_e2, dt_e2_ref;
    EXPECT_TRUE(DataTypeFromString(DataTypeString(dt_e), &dt_e2));
    EXPECT_EQ(dt_e, dt_e2);
    EXPECT_TRUE(DataTypeFromString(DataTypeString(dt_e_ref), &dt_e2_ref));
    EXPECT_EQ(dt_e_ref, dt_e2_ref);
  }
  ASSERT_FALSE(DataType_IsValid(e))
      << "Should define " << enum_descriptor->FindValueByNumber(e)->name()
      << "_REF to be " << e_ref;
  ASSERT_FALSE(DataType_IsValid(e_ref))
      << "Extra reference enum "
      << enum_descriptor->FindValueByNumber(e_ref)->name()
      << " without corresponding base enum with value " << e;
  ASSERT_LT(DataType_MAX, e_ref) << "Gap in reference types, missing value for "
                                 << e_ref;

  // Make sure there are no enums defined after the last regular type before
  // the first reference type.
  for (; e < DataType_MIN + kDataTypeRefOffset; ++e) {
    EXPECT_FALSE(DataType_IsValid(e))
        << "Discontinuous enum value "
        << enum_descriptor->FindValueByNumber(e)->name() << " = " << e;
  }
}

TEST(TypesTest, DataTypeFromString) {
  DataType dt;
  ASSERT_TRUE(DataTypeFromString("int32", &dt));
  EXPECT_EQ(DT_INT32, dt);
  ASSERT_TRUE(DataTypeFromString("int32_ref", &dt));
  EXPECT_EQ(DT_INT32_REF, dt);
  EXPECT_FALSE(DataTypeFromString("int32_ref_ref", &dt));
  EXPECT_FALSE(DataTypeFromString("foo", &dt));
  EXPECT_FALSE(DataTypeFromString("foo_ref", &dt));
  ASSERT_TRUE(DataTypeFromString("int64", &dt));
  EXPECT_EQ(DT_INT64, dt);
  ASSERT_TRUE(DataTypeFromString("int64_ref", &dt));
  EXPECT_EQ(DT_INT64_REF, dt);
  ASSERT_TRUE(DataTypeFromString("quint8_ref", &dt));
  EXPECT_EQ(DT_QUINT8_REF, dt);
  ASSERT_TRUE(DataTypeFromString("bfloat16", &dt));
  EXPECT_EQ(DT_BFLOAT16, dt);
}

template <typename T>
static bool GetQuantized() {
  return is_quantized<T>::value;
}

TEST(TypesTest, QuantizedTypes) {
  // NOTE: GUnit cannot parse is::quantized<TYPE>::value() within the
  // EXPECT_TRUE() clause, so we delegate through a template function.
  EXPECT_TRUE(GetQuantized<qint8>());
  EXPECT_TRUE(GetQuantized<quint8>());
  EXPECT_TRUE(GetQuantized<qint32>());

  EXPECT_FALSE(GetQuantized<int8>());
  EXPECT_FALSE(GetQuantized<uint8>());
  EXPECT_FALSE(GetQuantized<int16>());
  EXPECT_FALSE(GetQuantized<int32>());

  EXPECT_TRUE(DataTypeIsQuantized(DT_QINT8));
  EXPECT_TRUE(DataTypeIsQuantized(DT_QUINT8));
  EXPECT_TRUE(DataTypeIsQuantized(DT_QINT32));

  EXPECT_FALSE(DataTypeIsQuantized(DT_INT8));
  EXPECT_FALSE(DataTypeIsQuantized(DT_UINT8));
  EXPECT_FALSE(DataTypeIsQuantized(DT_INT16));
  EXPECT_FALSE(DataTypeIsQuantized(DT_INT32));
  EXPECT_FALSE(DataTypeIsQuantized(DT_BFLOAT16));
}

}  // namespace
}  // namespace tensorflow
