/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

#define EnumStringPair(val) \
  { val, #val }

std::pair<TensorFormat, const char*> test_data_formats[] = {
    EnumStringPair(FORMAT_NHWC),
    EnumStringPair(FORMAT_NCHW),
    EnumStringPair(FORMAT_NCHW_VECT_C),
};

std::pair<FilterTensorFormat, const char*> test_filter_formats[] = {
    EnumStringPair(FORMAT_HWIO),
    EnumStringPair(FORMAT_OIHW),
    EnumStringPair(FORMAT_OIHW_VECT_I),
};

// This is an alternative way of specifying the tensor dimension indexes for
// each tensor format. For now it can be used as a cross-check of the existing
// functions, but later could replace them.

// Represents the dimension indexes of an activations tensor format.
struct TensorDimMap {
  int n() const { return dim_n; }
  int h() const { return dim_h; }
  int w() const { return dim_w; }
  int c() const { return dim_c; }
  int spatial(int spatial_index) const { return spatial_dim[spatial_index]; }

  int dim_n, dim_h, dim_w, dim_c;
  int spatial_dim[3];
};

// Represents the dimension indexes of a filter tensor format.
struct FilterDimMap {
  int h() const { return dim_h; }
  int w() const { return dim_w; }
  int i() const { return dim_i; }
  int o() const { return dim_o; }
  int spatial(int spatial_index) const { return spatial_dim[spatial_index]; }

  int dim_h, dim_w, dim_i, dim_o;
  int spatial_dim[3];
};

// clang-format off

// Predefined constants specifying the actual dimension indexes for each
// supported tensor and filter format.
struct DimMaps {
#define StaCoExTensorDm static constexpr TensorDimMap
  //                                'N', 'H', 'W', 'C'    0,  1,  2
  StaCoExTensorDm kTdmInvalid =   { -1,  -1,  -1,  -1, { -1, -1, -1 } };
  // These arrays are indexed by the number of spatial dimensions in the format.
  StaCoExTensorDm kTdmNHWC[4] = { kTdmInvalid,
                                  {  0,  -1,   1,   2, {  1, -1, -1 } },  // 1D
                                  {  0,   1,   2,   3, {  1,  2, -1 } },  // 2D
                                  {  0,   2,   3,   4, {  1,  2,  3 } }   // 3D
                                };
  StaCoExTensorDm kTdmNCHW[4] = { kTdmInvalid,
                                  {  0,  -1,   2,   1, {  2, -1, -1 } },
                                  {  0,   2,   3,   1, {  2,  3, -1 } },
                                  {  0,   3,   4,   1, {  2,  3,  4 } }
                                };
#undef StaCoExTensorDm
#define StaCoExFilterDm static constexpr FilterDimMap
  //                                'H', 'W', 'I', 'O'    0   1   2
  StaCoExFilterDm kFdmInvalid =   { -1,  -1,  -1,  -1, { -1, -1, -1 } };
  StaCoExFilterDm kFdmHWIO[4] = { kFdmInvalid,
                                  { -1,   0,   1,   2, {  0, -1, -1 } },
                                  {  0,   1,   2,   3, {  0,  1, -1 } },
                                  {  1,   2,   3,   4, {  0,  1,  2 } }
                                };
  StaCoExFilterDm kFdmOIHW[4] = { kFdmInvalid,
                                  { -1,   2,   1,   0, {  2, -1, -1 } },
                                  {  2,   3,   1,   0, {  2,  3, -1 } },
                                  {  3,   4,   1,   0, {  2,  3,  4 } }
                                };
#undef StaCoExFilterDm
};

inline constexpr const TensorDimMap&
GetTensorDimMap(const int num_spatial_dims, const TensorFormat format) {
  return
      (format == FORMAT_NHWC) ? DimMaps::kTdmNHWC[num_spatial_dims] :
      (format == FORMAT_NCHW ||
       format == FORMAT_NCHW_VECT_C) ? DimMaps::kTdmNCHW[num_spatial_dims]
                                     : DimMaps::kTdmInvalid;
}

inline constexpr const FilterDimMap&
GetFilterDimMap(const int num_spatial_dims,
                const FilterTensorFormat format) {
  return
      (format == FORMAT_HWIO) ? DimMaps::kFdmHWIO[num_spatial_dims] :
      (format == FORMAT_OIHW ||
       format == FORMAT_OIHW_VECT_I) ? DimMaps::kFdmOIHW[num_spatial_dims]
                                     : DimMaps::kFdmInvalid;
}
// clang-format on

constexpr TensorDimMap DimMaps::kTdmInvalid;
constexpr TensorDimMap DimMaps::kTdmNHWC[4];
constexpr TensorDimMap DimMaps::kTdmNCHW[4];
constexpr FilterDimMap DimMaps::kFdmInvalid;
constexpr FilterDimMap DimMaps::kFdmHWIO[4];
constexpr FilterDimMap DimMaps::kFdmOIHW[4];

TEST(TensorFormatTest, FormatEnumsAndStrings) {
  const string prefix = "FORMAT_";
  for (auto& test_data_format : test_data_formats) {
    const char* stringified_format_enum = test_data_format.second;
    LOG(INFO) << stringified_format_enum << " = " << test_data_format.first;
    string expected_format_str = &stringified_format_enum[prefix.size()];
    TensorFormat format;
    EXPECT_TRUE(FormatFromString(expected_format_str, &format));
    string format_str = ToString(format);
    EXPECT_EQ(expected_format_str, format_str);
    EXPECT_EQ(test_data_format.first, format);
  }
  for (auto& test_filter_format : test_filter_formats) {
    const char* stringified_format_enum = test_filter_format.second;
    LOG(INFO) << stringified_format_enum << " = " << test_filter_format.first;
    string expected_format_str = &stringified_format_enum[prefix.size()];
    FilterTensorFormat format;
    EXPECT_TRUE(FilterFormatFromString(expected_format_str, &format));
    string format_str = ToString(format);
    EXPECT_EQ(expected_format_str, format_str);
    EXPECT_EQ(test_filter_format.first, format);
  }
}

template <int num_spatial_dims>
void RunDimensionIndexesTest() {
  for (auto& test_data_format : test_data_formats) {
    TensorFormat format = test_data_format.first;
    auto& tdm = GetTensorDimMap(num_spatial_dims, format);
    int num_dims = GetTensorDimsFromSpatialDims(num_spatial_dims, format);
    LOG(INFO) << ToString(format) << ", num_spatial_dims=" << num_spatial_dims
              << ", num_dims=" << num_dims;
    EXPECT_EQ(GetTensorBatchDimIndex(num_dims, format), tdm.n());
    EXPECT_EQ(GetTensorDimIndex<num_spatial_dims>(format, 'N'), tdm.n());
    EXPECT_EQ(GetTensorFeatureDimIndex(num_dims, format), tdm.c());
    EXPECT_EQ(GetTensorDimIndex<num_spatial_dims>(format, 'C'), tdm.c());
    for (int i = 0; i < num_spatial_dims; ++i) {
      EXPECT_EQ(GetTensorSpatialDimIndex(num_dims, format, i), tdm.spatial(i));
      EXPECT_EQ(GetTensorDimIndex<num_spatial_dims>(format, '0' + i),
                tdm.spatial(i));
    }
  }
  for (auto& test_filter_format : test_filter_formats) {
    FilterTensorFormat format = test_filter_format.first;
    auto& fdm = GetFilterDimMap(num_spatial_dims, format);
    int num_dims = GetFilterTensorDimsFromSpatialDims(num_spatial_dims, format);
    LOG(INFO) << ToString(format) << ", num_spatial_dims=" << num_spatial_dims
              << ", num_dims=" << num_dims;
    EXPECT_EQ(GetFilterTensorOutputChannelsDimIndex(num_dims, format), fdm.o());
    EXPECT_EQ(GetFilterDimIndex<num_spatial_dims>(format, 'O'), fdm.o());
    EXPECT_EQ(GetFilterTensorInputChannelsDimIndex(num_dims, format), fdm.i());
    EXPECT_EQ(GetFilterDimIndex<num_spatial_dims>(format, 'I'), fdm.i());
    for (int i = 0; i < num_spatial_dims; ++i) {
      EXPECT_EQ(GetFilterTensorSpatialDimIndex(num_dims, format, i),
                fdm.spatial(i));
      EXPECT_EQ(GetFilterDimIndex<num_spatial_dims>(format, '0' + i),
                fdm.spatial(i));
    }
  }
}

TEST(TensorFormatTest, DimensionIndexes) {
  RunDimensionIndexesTest<1>();
  RunDimensionIndexesTest<2>();
  RunDimensionIndexesTest<3>();
}

}  // namespace tensorflow
