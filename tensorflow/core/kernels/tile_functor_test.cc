/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/tile_functor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Forward declarations  that will be defined in tile_functor_cpu.cc
namespace internal {
template <Device, int64>
void TileSimple(const Device& device, Tensor* out, const Tensor& in,
                const gtl::ArraySlice<int32>& multiples_array);

}  // end namespace internal

namespace {

class TileOpUtilTest : public OpsTestBase {
  protected:
    template <typename T>
    void TestTileSimple(std::initializer_list<int64> in,
                        TensorShape in_shape,
                        gtl::ArraySlice<int32> multiples_array,
                        std::initializer_list<int64> out) {
      DataType value_type = tensorflow::DataTypeToEnum<T>::value;
      Tensor input(value_type, in_shape);
      test::FillValues<T>(&input, in);
      TensorShape out_shape;
      for (int i = 0; i < in_shape.dims(); ++i) {
        out_shape.AddDim(in_shape.dim_size(i) * multiples_array[i]);
      }
      TensorShape output_shape(out_shape);
      Tensor output(value_type, output_shape);
      Tensor expected(value_type, output_shape);
      test::FillValues<T>(&expected, out);
      internal::TileSimple<Device, T>(*(this->device_.get()), &output, input, multiples_array);
      test::ExpectTensorEqual<T>(output, expected);
    }
};


#define RUN_TEST(VALTYPE)                                                     \
  TEST_F(TileOpUtilTest, Tile_1D_##VALTYPE) {                                 \
    TestTileSimple<VALTYPE>({1,2,3}, {3}, {2}, {1,2,3,1,2,3});                         \
  }  \
  TEST_F(TileOpUtilTest, Tile_2D_##VALTYPE) {                                 \
    TestTileSimple<VALTYPE>({1,2,3,4,5,6}, {2,3}, {2,2},   \
                            {1,2,3,1,2,3,4,5,6,4,5,6,1,2,3,1,2,3,4,5,6,4,5,6});                         \
  } \
  TEST_F(TileOpUtilTest, Tile_3D_##VALTYPE) {                                 \
    TestTileSimple<VALTYPE>({1,2,3,4,5,6,7,8,9,10,11,12}, {2,2,3}, {2,2,3},   \
                            {1,2,3,1,2,3,1,2,3, \
                             4,5,6,4,5,6,4,5,6,   \
                             1,2,3,1,2,3,1,2,3, \
                             4,5,6,4,5,6,4,5,6,                         \
                             7,8,9,7,8,9,7,8,9, \
                             10,11,12,10,11,12,10,11,12,   \
                             7,8,9,7,8,9,7,8,9, \
                             10,11,12,10,11,12,10,11,12, \
                             1,2,3,1,2,3,1,2,3, \
                             4,5,6,4,5,6,4,5,6,                         \
                             1,2,3,1,2,3,1,2,3, \
                             4,5,6,4,5,6,4,5,6,                         \
                             7,8,9,7,8,9,7,8,9, \
                             10,11,12,10,11,12,10,11,12,   \
                             7,8,9,7,8,9,7,8,9, \
                             10,11,12,10,11,12,10,11,12}); \
  }

RUN_TEST(int64);
} // end namespace

} // end namespace tensorflow
