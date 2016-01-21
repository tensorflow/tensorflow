/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

namespace f = test::function;
typedef FunctionDefHelper FDH;

class ArrayGradTest : public ::testing::Test {};

Session* NewSession() {
  SessionOptions opts;
  (*opts.config.mutable_device_count())["CPU"] = 1;
  return NewSession(opts);
}

std::vector<Tensor> PackGrad(const Tensor& x0, const Tensor& x1,
                             const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("x1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
               {{"f", FDH::FunctionRef("Pack", {{"N", 2}, {"T", T}})},
                {"Tin", DataTypeSlice{T, T, T}},
                {"Tout", DataTypeSlice{T, T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x0:0", x0}, {"x1:0", x1}, {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, PackGrad) {
  Tensor x0(DT_FLOAT, {2, 3});
  x0.flat<float>().setZero();
  Tensor x1(DT_FLOAT, {2, 3});
  x1.flat<float>().setZero();
  Tensor dy(DT_FLOAT, {2, 2, 3});
  test::FillIota<float>(&dy, 0);
  auto dx = PackGrad(x0, x1, dy);
  test::ExpectClose(dx[0],
                    test::AsTensor<float>({0., 1., 2., 3., 4., 5.}, {2, 3}));
  test::ExpectClose(dx[1],
                    test::AsTensor<float>({6., 7., 8., 9., 10., 11.}, {2, 3}));
}

std::vector<Tensor> UnpackGrad(const Tensor& x, const Tensor& dy0,
                               const Tensor& dy1) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "dy0", "dy1"},
               {{"f", FDH::FunctionRef("Unpack", {{"num", 2}, {"T", T}})},
                {"Tin", DataTypeSlice{T, T, T}},
                {"Tout", DataTypeSlice{T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"dy0:0", dy0}, {"dy1:0", dy1}}, {"dx:0"},
                        {}, &out));
  CHECK_EQ(out.size(), 1);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, UnpackGrad) {
  Tensor x(DT_FLOAT, {2, 2, 3});
  x.flat<float>().setZero();
  Tensor dy0(DT_FLOAT, {2, 3});
  Tensor dy1(DT_FLOAT, {2, 3});
  test::FillIota<float>(&dy0, 0);
  test::FillIota<float>(&dy1, 100);
  auto dx = UnpackGrad(x, dy0, dy1);
  test::ExpectClose(dx[0], test::AsTensor<float>({0., 1., 2., 3., 4., 5., 100.,
                                                  101., 102., 103., 104., 105.},
                                                 {2, 2, 3}));
}

std::vector<Tensor> ConcatGrad(int dim, const Tensor& x0, const Tensor& x1,
                               const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("dim", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("x0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("x1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"dim", "x0", "x1", "dy"},
               {{"f", FDH::FunctionRef("Concat", {{"N", 2}, {"T", T}})},
                {"Tin", DataTypeSlice{DT_INT32, T, T, T}},
                {"Tout", DataTypeSlice{DT_INT32, T, T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run(
      {{"dim", test::AsScalar(dim)}, {"x0:0", x0}, {"x1:0", x1}, {"dy:0", dy}},
      {"dx:0", "dx:1", "dx:2"}, {}, &out));
  CHECK_EQ(out.size(), 3);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, ConcatGrad) {
  Tensor x0(DT_FLOAT, {2, 3, 5});
  x0.flat<float>().setZero();
  Tensor x1(DT_FLOAT, {2, 1, 5});
  x1.flat<float>().setZero();
  Tensor dy(DT_FLOAT, {2, 4, 5});
  test::FillIota<float>(&dy, 0);
  auto dx = ConcatGrad(1, x0, x1, dy);
  test::ExpectTensorEqual<int32>(dx[0], test::AsScalar(0));
  test::ExpectClose(
      dx[1],
      test::AsTensor<float>({0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
                             10., 11., 12., 13., 14., 20., 21., 22., 23., 24.,
                             25., 26., 27., 28., 29., 30., 31., 32., 33., 34.},
                            {2, 3, 5}));
  test::ExpectClose(dx[2], test::AsTensor<float>({15., 16., 17., 18., 19., 35.,
                                                  36., 37., 38., 39.},
                                                 {2, 1, 5}));
}

std::vector<Tensor> SplitGrad(int dim, const Tensor& x, const Tensor& dy0,
                              const Tensor& dy1) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("dim", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"dim", "x", "dy0", "dy1"},
               {{"f", FDH::FunctionRef(
                          "Split",
                          {{"split_dim", dim}, {"num_split", 2}, {"T", T}})},
                {"Tin", DataTypeSlice{DT_INT32, T, T, T}},
                {"Tout", DataTypeSlice{DT_INT32, T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"dim", test::AsScalar(dim)},
                         {"x:0", x},
                         {"dy0:0", dy0},
                         {"dy1:0", dy1}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, SplitGrad) {
  Tensor x(DT_FLOAT, {2, 4, 5});
  x.flat<float>().setZero();
  Tensor dy0(DT_FLOAT, {2, 2, 5});
  Tensor dy1(DT_FLOAT, {2, 2, 5});
  test::FillIota<float>(&dy0, 0);
  test::FillIota<float>(&dy1, 100);
  auto dx = SplitGrad(1, x, dy0, dy1);
  test::ExpectTensorEqual<int32>(dx[0], test::AsScalar(0));
  test::ExpectClose(
      dx[1], test::AsTensor<float>(
                 {0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
                  100., 101., 102., 103., 104., 105., 106., 107., 108., 109.,
                  10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,
                  110., 111., 112., 113., 114., 115., 116., 117., 118., 119.},
                 {2, 4, 5}));
}

}  // namespace tensorflow
