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

#include <vector>
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
                             const Tensor& dy, int axis) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("x1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("axis", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
               {{"f", FDH::FunctionRef("Pack",
                                       {{"N", 2}, {"T", T}, {"axis", axis}})},
                {"Tin", DataTypeSlice{T, T, T}},
                {"Tout", DataTypeSlice{T, T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x0:0", x0},
                         {"x1:0", x1},
                         {"axis:0", test::AsScalar(axis)},
                         {"dy:0", dy}},
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
  auto dx = PackGrad(x0, x1, dy, 0);
  test::ExpectClose(dx[0],
                    test::AsTensor<float>({0., 1., 2., 3., 4., 5.}, {2, 3}));
  test::ExpectClose(dx[1],
                    test::AsTensor<float>({6., 7., 8., 9., 10., 11.}, {2, 3}));
}

std::vector<Tensor> UnpackGrad(const Tensor& x, const Tensor& dy0,
                               const Tensor& dy1, int axis) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("axis", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "dy0", "dy1"},
               {{"f", FDH::FunctionRef("Unpack",
                                       {{"num", 2}, {"T", T}, {"axis", axis}})},
                {"Tin", DataTypeSlice{T, T, T}},
                {"Tout", DataTypeSlice{T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x},
                         {"axis:0", test::AsScalar(axis)},
                         {"dy0:0", dy0},
                         {"dy1:0", dy1}},
                        {"dx:0"}, {}, &out));
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
  auto dx = UnpackGrad(x, dy0, dy1, 0);
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

std::vector<Tensor> ConcatGradV2(int dim, const Tensor& x0, const Tensor& x1,
                                 const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x0", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("x1", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dim", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x0", "x1", "dim", "dy"},
               {{"f", FDH::FunctionRef("ConcatV2", {{"N", 2}, {"T", T}})},
                {"Tin", DataTypeSlice{T, T, DT_INT32, T}},
                {"Tout", DataTypeSlice{T, T, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run(
      {{"x0:0", x0}, {"x1:0", x1}, {"dim", test::AsScalar(dim)}, {"dy:0", dy}},
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

  dx = ConcatGradV2(1, x0, x1, dy);
  test::ExpectTensorEqual<int32>(dx[dx.size() - 1], test::AsScalar(0));
  test::ExpectClose(
      dx[0],
      test::AsTensor<float>({0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
                             10., 11., 12., 13., 14., 20., 21., 22., 23., 24.,
                             25., 26., 27., 28., 29., 30., 31., 32., 33., 34.},
                            {2, 3, 5}));
  test::ExpectClose(dx[1], test::AsTensor<float>({15., 16., 17., 18., 19., 35.,
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

std::vector<Tensor> ReshapeGrad(const Tensor& x, const Tensor& s,
                                const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("s", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "s", "dy"},
               {{"f", FDH::FunctionRef("Reshape", {{"T", T}})},
                {"Tin", DataTypeSlice{T, DT_INT32, T}},
                {"Tout", DataTypeSlice{T, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"s:0", s}, {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, ReshapeGrad) {
  Tensor x(DT_FLOAT, {2, 4, 5});
  x.flat<float>().setZero();
  auto s = test::AsTensor<int32>({8, 5});
  Tensor dy(DT_FLOAT, {8, 5});
  test::FillIota<float>(&dy, 73);
  auto dx = ReshapeGrad(x, s, dy);
  test::ExpectClose(
      dx[0], test::AsTensor<float>(
                 {73.,  74.,  75.,  76.,  77.,  78.,  79.,  80.,  81.,  82.,
                  83.,  84.,  85.,  86.,  87.,  88.,  89.,  90.,  91.,  92.,
                  93.,  94.,  95.,  96.,  97.,  98.,  99.,  100., 101., 102.,
                  103., 104., 105., 106., 107., 108., 109., 110., 111., 112.},
                 {2, 4, 5}));
  test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0}));
}

std::vector<Tensor> ExpandDimsGrad(const Tensor& x, const Tensor& s,
                                   const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("s", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "s", "dy"},
               {{"f", FDH::FunctionRef("ExpandDims", {{"T", T}})},
                {"Tin", DataTypeSlice{T, DT_INT32, T}},
                {"Tout", DataTypeSlice{T, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"s:0", s}, {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, ExpandDimsGrad) {
  Tensor x(DT_FLOAT, {2, 4, 5});
  x.flat<float>().setZero();
  auto s = test::AsTensor<int32>({1});
  Tensor dy(DT_FLOAT, {2, 1, 4, 5});
  test::FillIota<float>(&dy, 73);
  auto dx = ExpandDimsGrad(x, s, dy);
  test::ExpectClose(
      dx[0], test::AsTensor<float>(
                 {73.,  74.,  75.,  76.,  77.,  78.,  79.,  80.,  81.,  82.,
                  83.,  84.,  85.,  86.,  87.,  88.,  89.,  90.,  91.,  92.,
                  93.,  94.,  95.,  96.,  97.,  98.,  99.,  100., 101., 102.,
                  103., 104., 105., 106., 107., 108., 109., 110., 111., 112.},
                 {2, 4, 5}));
  test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0}));
}

std::vector<Tensor> SqueezeGrad(const Tensor& x, const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "dy"},
               {{"f", FDH::FunctionRef("Squeeze", {{"T", T}})},
                {"Tin", DataTypeSlice{T, T}},
                {"Tout", DataTypeSlice{T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"dy:0", dy}}, {"dx:0"}, {}, &out));
  CHECK_EQ(out.size(), 1);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, SqueezeGrad) {
  Tensor x(DT_FLOAT, {2, 1, 3});
  x.flat<float>().setZero();
  Tensor dy(DT_FLOAT, {2, 3});
  test::FillIota<float>(&dy, 1);
  auto dx = SqueezeGrad(x, dy);
  test::ExpectClose(dx[0],
                    test::AsTensor<float>({1., 2., 3., 4., 5., 6.}, {2, 1, 3}));
}

std::vector<Tensor> TransposeGrad(const Tensor& x, const Tensor& p,
                                  const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("p", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "p", "dy"},
               {{"f", FDH::FunctionRef("Transpose", {{"T", T}})},
                {"Tin", DataTypeSlice{T, DT_INT32, T}},
                {"Tout", DataTypeSlice{T, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"p:0", p}, {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, TransposeGrad) {
  Tensor x(DT_FLOAT, {2, 4, 5});
  x.flat<float>().setZero();
  auto p = test::AsTensor<int32>({2, 0, 1});
  Tensor dy(DT_FLOAT, {5, 2, 4});
  test::FillIota<float>(&dy, 0);
  auto dx = TransposeGrad(x, p, dy);
  test::ExpectClose(dx[0], test::AsTensor<float>(
                               {0., 8.,  16., 24., 32., 1., 9.,  17., 25., 33.,
                                2., 10., 18., 26., 34., 3., 11., 19., 27., 35.,
                                4., 12., 20., 28., 36., 5., 13., 21., 29., 37.,
                                6., 14., 22., 30., 38., 7., 15., 23., 31., 39.},
                               {2, 4, 5}));
  test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0, 0}));
}

std::vector<Tensor> ReverseGrad(const Tensor& x, const Tensor& dims,
                                const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dims", "Placeholder", {}, {{"dtype", DT_BOOL}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("dx", "SymbolicGradient", {"x", "dims", "dy"},
               {{"f", FDH::FunctionRef("Reverse", {{"T", T}})},
                {"Tin", DataTypeSlice{T, DT_BOOL, T}},
                {"Tout", DataTypeSlice{T, DT_BOOL}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"dims:0", dims}, {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, ReverseGrad) {
  Tensor x(DT_FLOAT, {2, 3});
  x.flat<float>().setZero();
  auto dims = test::AsTensor<bool>({false, true});
  Tensor dy(DT_FLOAT, {2, 3});
  test::FillIota<float>(&dy, 1);
  auto dx = ReverseGrad(x, dims, dy);
  test::ExpectClose(dx[0],
                    test::AsTensor<float>({3., 2., 1., 6., 5., 4.}, {2, 3}));
  test::ExpectTensorEqual<bool>(dx[1], test::AsTensor<bool>({false, false}));
}

std::vector<Tensor> ReverseV2Grad(const Tensor& x, const Tensor& axis,
                                  const Tensor& dy) {
  auto T = DT_FLOAT;
  auto Tidx = DT_INT32;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("axis", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef(
           "dx", "SymbolicGradient", {"x", "axis", "dy"},
           {{"f", FDH::FunctionRef("ReverseV2", {{"T", T}, {"Tidx", Tidx}})},
            {"Tin", DataTypeSlice{T, DT_INT32, T}},
            {"Tout", DataTypeSlice{T, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"axis:0", axis}, {"dy:0", dy}},
                        {"dx:0", "dx:1"}, {}, &out));
  CHECK_EQ(out.size(), 2);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, ReverseV2Grad) {
  Tensor x(DT_FLOAT, {2, 3});
  x.flat<float>().setZero();
  auto axis = test::AsTensor<int32>({1});
  Tensor dy(DT_FLOAT, {2, 3});
  test::FillIota<float>(&dy, 1);
  auto dx = ReverseV2Grad(x, axis, dy);
  test::ExpectTensorEqual<float>(
      dx[0], test::AsTensor<float>({3., 2., 1., 6., 5., 4.}, {2, 3}));
  test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0}));
}

std::vector<Tensor> SliceGrad(const Tensor& x, const Tensor& b, const Tensor& s,
                              const Tensor& dy) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("b", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("s", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef(
           "dx", "SymbolicGradient", {"x", "b", "s", "dy"},
           {{"f", FDH::FunctionRef("Slice", {{"T", T}, {"Index", DT_INT32}})},
            {"Tin", DataTypeSlice{T, DT_INT32, DT_INT32, T}},
            {"Tout", DataTypeSlice{T, DT_INT32, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x}, {"b:0", b}, {"s:0", s}, {"dy:0", dy}},
                        {"dx:0", "dx:1", "dx:2"}, {}, &out));
  CHECK_EQ(out.size(), 3);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, SliceGrad) {
  Tensor x(DT_FLOAT, {2, 3, 4});
  x.flat<float>().setZero();
  auto begin = test::AsTensor<int32>({1, 1, 1});
  auto size = test::AsTensor<int32>({1, 2, 2});
  Tensor dy(DT_FLOAT, {1, 2, 2});
  test::FillIota<float>(&dy, 1);
  auto dx = SliceGrad(x, begin, size, dy);
  test::ExpectClose(dx[0],
                    test::AsTensor<float>(
                        {
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 1., 2., 0., 0., 3., 4., 0.,
                        },
                        {2, 3, 4}));
  test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0, 0}));
  test::ExpectTensorEqual<int32>(dx[2], test::AsTensor<int32>({0, 0, 0}));
}

std::vector<Tensor> StridedSliceGrad(const Tensor& x, const Tensor& begin,
                                     const Tensor& end, const Tensor& strides,
                                     const Tensor& dy, int32 begin_mask,
                                     int32 end_mask, int32 ellipsis_mask,
                                     int32 new_axis_mask,
                                     int32 shrink_axis_mask) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("begin", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("end", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("strides", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef(
           "dx", "SymbolicGradient", {"x", "begin", "end", "strides", "dy"},
           {{"f", FDH::FunctionRef("StridedSlice",
                                   {
                                       {"T", T},
                                       {"Index", DT_INT32},
                                       {"begin_mask", begin_mask},
                                       {"end_mask", end_mask},
                                       {"new_axis_mask", new_axis_mask},
                                       {"shrink_axis_mask", shrink_axis_mask},
                                       {"ellipsis_mask", ellipsis_mask},
                                   })},
            {"Tin", DataTypeSlice{T, DT_INT32, DT_INT32, DT_INT32, T}},
            {"Tout", DataTypeSlice{T, DT_INT32, DT_INT32, DT_INT32}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"x:0", x},
                         {"begin:0", begin},
                         {"end:0", end},
                         {"strides:0", strides},
                         {"dy:0", dy}},
                        {"dx:0", "dx:1", "dx:2", "dx:3"}, {}, &out));
  CHECK_EQ(out.size(), 4);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

std::vector<Tensor> StridedSliceGradGrad(
    const Tensor& shape, const Tensor& begin, const Tensor& end,
    const Tensor& strides, const Tensor& dy, const Tensor& grad,
    int32 begin_mask, int32 end_mask, int32 ellipsis_mask, int32 new_axis_mask,
    int32 shrink_axis_mask) {
  auto T = DT_FLOAT;
  auto gdef = test::function::GDef(
      {f::NDef("shape", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("begin", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("end", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("strides", "Placeholder", {}, {{"dtype", DT_INT32}}),
       f::NDef("dy", "Placeholder", {}, {{"dtype", T}}),
       f::NDef("grad", "Placeholder", {}, {{"dtype", T}}),
       f::NDef(
           "dx", "SymbolicGradient",
           {"shape", "begin", "end", "strides", "dy", "grad"},
           {{"f", FDH::FunctionRef("StridedSliceGrad",
                                   {
                                       {"T", T},
                                       {"Index", DT_INT32},
                                       {"begin_mask", begin_mask},
                                       {"end_mask", end_mask},
                                       {"new_axis_mask", new_axis_mask},
                                       {"shrink_axis_mask", shrink_axis_mask},
                                       {"ellipsis_mask", ellipsis_mask},
                                   })},
            {"Tin",
             DataTypeSlice{DT_INT32, DT_INT32, DT_INT32, DT_INT32, T, T}},
            {"Tout",
             DataTypeSlice{DT_INT32, DT_INT32, DT_INT32, DT_INT32, T}}})});
  VLOG(1) << DebugStringWhole(gdef);
  auto sess = NewSession();
  TF_CHECK_OK(sess->Create(gdef));
  std::vector<Tensor> out;
  TF_CHECK_OK(sess->Run({{"shape:0", shape},
                         {"begin:0", begin},
                         {"end:0", end},
                         {"strides:0", strides},
                         {"dy:0", dy},
                         {"grad:0", grad}},
                        {"dx:0", "dx:1", "dx:2", "dx:3", "dx:4"}, {}, &out));
  CHECK_EQ(out.size(), 5);
  TF_CHECK_OK(sess->Close());
  delete sess;
  return out;
}

TEST_F(ArrayGradTest, StridedSliceGrad) {
  Tensor x(DT_FLOAT, {2, 3, 4});
  x.flat<float>().setZero();
  Tensor x_shape = test::AsTensor<int32>({2, 3, 4}, {3});

  {
    auto start = test::AsTensor<int32>({1, 1, 1});
    auto stop = test::AsTensor<int32>({2, 3, 3});
    auto strides = test::AsTensor<int32>({1, 1, 1});
    Tensor dy(DT_FLOAT, {1, 2, 2});
    test::FillIota<float>(&dy, 1);
    int begin_mask = 0, end_mask = 0, new_axis_mask = 0, shrink_axis_mask = 0,
        ellipsis_mask = 0;
    auto dx =
        StridedSliceGrad(x, start, stop, strides, dy, begin_mask, end_mask,
                         ellipsis_mask, new_axis_mask, shrink_axis_mask);
    test::ExpectClose(dx[0],
                      test::AsTensor<float>(
                          {
                              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 1., 2., 0., 0., 3., 4., 0.,
                          },
                          {2, 3, 4}));
    test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0, 0}));
    test::ExpectTensorEqual<int32>(dx[2], test::AsTensor<int32>({0, 0, 0}));
    auto ddx = StridedSliceGradGrad(x_shape, start, stop, strides, dy, dx[0],
                                    begin_mask, end_mask, ellipsis_mask,
                                    new_axis_mask, shrink_axis_mask);
    test::ExpectClose(ddx[4], dy);
  }

  // test equivalent of python tf.gradients(foo[1:2, 1:3, 1:3])
  {
    auto start = test::AsTensor<int32>({1, 1, 1});
    auto stop = test::AsTensor<int32>({2, 3, 3});
    auto strides = test::AsTensor<int32>({1, 1, 1});
    Tensor dy(DT_FLOAT, {1, 2, 2});
    test::FillIota<float>(&dy, 1);
    int begin_mask = 0, end_mask = 0, new_axis_mask = 0, shrink_axis_mask = 0,
        ellipsis_mask = 0;
    auto dx =
        StridedSliceGrad(x, start, stop, strides, dy, begin_mask, end_mask,
                         ellipsis_mask, new_axis_mask, shrink_axis_mask);
    test::ExpectClose(dx[0],
                      test::AsTensor<float>(
                          {
                              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 1., 2., 0., 0., 3., 4., 0.,
                          },
                          {2, 3, 4}));
    test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0, 0}));
    test::ExpectTensorEqual<int32>(dx[2], test::AsTensor<int32>({0, 0, 0}));
    auto ddx = StridedSliceGradGrad(x_shape, start, stop, strides, dy, dx[0],
                                    begin_mask, end_mask, ellipsis_mask,
                                    new_axis_mask, shrink_axis_mask);
    test::ExpectClose(ddx[4], dy);
  }

  // test equivalent of python tf.gradients(foo[1, 1:, :-2, None])
  {
    int dontcare = 66;
    auto start = test::AsTensor<int32>({1, 1, dontcare, dontcare});
    auto stop = test::AsTensor<int32>({2, dontcare, -2, dontcare});
    auto strides = test::AsTensor<int32>({1, 1, 1, dontcare});
    Tensor dy(DT_FLOAT, {2, 2, 1});
    test::FillIota<float>(&dy, 1);
    int begin_mask = 4, end_mask = 2, new_axis_mask = 8, shrink_axis_mask = 1,
        ellipsis_mask = 0;
    auto dx =
        StridedSliceGrad(x, start, stop, strides, dy, begin_mask, end_mask,
                         ellipsis_mask, new_axis_mask, shrink_axis_mask);
    test::ExpectClose(dx[0],
                      test::AsTensor<float>(
                          {
                              0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 1., 2., 0., 0., 3., 4., 0., 0.,
                          },
                          {2, 3, 4}));
    test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0, 0, 0}));
    test::ExpectTensorEqual<int32>(dx[2], test::AsTensor<int32>({0, 0, 0, 0}));
    auto ddx = StridedSliceGradGrad(x_shape, start, stop, strides, dy, dx[0],
                                    begin_mask, end_mask, ellipsis_mask,
                                    new_axis_mask, shrink_axis_mask);
    test::ExpectClose(ddx[4], dy);
  }

  // test equivalent of tf.gradients(foo[1, ...]) i.e. foo[1, 0:3, 0:4]
  {
    int dontcare = 66;
    auto start = test::AsTensor<int32>({1, dontcare});
    auto stop = test::AsTensor<int32>({2, dontcare});
    auto strides = test::AsTensor<int32>({1, 1});
    Tensor dy(DT_FLOAT, {3, 4});
    test::FillIota<float>(&dy, 1);
    int begin_mask = 0, end_mask = 0, new_axis_mask = 0, shrink_axis_mask = 1,
        ellipsis_mask = 2;
    auto dx =
        StridedSliceGrad(x, start, stop, strides, dy, begin_mask, end_mask,
                         ellipsis_mask, new_axis_mask, shrink_axis_mask);
    test::ExpectClose(dx[0],
                      test::AsTensor<float>(
                          {
                              0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0.,  0.,
                              1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
                          },
                          {2, 3, 4}));
    test::ExpectTensorEqual<int32>(dx[1], test::AsTensor<int32>({0, 0}));
    test::ExpectTensorEqual<int32>(dx[2], test::AsTensor<int32>({0, 0}));
    auto ddx = StridedSliceGradGrad(x_shape, start, stop, strides, dy, dx[0],
                                    begin_mask, end_mask, ellipsis_mask,
                                    new_axis_mask, shrink_axis_mask);
    test::ExpectClose(ddx[4], dy);
  }
}

}  // namespace tensorflow
