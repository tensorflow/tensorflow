/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "tensorflow/core/common_runtime/mkl_graph_testing_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

// clang-format off

// Test set 1: Conv2D + AddBias

// C=Conv2D(A,B); E=BiasAdd(C,D); Z=Zeta(E,Y)
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);      \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: '" #INPUT "'}"                           \
              "node { name: 'C' op: 'Conv2D'"                                \
              " attr { key: 'T'                value { type:" #T " } }"      \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B']}"                                          \
              "node { name: 'D' op: '" #INPUT "'}"                           \
              "node { name: 'E' op: 'BiasAdd'"                               \
              " attr { key: 'T'                value { type:" #T " } }"      \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['C', 'D'] }"                                         \
              "node { name: 'Y' op: '" #INPUT "'}"                           \
              "node { name: 'Z' op: 'Zeta'"                                  \
              " attr {key: 'T'                 value { type:" #T " } }"      \
              " input: ['E', 'Y']}");                                        \
    if (!NativeFormatEnabled()) {                                            \
      EXPECT_EQ(                                                             \
          DoMklLayoutOptimizationPass(),                                     \
          "A(" #INPUT ");B(" #INPUT ");D(" #INPUT ");DMT/_0(Const);"         \
          "DMT/_1(Const);DMT/_2(Const);E(_MklConv2DWithBias);Y(" #INPUT ");" \
          "Z(Zeta)|A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"\
          "A:control->DMT/_2:control;B->E:1;D->E:2;DMT/_0->E:3;DMT/_1->E:4;" \
          "DMT/_2->E:5;E->Z;Y->Z:1");                                        \
    } else {                                                                 \
      EXPECT_EQ(                                                             \
          DoMklLayoutOptimizationPass(),                                     \
          "A(" #INPUT ");B(" #INPUT ");D(" #INPUT ");"                       \
          "E(_MklNativeConv2DWithBias);Y(" #INPUT ");Z(Zeta)"                \
          "|A->E;B->E:1;D->E:2;E->Z;Y->Z:1");                                \
    }                                                                        \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Positive);
#undef REGISTER_TEST


// Graph contains only Conv2D, no AddBias.
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type:" #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}");                                      \
    if (!NativeFormatEnabled()) {                                          \
      EXPECT_EQ(                                                           \
          DoMklLayoutOptimizationPass(),                                   \
          "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);DMT/_0(Const);"       \
          "DMT/_1(Const)|A->C;A:control->DMT/_0:control;A:control->"       \
          "DMT/_1:control;B->C:1;DMT/_0->C:2;DMT/_1->C:3");                \
    } else {                                                               \
      EXPECT_EQ(                                                           \
          DoMklLayoutOptimizationPass(),                                   \
          "A(" #INPUT ");B(" #INPUT ");C(_MklNativeConv2D)"                \
          "|A->C;B->C:1");                                                 \
    }                                                                      \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_NoAddBias);
#undef REGISTER_TEST

// Conv2D output does not go to BiasAdd.
#define REGISTER_TEST(NAME, T, INPUT)                                          \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                      \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                             \
              "node { name: 'B' op: '" #INPUT "'}"                             \
              "node { name: 'C' op: 'Conv2D'"                                  \
              " attr { key: 'T'                value { type:" #T "} }"         \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"          \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"           \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "     \
              "i:1, i:1} } }"                                                  \
              " attr { key: 'padding'          value { s: 'SAME' } }"          \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "     \
              "i:1, i:1} } }"                                                  \
              " input: ['A', 'B']}"                                            \
              "node { name: 'D' op: '" #INPUT "'}"                             \
              "node { name: 'E' op: '" #INPUT "'}"                             \
              "node { name: 'F' op: 'BiasAdd'"                                 \
              " attr { key: 'T'                value { type:" #T "} }"         \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"          \
              " input: ['D', 'E'] }");                                         \
    if (!NativeFormatEnabled()) {                                              \
      EXPECT_EQ(DoMklLayoutOptimizationPass(),                                 \
                "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(" #INPUT ");"     \
                "DMT/_0(Const);DMT/_1(Const);E(" #INPUT ");F(BiasAdd)|A->C;"   \
                "A:control->DMT/_0:control;A:control->DMT/_1:control;B->C:1;"  \
                "D->F;DMT/_0->C:2;DMT/_1->C:3;E->F:1");                        \
    } else {                                                                   \
      EXPECT_EQ(DoMklLayoutOptimizationPass(),                                 \
      "A(" #INPUT ");B(" #INPUT ");C(_MklNativeConv2D);D(" #INPUT ");"         \
      "E(" #INPUT ");F(BiasAdd)|A->C;B->C:1;D->F;E->F:1");                     \
    }                                                                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_Dataflow1);
#undef REGISTER_TEST

// Conv2D has two outgoing edges: BiasAdd and some other dummy node (Zeta).
// Merge should not be done in such case.
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: '" #INPUT "'}"                         \
              "node { name: 'F' op: 'BiasAdd'"                             \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " input: ['D', 'E'] }"                                       \
              "node { name: 'G' op: 'Zeta'"                                \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " input: ['C', 'E'] }");                                     \
    if (!NativeFormatEnabled()) {                                            \
      EXPECT_EQ(                                                             \
          DoMklLayoutOptimizationPass(),                                     \
          "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(" #INPUT ");"         \
          "DMT/_0(Const);DMT/_1(Const);E(" #INPUT ");F(BiasAdd);G(Zeta)|"    \
          "A->C;A:control->DMT/_0:control;A:control->DMT/_1:control;B->C:1;" \
          "C->G;D->F;DMT/_0->C:2;DMT/_1->C:3;E->F:1;E->G:1");                \
    } else {                                                                 \
      EXPECT_EQ(                                                             \
          DoMklLayoutOptimizationPass(),                                     \
          "A(" #INPUT ");B(" #INPUT ");C(_MklNativeConv2D);"                 \
          "D(" #INPUT ");E(" #INPUT ");F(BiasAdd);G(Zeta)"                   \
          "|A->C;B->C:1;C->G;D->F;E->F:1;E->G:1");                           \
    }                                                                        \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_Dataflow2);
#undef REGISTER_TEST

// data_format attribute value mismatch. Merge should not be done
// in such case.
#define REGISTER_TEST(NAME, T, INPUT)                                      \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                  \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                         \
              "node { name: 'B' op: '" #INPUT "'}"                         \
              "node { name: 'C' op: 'Conv2D'"                              \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"      \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"       \
              " attr { key: 'strides'          value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " attr { key: 'padding'          value { s: 'SAME' } }"      \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, " \
              "i:1, i:1} } }"                                              \
              " input: ['A', 'B']}"                                        \
              "node { name: 'D' op: '" #INPUT "'}"                         \
              "node { name: 'E' op: 'BiasAdd'"                             \
              " attr { key: 'T'                value { type: " #T " } }"   \
              " attr { key: 'data_format'      value { s: 'NHCW' } }"      \
              " input: ['C', 'D'] }");                                     \
    if (!NativeFormatEnabled()) {                                            \
      EXPECT_EQ(DoMklLayoutOptimizationPass(),                               \
                "A(" #INPUT ");B(" #INPUT ");C(_MklConv2D);D(" #INPUT ");"   \
                "DMT/_0(Const);DMT/_1(Const);E(BiasAdd)|A->C;A:control->"    \
                "DMT/_0:control;A:control->DMT/_1:control;B->C:1;C->E;"      \
                "D->E:1;DMT/_0->C:2;DMT/_1->C:3");                           \
    } else {                                                                 \
      EXPECT_EQ(DoMklLayoutOptimizationPass(),                               \
                "A(" #INPUT ");B(" #INPUT ");C(_MklNativeConv2D);"           \
                "D(" #INPUT ");E(BiasAdd)|A->C;B->C:1;C->E;D->E:1");         \
    }                                                                        \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DWithBias_Negative_AttrMismatch);
#undef REGISTER_TEST

// Test set 2: BiasAddGrad + Conv2DBackpropFilter fusion tests

#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: 'Int32Input'}"                           \
              "node { name: 'C' op: '" #INPUT "'}"                           \
              "node { name: 'D' op: 'Conv2DBackpropFilter'"                  \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B', 'C'] }"                                    \
              "node { name: 'E' op: 'BiasAddGrad'"                           \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['C'] }");                                            \
    if (!NativeFormatEnabled()) {                                              \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
          "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
          "D(_MklConv2DBackpropFilterWithBias);DMT/_0(Const);DMT/_1(Const);"   \
          "DMT/_2(Const)|A->D;A:control->DMT/_0:control;A:control->"           \
          "DMT/_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;"            \
          "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                              \
    } else {                                                                   \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
          "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
          "D(_MklNativeConv2DBackpropFilterWithBias)|A->D;B->D:1;C->D:2");     \
    }                                                                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Positive);
#undef REGISTER_TEST

// BiasAddGrad fusion in the presence of BackpropFilter. But nodes do not match
// criteria for rewrite. So rewrite should not happen. 3rd input of
// Conv2DBackpropFilter is different than input to BiasAddGrad.
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: 'Int32Input'}"                           \
              "node { name: 'C' op: '" #INPUT "'}"                           \
              "node { name: 'D' op: 'Conv2DBackpropFilter'"                  \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B', 'C'] }"                                    \
              "node { name: 'E' op: 'BiasAddGrad'"                           \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['A'] }");                                            \
    if (!NativeFormatEnabled()) {                                              \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
          "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
          "D(_MklConv2DBackpropFilter);DMT/_0(Const);DMT/_1(Const);"           \
          "DMT/_2(Const);E(BiasAddGrad)|A->D;A->E;A:control->DMT/_0:control;"  \
          "A:control->DMT/_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;" \
          "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                              \
    } else {                                                                   \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
          "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
          "D(_MklNativeConv2DBackpropFilter);E(BiasAddGrad)"                   \
          "|A->D;A->E;B->D:1;C->D:2");                                         \
    }                                                                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Negative1);
#undef REGISTER_TEST

// BiasAddGrad fusion, but nodes do not match criteria for fusion.
// Different input formats.
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: 'Int32Input'}"                           \
              "node { name: 'C' op: '" #INPUT "'}"                           \
              "node { name: 'D' op: 'Conv2DBackpropFilter'"                  \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B', 'C'] }"                                    \
              "node { name: 'E' op: 'BiasAddGrad'"                           \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NHWC' } }"        \
              " input: ['A'] }");                                            \
    if (!NativeFormatEnabled()) {                                              \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
          "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                         \
          "D(_MklConv2DBackpropFilter);DMT/_0(Const);DMT/_1(Const);"           \
          "DMT/_2(Const);E(BiasAddGrad)|A->D;A->E;A:control->DMT/_0:control;"  \
          "A:control->DMT/_1:control;A:control->DMT/_2:control;B->D:1;C->D:2;" \
          "DMT/_0->D:3;DMT/_1->D:4;DMT/_2->D:5");                              \
    } else {                                                                   \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
           "A(" #INPUT ");B(Int32Input);C(" #INPUT ");"                        \
           "D(_MklNativeConv2DBackpropFilter);E(BiasAddGrad)"                  \
           "|A->D;A->E;B->D:1;C->D:2");                                        \
    }                                                                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Negative2);
#undef REGISTER_TEST

// BiasAddGrad fusion in the presence of BackpropFilter only. Fusion is done
// before node rewrite. Check this ordering.
#define REGISTER_TEST(NAME, T, INPUT)                                       \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                   \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                          \
              "node { name: 'B' op: '" #INPUT "'}"                          \
              "node { name: 'C' op: '" #INPUT "'}"                          \
              "node { name: 'M' op: '_MklInput'}"                           \
              "node { name: 'N' op: '_MklInput'}"                           \
              "node { name: 'O' op: '_MklInput'}"                           \
              "node { name: 'D' op: '_MklConv2DWithBias'"                   \
              " attr { key: 'T'                value { type: " #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"       \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"        \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "  \
              "i:1, i:1} } }"                                               \
              " attr { key: 'padding'          value { s: 'SAME' } }"       \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "  \
              "i:1, i:1} } }"                                               \
              " input: ['A', 'B', 'C', 'M', 'N', 'O']}"                     \
              "node { name: 'E' op: 'Zeta'"                                 \
              " attr {key: 'T'                 value { type: " #T " } }"    \
              " input: ['D', 'A']}"                                         \
              "node { name: 'F' op: 'Int32Input'}"                          \
              "node { name: 'G' op: '_MklConv2DBackpropFilter'"             \
              " attr { key: 'T'                value { type: " #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"       \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"        \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "  \
              "i:1, i:1} } }"                                               \
              " attr { key: 'padding'          value { s: 'SAME' } }"       \
              " input: ['E', 'F', 'A', 'M', 'N', 'O'] }"                    \
              "node { name: 'H' op: 'BiasAddGrad'"                          \
              " attr { key: 'T'                value { type: " #T " } }"    \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"       \
              " input: ['E'] }");                                           \
    EXPECT_EQ(                                                              \
        DoMklLayoutOptimizationPass(),                                      \
        "A(" #INPUT ");B(" #INPUT ");C(" #INPUT ");D(_MklConv2DWithBias);"  \
        "E(Zeta);F(Int32Input);G(_MklConv2DBackpropFilter);H(BiasAddGrad);" \
        "M(_MklInput);N(_MklInput);O(_MklInput)|A->D;A->E:1;A->G:2;B->D:1;" \
        "C->D:2;D->E;E->G;E->H;F->G:1;M->D:3;M->G:3;N->D:4;N->G:4;O->D:5;"  \
        "O->G:5");                                                          \
  }
REGISTER_TEST_ALL_TYPES(NodeMerge_Conv2DBackpropFilterFusion_Negative3);
#undef REGISTER_TEST

// C=Conv2D(A,B); E=BiasAdd(C,D); Y=Zeta(E,X);
// G=Conv2DBackpropInput(F,B,E)
// This is a case of node rewrite followed by node merge followed by connecting
// filter output of Conv2DWithBias to filter input of Conv2DBackpropInput.
#define REGISTER_TEST(NAME, T, INPUT)                                        \
  TEST_F(MklLayoutPassTest, NAME##_##T) {                                    \
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);      \
    InitGraph("node { name: 'A' op: '" #INPUT "'}"                           \
              "node { name: 'B' op: '" #INPUT "'}"                           \
              "node { name: 'C' op: 'Conv2D'"                                \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['A', 'B']}"                                          \
              "node { name: 'D' op: '" #INPUT "'}"                           \
              "node { name: 'E' op: 'BiasAdd'"                               \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " input: ['C', 'D'] }"                                         \
              "node { name: 'X' op: '" #INPUT "'}"                           \
              "node { name: 'Y' op: 'Zeta'"                                  \
              " attr {key: 'T'                 value { type: " #T " } }"     \
              " input: ['E', 'X']}"                                          \
              "node { name: 'F' op: 'Int32Input'}"                           \
              "node { name: 'G' op: 'Conv2DBackpropInput'"                   \
              " attr { key: 'T'                value { type: " #T " } }"     \
              " attr { key: 'data_format'      value { s: 'NCHW' } }"        \
              " attr { key: 'use_cudnn_on_gpu' value { b: false } }"         \
              " attr { key: 'strides'          value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " attr { key: 'padding'          value { s: 'SAME' } }"        \
              " attr { key: 'dilations'        value { list: {i: 1, i:1, "   \
              "i:1, i:1} } }"                                                \
              " input: ['F', 'B', 'E']}"                                     \
              "node { name: 'Z' op: 'Zeta'"                                  \
              " attr {key: 'T'                 value { type: " #T " } }"     \
              " input: ['G', 'X']}");                                        \
    if (!NativeFormatEnabled()) {                                              \
      EXPECT_EQ(                                                               \
          DoMklLayoutOptimizationPass(),                                       \
          "A(" #INPUT ");B(" #INPUT ");D(" #INPUT ");DMT/_0(Const);"           \
          "DMT/_1(Const);DMT/_2(Const);DMT/_3(Const);E(_MklConv2DWithBias);"   \
          "F(Int32Input);G(_MklConv2DBackpropInput);X(" #INPUT ");Y(Zeta);"    \
          "Z(Zeta)|A->E;A:control->DMT/_0:control;A:control->DMT/_1:control;"  \
          "A:control->DMT/_2:control;B->E:1;D->E:2;DMT/_0->E:3;"               \
          "DMT/_1->E:4;DMT/_2->E:5;DMT/_3->G:3;E->G:2;E->Y;E:1->G:1;E:2->G:5;" \
          "E:3->G:4;F->G;F:control->DMT/_3:control;G->Z;X->Y:1;X->Z:1");       \
    }                                                                          \
  }
// TODO(intel-tf): Enable bfloat16 test when we enable the operator.
REGISTER_TEST_FLOAT32(NodeMerge_Conv2DWithBias_ConvBpropInput_FilterFwd);
#undef REGISTER_TEST

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL && ENABLE_MKL
