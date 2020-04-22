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

// TODO(intel): Improve error handling in this file; instead of CHECK failing
// all over the place, we should log an error and execute the original graph.
#ifdef INTEL_MKL

#include "tensorflow/core/graph/mkl_layout_pass.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

// This pass implements rewriting of graph to support following scenarios:
// (A) Merging nodes in the graph
// (B) Rewriting a node in the graph to a new node
//     Rewrite happens under following scenario:
//     - Propagating Mkl layout as an additional output tensor
//        (we will loosely call a tensor that carries Mkl layout as Mkl tensor
//         henceforth.) from every Mkl supported NN layer.
//
// Example of A : Merging nodes in the graph
// -----------------------------------------
// Currently, we merge Conv2D+AddBias together. Consider Conv2D and BiasAdd as:
//
//           O = Conv2D(A, B)
//           P = BiasAdd(O, C)
//
// We merge them into Conv2DWithBias as:
//           P = _MklConv2DWithBias(A, A_m, B, B_m, C, C_m)
//
// The meaning of A_m, B_m and C_m is explained in B.1.
//
// Merge rules:
//  - The merge for Conv2D and BiasAdd happens when the output of Conv2D _only_
//    goes to BiasAdd.
//  - Also, the intersection of attributes of both the nodes must have same
//    values.
//  - Both the nodes must have been assigned to same device (if any).
//
// Example of B.1 : Rewriting nodes to Mkl nodes
// ---------------------------------------------
// Consider a Relu node. Current definition of Relu node looks like:
//
//           O = Relu(A)
//
// Relu has 1 input (A), and 1 output (O).
//
// This rewrite pass will generate a new graph node for Relu (new node is
// called MklRelu) as:
//
//          O, O_m = MklRelu(A, A_m)
//
// MklRelu has 2 inputs (A and A_m) and 2 outputs (O and O_m). Here input A is
// same as input A of Relu; output O is same as output O of Relu. O_m is the
// additional output tensor that will be set by MklRelu, and it represents
// Mkl tensor corresponding to O -- in other words, O_m is some kind of
// metadata for O. A_m is additional input of Relu, and it represents metadata
// for A - as O_m is metadata for O, A_m is metadata for A. MklRelu receives
// this metadata from previous node in the graph.
//
// When a previous node in the graph is an Mkl node, A_m will represent a valid
// Mkl tensor. But when a previous node is not an Mkl node, A_m will represent
// a dummy Mkl tensor.
//
// Rewriting rules:
//  - Selection of a node for rewriting happens by registering the op type of
//    the node with the rewriting pass. If the op type is not registered, then
//    all nodes of this op type will not be rewritten.
//  - Number of inputs after rewriting:
//      Since for every input Tensorflow tensor, the rewritten node gets Mkl
//      tensor(s), rewritten node gets 2*N inputs, where N is the number of
//      inputs for the original node.
//  - Number of outputs after rewriting:
//      Since for every output Tensorflow tensor, the rewritten node generates
//      Mkl tensor(s), the rewritten node generates 2*N outputs, where N is the
//      number of outputs of the original node.
//  - Ordering of Tensorflow tensors and Mkl tensors:
//      Since every rewritten node generates twice the number of inputs and
//      outputs, one could imagine various orderings among Tensorflow tensors
//      and Mkl tensors. E.g., assume an op 'Conv2D' that takes (A, B) as
//      inputs, then the new op '_MklConv2D' can take inputs A, B, A_m and B_m
//      in A, A_m, B, B_m order or it can also take them in A, B, A_m, B_m
//      order. Among N inputs one can get N! permutations.
//
//      So the question is: which order do we follow? We support 2 types of
//      orderings: (1) interleaved, and (2) contiguous. Interleaved ordering
//      follows an intuitive order where an Mkl tensor follows the
//      corresponding Tensorflow tensor immediately. In the context of the
//      above example, it will be: A, A_m, B, B_m. Note that the ordering rule
//      applies to both the inputs and outputs. Contiguous ordering means
//      all the Tensorflow tensors are contiguous followed by all the Mkl
//      tensors. We use contiguous ordering as default.
//
// Graph rewrite algorithm:
//      Algorithm: Graph Rewrite
//      Input: Graph G, Names of the nodes to rewrite and their new names
//      Output: Modified Graph G' if the nodes are modified, G otherwise.
//      Start:
//        N = Topological_Sort(G) // N is a set of nodes in toposort order.
//        foreach node n in N
//        do
//          if (Is_MKL_Op(n))  // Can this node accept an Mkl layout as input.
//          then
//            E = set of <incoming edge and its src_output slot> of n
//            E' = {}   // a new set of edges for rewritten node
//            foreach <e,s> in E
//            do
//              E' U {<e,s>}  // First copy edge which generates Tensorflow
//                            // tensor as it is
//              m = Source node of edge e
//              if Is_Rewritten(m)  // Did we rewrite this node in this pass?
//              then
//                E' U {<m,s+1>}    // If yes, then m will generate an Mkl
//                                  // tensor as an additional output.
//              else
//                d = Generate_Dummy_Mkl_Tensor()  // If not, generate a dummy
//                                                 // Mkl tensor.
//                E' U {<d,0>}  // The dummy Mkl tensor has only 1 output slot.
//              fi
//            done
//            n' = Build_New_Node(G,new_name,E')
//            Mark_Rewritten(n')  // Mark the new node as being rewritten.
//          fi
//        done
//
//      Explanation:
//        For graph rewrite, we visit nodes of the input graph in the
//        topological sort order. With this ordering, we visit nodes in the
//        top-to-bottom fashion. We need this order because while visiting a
//        node we want that all of its input nodes are visited and rewritten if
//        applicable. This is because if we need to rewrite a given node
//        then all of its input nodes need to be fixed (in other words they
//        cannot be deleted later.)
//
//        While visiting a node, we first check if the op type of the node is
//        an Mkl op. If it is, then we rewrite that node after constructing
//        new inputs to the node. If the op type of the node is not Mkl op,
//        then we do not rewrite that node.
//
// Handling workspace propagation for certain ops:
//
//        Certain backward ops in MKL (MaxPool, LRN and BatchNorm) require
//        passing of a workspace from their respective forward ops. Workspace
//        tensors provide memory for storing results of intermediate operations
//        which are helpful in backward propagation. TensorFlow does not have
//        a notion of a workspace and as a result does not allow producing
//        additional outputs from these forward ops. For these ops, we need
//        to add 2 extra edges between forward ops and their corresponding
//        backward ops - the first extra edge carries a workspace tensor and
//        the second one carries an Mkl tensor for the workspace tensor.
//
//        Example:
//
//        Typical graph for MaxPool and its gradient looks like:
//
//        A = MaxPool(T)
//        B = MaxPoolGrad(X, A, Y)
//
//        We will transform this graph to propagate the workspace as:
//        (with the contiguous ordering)
//
//        A, W, A_m, W_m = MklMaxPool(T, T_m)
//        B, B_m = MklMaxPoolGrad(X, A, Y, W, X_m, A_m, Y_m, W_m)
//
//        Here W is the workspace tensor. Transformed tensor names with the
//        suffix _m are Mkl tensors, and this transformation has been done
//        using the algorithm discussed earlier. The transformation for
//        workspace propagation only adds extra outputs (W, W_m) for a forward
//        op and connects them to the corresponding backward ops.
//
//        Terms:
//
//        Forward op name = name of the op in the forward pass
//          where a workspace tensor originates (MaxPool in this example)
//        Backward op name = name of the op in the backward pass that receives
//          a workspace tensor from the forward op (MaxPoolGrad in the example)
//        Slot = Position of the output or input slot that will be
//               used by the workspace tensor (1 for MklMaxPool as W is the 2nd
//               output of MaxPool (0 is 1st); 3 for MklMaxPoolGrad)
//
//        Question:
//
//        How do we associate a backward op to a forward op? There can be more
//        than one op with the exact same name.
//
//        In this example, we associate MaxPoolGrad with MaxPool. But there
//        could be more than one MaxPool ops. To solve this problem, we look
//        for _direct_ edge between a forward op and a backward op (tensor A is
//        flowing along this edge in the example).
//
//        How do we transform forward and backward ops when there is no direct
//        edge between them? In such a case, we generate dummy tensors for
//        workspace tensors. For the example, transformation of MaxPool will
//        be exactly same as it would be when there is a direct edge between
//        the forward and the backward op --- it is just that MaxPool won't
//        generate any workspace tensor. For MaxPoolGrad, the transformation
//        will also be same, but instead of connecting W and W_m with the
//        outputs of MaxPool, we will produce dummy tensors for them, and we
//        will set workspace_enabled attribute to false.
//
class MklLayoutRewritePass : public GraphOptimizationPass {
 public:
  MklLayoutRewritePass() {
    // NOTE: names are alphabetically sorted.
    csinfo_.addn = "AddN";
    csinfo_.avg_pool = "AvgPool";
    csinfo_.avg_pool_grad = "AvgPoolGrad";
    csinfo_.avg_pool3d = "AvgPool3D";
    csinfo_.avg_pool3d_grad = "AvgPool3DGrad";
    csinfo_.batch_matmul = "BatchMatMul";
    csinfo_.batch_matmul_v2 = "BatchMatMulV2";
    csinfo_.bias_add = "BiasAdd";
    csinfo_.bias_add_grad = "BiasAddGrad";
    csinfo_.concat = "Concat";
    csinfo_.concatv2 = "ConcatV2";
    csinfo_.conjugate_transpose = "ConjugateTranspose";
    csinfo_.conv2d = "Conv2D";
    csinfo_.conv2d_with_bias = "__MklDummyConv2DWithBias";
    csinfo_.conv2d_grad_input = "Conv2DBackpropInput";
    csinfo_.conv2d_grad_filter = "Conv2DBackpropFilter";
    csinfo_.conv2d_grad_filter_with_bias =
        "__MklDummyConv2DBackpropFilterWithBias";
    csinfo_.conv3d = "Conv3D";
    csinfo_.conv3d_grad_input = "Conv3DBackpropInputV2";
    csinfo_.conv3d_grad_filter = "Conv3DBackpropFilterV2";
    csinfo_.depthwise_conv2d = "DepthwiseConv2dNative";
    csinfo_.depthwise_conv2d_grad_input = "DepthwiseConv2dNativeBackpropInput";
    csinfo_.depthwise_conv2d_grad_filter =
        "DepthwiseConv2dNativeBackpropFilter";
    csinfo_.dequantize = "Dequantize";
    csinfo_.fused_batch_norm = "FusedBatchNorm";
    csinfo_.fused_batch_norm_grad = "FusedBatchNormGrad";
    csinfo_.fused_batch_norm_v2 = "FusedBatchNormV2";
    csinfo_.fused_batch_norm_grad_v2 = "FusedBatchNormGradV2";
    csinfo_.fused_batch_norm_v3 = "FusedBatchNormV3";
    csinfo_.fused_batch_norm_grad_v3 = "FusedBatchNormGradV3";
    csinfo_.fused_conv2d = "_FusedConv2D";
    csinfo_.fused_depthwise_conv2d = "_FusedDepthwiseConv2dNative";
    csinfo_.fused_matmul = "_FusedMatMul";
    csinfo_.identity = "Identity";
    csinfo_.leakyrelu = "LeakyRelu";
    csinfo_.leakyrelu_grad = "LeakyReluGrad";
    csinfo_.lrn = "LRN";
    csinfo_.lrn_grad = "LRNGrad";
    csinfo_.matmul = "MatMul";
    csinfo_.max_pool = "MaxPool";
    csinfo_.max_pool_grad = "MaxPoolGrad";
    csinfo_.max_pool3d = "MaxPool3D";
    csinfo_.max_pool3d_grad = "MaxPool3DGrad";
    csinfo_.mkl_conv2d = "_MklConv2D";
    csinfo_.mkl_conv2d_grad_input = "_MklConv2DBackpropInput";
    csinfo_.mkl_conv2d_grad_filter = "_MklConv2DBackpropFilter";
    csinfo_.mkl_conv2d_with_bias = "_MklConv2DWithBias";
    csinfo_.mkl_conv2d_grad_filter_with_bias =
        "_MklConv2DBackpropFilterWithBias";
    csinfo_.mkl_depthwise_conv2d_grad_input =
        "_MklDepthwiseConv2dNativeBackpropInput";
    csinfo_.mkl_depthwise_conv2d_grad_filter =
        "_MklDepthwiseConv2dNativeBackpropFilter";
    csinfo_.mkl_fused_conv2d = "_MklFusedConv2D";
    csinfo_.mkl_fused_depthwise_conv2d = "_MklFusedDepthwiseConv2dNative";
    csinfo_.mkl_fused_matmul = "_MklFusedMatMul";
    csinfo_.mkl_pad_with_conv2d = "_MklPadWithConv2D";
    csinfo_.mkl_pad_with_fused_conv2d = "_MklPadWithFusedConv2D";
    csinfo_.pad = "Pad";
    csinfo_.pad_with_conv2d = "__MklDummyPadWithConv2D";
    csinfo_.pad_with_fused_conv2d = "__MklDummyPadWithFusedConv2D";
    csinfo_.quantized_avg_pool = "QuantizedAvgPool";
    csinfo_.quantized_concatv2 = "QuantizedConcatV2";
    csinfo_.quantized_conv2d = "QuantizedConv2D";
    csinfo_.quantized_conv2d_per_channel = "QuantizedConv2DPerChannel";
    csinfo_.quantized_conv2d_with_requantize = "QuantizedConv2DAndRequantize";
    csinfo_.quantized_conv2d_with_bias = "QuantizedConv2DWithBias";
    csinfo_.quantized_conv2d_with_bias_and_requantize =
        "QuantizedConv2DWithBiasAndRequantize";
    csinfo_.quantized_conv2d_and_relu = "QuantizedConv2DAndRelu";
    csinfo_.quantized_conv2d_and_relu_and_requantize =
        "QuantizedConv2DAndReluAndRequantize";
    csinfo_.quantized_conv2d_with_bias_and_relu =
        "QuantizedConv2DWithBiasAndRelu";
    csinfo_.quantized_conv2d_with_bias_and_relu_and_requantize =
        "QuantizedConv2DWithBiasAndReluAndRequantize";
    csinfo_.quantized_max_pool = "QuantizedMaxPool";
    csinfo_.quantized_conv2d_with_bias_sum_and_relu =
        "QuantizedConv2DWithBiasSumAndRelu";
    csinfo_.quantized_conv2d_with_bias_sum_and_relu_and_requantize =
        "QuantizedConv2DWithBiasSumAndReluAndRequantize";
    csinfo_.quant_conv2d_with_bias_signed_sum_and_relu_and_requantize =
        "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize";
    csinfo_.quantized_matmul_with_bias = "QuantizedMatMulWithBias";
    csinfo_.quantized_matmul_with_bias_and_relu =
        "QuantizedMatMulWithBiasAndRelu";
    csinfo_.quantized_matmul_with_bias_and_relu_and_requantize =
        "QuantizedMatMulWithBiasAndReluAndRequantize";
    csinfo_.quantized_matmul_with_bias_and_dequantize =
        "QuantizedMatMulWithBiasAndDequantize";
    csinfo_.quantized_matmul_with_bias_and_requantize =
        "QuantizedMatMulWithBiasAndRequantize";
    csinfo_.quantized_depthwise_conv2d = "QuantizedDepthwiseConv2D";
    csinfo_.quantized_depthwise_conv2d_with_bias =
        "QuantizedDepthwiseConv2DWithBias";
    csinfo_.quantized_depthwise_conv2d_with_bias_and_relu =
        "QuantizedDepthwiseConv2DWithBiasAndRelu";
    csinfo_.quantized_depthwise_conv2d_with_bias_and_relu_and_requantize =
        "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize";
    csinfo_.quantize_v2 = "QuantizeV2";
    csinfo_.relu = "Relu";
    csinfo_.relu_grad = "ReluGrad";
    csinfo_.relu6 = "Relu6";
    csinfo_.relu6_grad = "Relu6Grad";
    csinfo_.requantize = "Requantize";
    csinfo_.tanh = "Tanh";
    csinfo_.tanh_grad = "TanhGrad";
    csinfo_.reshape = "Reshape";
    csinfo_.slice = "Slice";
    csinfo_.softmax = "Softmax";
    csinfo_.split = "Split";
    csinfo_.transpose = "Transpose";
    // Element-wise ops. Ensure you also add any new ops to IsOpElementWise
    // in the MklUtil.h (IsMklElementWiseOp method) to ensure that the
    // MklInputConversion op is added before it.
    csinfo_.add = "Add";
    csinfo_.add_v2 = "AddV2";
    csinfo_.maximum = "Maximum";
    csinfo_.mul = "Mul";
    csinfo_.squared_difference = "SquaredDifference";
    csinfo_.sub = "Sub";
    // End - element-wise ops. See note above.

    // NOTE: names are alphabetically sorted.
    rinfo_.push_back({csinfo_.addn, mkl_op_registry::GetMklOpName(csinfo_.addn),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.add, mkl_op_registry::GetMklOpName(csinfo_.add),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.add_v2,
                      mkl_op_registry::GetMklOpName(csinfo_.add_v2),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.avg_pool, mkl_op_registry::GetMklOpName(csinfo_.avg_pool),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.avg_pool_grad,
                      mkl_op_registry::GetMklOpName(csinfo_.avg_pool_grad),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.avg_pool3d, mkl_op_registry::GetMklOpName(csinfo_.avg_pool3d),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.avg_pool3d_grad,
                      mkl_op_registry::GetMklOpName(csinfo_.avg_pool3d_grad),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.batch_matmul,
                      mkl_op_registry::GetMklOpName(csinfo_.batch_matmul),
                      CopyAttrsAll, AlwaysRewrite, kRewriteForOpNameChange});
    rinfo_.push_back({csinfo_.batch_matmul_v2,
                      mkl_op_registry::GetMklOpName(csinfo_.batch_matmul_v2),
                      CopyAttrsAll, AlwaysRewrite, kRewriteForOpNameChange});
    rinfo_.push_back(
        {csinfo_.concat, mkl_op_registry::GetMklOpName(csinfo_.concat),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.concatv2, mkl_op_registry::GetMklOpName(csinfo_.concatv2),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.conjugate_transpose,
         mkl_op_registry::GetMklOpName(csinfo_.conjugate_transpose),
         CopyAttrsAll, AlwaysRewrite, kRewriteForOpNameChange});
    rinfo_.push_back({csinfo_.conv2d,
                      mkl_op_registry::GetMklOpName(csinfo_.conv2d),
                      CopyAttrsConvCheckConstFilter, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv2d_with_bias, csinfo_.mkl_conv2d_with_bias,
                      CopyAttrsConvCheckConstFilter, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv2d_grad_filter,
                      mkl_op_registry::GetMklOpName(csinfo_.conv2d_grad_filter),
                      CopyAttrsConv, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv2d_grad_filter_with_bias,
                      csinfo_.mkl_conv2d_grad_filter_with_bias, CopyAttrsConv,
                      AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv2d_grad_input,
                      mkl_op_registry::GetMklOpName(csinfo_.conv2d_grad_input),
                      CopyAttrsConv, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv3d,
                      mkl_op_registry::GetMklOpName(csinfo_.conv3d),
                      CopyAttrsConvCheckConstFilter, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv3d_grad_filter,
                      mkl_op_registry::GetMklOpName(csinfo_.conv3d_grad_filter),
                      CopyAttrsConv, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.conv3d_grad_input,
                      mkl_op_registry::GetMklOpName(csinfo_.conv3d_grad_input),
                      CopyAttrsConv, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.depthwise_conv2d,
                      mkl_op_registry::GetMklOpName(csinfo_.depthwise_conv2d),
                      CopyAttrsConv2DDepthwiseCheckConstFilter, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.depthwise_conv2d_grad_input,
         mkl_op_registry::GetMklOpName(csinfo_.depthwise_conv2d_grad_input),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.depthwise_conv2d_grad_filter,
         mkl_op_registry::GetMklOpName(csinfo_.depthwise_conv2d_grad_filter),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.dequantize, mkl_op_registry::GetMklOpName(csinfo_.dequantize),
         CopyAttrsAll, DequantizeRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.fused_batch_norm,
                      mkl_op_registry::GetMklOpName(csinfo_.fused_batch_norm),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.fused_batch_norm_grad,
         mkl_op_registry::GetMklOpName(csinfo_.fused_batch_norm_grad),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.fused_batch_norm_v2,
         mkl_op_registry::GetMklOpName(csinfo_.fused_batch_norm_v2),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.fused_batch_norm_grad_v2,
         mkl_op_registry::GetMklOpName(csinfo_.fused_batch_norm_grad_v2),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});

    // Using CopyAttrsAll for V3 on CPU, as there are no additional
    // attributes.
    rinfo_.push_back(
        {csinfo_.fused_batch_norm_v3,
         mkl_op_registry::GetMklOpName(csinfo_.fused_batch_norm_v3),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.fused_batch_norm_grad_v3,
         mkl_op_registry::GetMklOpName(csinfo_.fused_batch_norm_grad_v3),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.fused_conv2d, csinfo_.mkl_fused_conv2d,
                      CopyAttrsFusedConv2D, FusedConv2DRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.fused_depthwise_conv2d,
                      csinfo_.mkl_fused_depthwise_conv2d, CopyAttrsFusedConv2D,
                      FusedDepthwiseConv2DRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.fused_matmul, csinfo_.mkl_fused_matmul,
                      CopyAttrsAllCheckConstFilter, FusedMatMulRewrite});

    rinfo_.push_back({csinfo_.identity,
                      mkl_op_registry::GetMklOpName(csinfo_.identity),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.lrn, mkl_op_registry::GetMklOpName(csinfo_.lrn),
                      CopyAttrsAll, LrnRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.lrn_grad, mkl_op_registry::GetMklOpName(csinfo_.lrn_grad),
         CopyAttrsAll, LrnGradRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.matmul,
                      mkl_op_registry::GetMklOpName(csinfo_.matmul),
                      CopyAttrsAll, AlwaysRewrite, kRewriteForOpNameChange});
    rinfo_.push_back(
        {csinfo_.leakyrelu, mkl_op_registry::GetMklOpName(csinfo_.leakyrelu),
         CopyAttrsAll, LeakyReluRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.leakyrelu_grad,
                      mkl_op_registry::GetMklOpName(csinfo_.leakyrelu_grad),
                      CopyAttrsAll, LeakyReluRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.max_pool,
                      mkl_op_registry::GetMklOpName(csinfo_.max_pool),
                      CopyAttrsAll, NonDepthBatchWisePoolRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.max_pool_grad,
                      mkl_op_registry::GetMklOpName(csinfo_.max_pool_grad),
                      CopyAttrsAll, MaxpoolGradRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.max_pool3d,
                      mkl_op_registry::GetMklOpName(csinfo_.max_pool3d),
                      CopyAttrsAll, NonDepthBatchWisePoolRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.max_pool3d_grad,
                      mkl_op_registry::GetMklOpName(csinfo_.max_pool3d_grad),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.maximum,
                      mkl_op_registry::GetMklOpName(csinfo_.maximum),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.mul, mkl_op_registry::GetMklOpName(csinfo_.mul),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.pad_with_conv2d, csinfo_.mkl_pad_with_conv2d,
                      CopyAttrsPadWithConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.pad_with_fused_conv2d,
                      csinfo_.mkl_pad_with_fused_conv2d,
                      CopyAttrsPadWithFusedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_avg_pool,
                      mkl_op_registry::GetMklOpName(csinfo_.quantized_avg_pool),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_concatv2,
                      mkl_op_registry::GetMklOpName(csinfo_.quantized_concatv2),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_conv2d,
                      mkl_op_registry::GetMklOpName(csinfo_.quantized_conv2d),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_conv2d_per_channel,
         mkl_op_registry::GetMklOpName(csinfo_.quantized_conv2d_per_channel),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_conv2d_with_requantize,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_conv2d_with_requantize),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_conv2d_with_bias,
         mkl_op_registry::GetMklOpName(csinfo_.quantized_conv2d_with_bias),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_conv2d_with_bias_and_requantize,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_conv2d_with_bias_and_requantize),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_conv2d_and_relu,
         mkl_op_registry::GetMklOpName(csinfo_.quantized_conv2d_and_relu),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_conv2d_and_relu_and_requantize,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_conv2d_and_relu_and_requantize),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_conv2d_with_bias_and_relu,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_conv2d_with_bias_and_relu),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_conv2d_with_bias_and_relu_and_requantize,
         mkl_op_registry::GetMklOpName(
             csinfo_.quantized_conv2d_with_bias_and_relu_and_requantize),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_max_pool,
                      mkl_op_registry::GetMklOpName(csinfo_.quantized_max_pool),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_conv2d_with_bias_sum_and_relu,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_conv2d_with_bias_sum_and_relu),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_conv2d_with_bias_sum_and_relu_and_requantize,
         mkl_op_registry::GetMklOpName(
             csinfo_.quantized_conv2d_with_bias_sum_and_relu_and_requantize),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quant_conv2d_with_bias_signed_sum_and_relu_and_requantize,
         mkl_op_registry::GetMklOpName(
             csinfo_.quant_conv2d_with_bias_signed_sum_and_relu_and_requantize),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_matmul_with_bias,
         mkl_op_registry::GetMklOpName(csinfo_.quantized_matmul_with_bias),
         CopyAttrsQuantizedMatMulWithBias, AlwaysRewrite});
    rinfo_.push_back({csinfo_.quantized_matmul_with_bias_and_relu,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_matmul_with_bias_and_relu),
                      CopyAttrsQuantizedMatMulWithBias, AlwaysRewrite});
    rinfo_.push_back(
        {csinfo_.quantized_matmul_with_bias_and_relu_and_requantize,
         mkl_op_registry::GetMklOpName(
             csinfo_.quantized_matmul_with_bias_and_relu_and_requantize),
         CopyAttrsQuantizedMatMulWithBias, AlwaysRewrite});
    rinfo_.push_back({csinfo_.quantized_matmul_with_bias_and_requantize,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_matmul_with_bias_and_requantize),
                      CopyAttrsQuantizedMatMulWithBias, AlwaysRewrite});
    rinfo_.push_back({csinfo_.quantized_matmul_with_bias_and_dequantize,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_matmul_with_bias_and_dequantize),
                      CopyAttrsQuantizedMatMulWithBiasAndDequantize,
                      AlwaysRewrite});
    rinfo_.push_back(
        {csinfo_.quantized_depthwise_conv2d,
         mkl_op_registry::GetMklOpName(csinfo_.quantized_depthwise_conv2d),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantized_depthwise_conv2d_with_bias,
                      mkl_op_registry::GetMklOpName(
                          csinfo_.quantized_depthwise_conv2d_with_bias),
                      CopyAttrsQuantizedConv2D, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_depthwise_conv2d_with_bias_and_relu,
         mkl_op_registry::GetMklOpName(
             csinfo_.quantized_depthwise_conv2d_with_bias_and_relu),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.quantized_depthwise_conv2d_with_bias_and_relu_and_requantize,
         mkl_op_registry::GetMklOpName(
             csinfo_
                 .quantized_depthwise_conv2d_with_bias_and_relu_and_requantize),
         CopyAttrsQuantizedConv2D, AlwaysRewrite,
         kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.quantize_v2,
                      mkl_op_registry::GetMklOpName(csinfo_.quantize_v2),
                      CopyAttrsAll, QuantizeOpRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.relu, mkl_op_registry::GetMklOpName(csinfo_.relu),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.relu_grad, mkl_op_registry::GetMklOpName(csinfo_.relu_grad),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.relu6, mkl_op_registry::GetMklOpName(csinfo_.relu6),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.relu6_grad, mkl_op_registry::GetMklOpName(csinfo_.relu6_grad),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.requantize, mkl_op_registry::GetMklOpName(csinfo_.requantize),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    // Disable these two MKL operators for now due to some test failures caused
    // by these two ops
    /*
    rinfo_.push_back({csinfo_.tanh,
                      mkl_op_registry::GetMklOpName(csinfo_.tanh),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.tanh_grad,
                      mkl_op_registry::GetMklOpName(csinfo_.tanh_grad),
                      CopyAttrsAll, AlwaysRewrite,
                      kRewriteForLayoutPropagation});
    */
    rinfo_.push_back(
        {csinfo_.reshape, mkl_op_registry::GetMklOpName(csinfo_.reshape),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.slice,
                      mkl_op_registry::GetMklOpName(csinfo_.slice),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back(
        {csinfo_.softmax, mkl_op_registry::GetMklOpName(csinfo_.softmax),
         CopyAttrsAll, AlwaysRewrite, kRewriteForLayoutPropagation});

    rinfo_.push_back({csinfo_.squared_difference,
                      mkl_op_registry::GetMklOpName(csinfo_.squared_difference),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.sub, mkl_op_registry::GetMklOpName(csinfo_.sub),
                      CopyAttrsAll, RewriteIfAtleastOneMklInput,
                      kRewriteForLayoutPropagation});
    rinfo_.push_back({csinfo_.transpose,
                      mkl_op_registry::GetMklOpName(csinfo_.transpose),
                      CopyAttrsAll, AlwaysRewrite, kRewriteForOpNameChange});

    // Add info about which ops to add workspace edge to and the slots.
    wsinfo_.push_back({csinfo_.lrn, csinfo_.lrn_grad, 0, 2, 1, 3});
    wsinfo_.push_back({csinfo_.max_pool, csinfo_.max_pool_grad, 0, 1, 1, 3});
    wsinfo_.push_back(
        {csinfo_.max_pool3d, csinfo_.max_pool3d_grad, 0, 1, 1, 3});

    // Add a rule for merging nodes
    minfo_.push_back({csinfo_.conv2d, csinfo_.bias_add,
                      csinfo_.conv2d_with_bias, GetConv2DOrBiasAdd});

    minfo_.push_back({csinfo_.conv2d_grad_filter, csinfo_.bias_add_grad,
                      csinfo_.conv2d_grad_filter_with_bias,
                      GetConv2DBackpropFilterOrBiasAddGrad});
    // Merge Pad and Conv2d, only if the pad op is "Pad"
    // Doesn't merge if pad op is "PadV2" or "MirrorPad"
    minfo_.push_back(
        {csinfo_.pad, csinfo_.conv2d, csinfo_.pad_with_conv2d, GetPadOrConv2D});

    minfo_.push_back({csinfo_.pad, csinfo_.fused_conv2d,
                      csinfo_.pad_with_fused_conv2d, GetPadOrFusedConv2D});

    // The fusion patterns in "finfo_" that show up first will get applied
    // first, for example, graph "A->B->C-D" and finfo_ is {A->B->C to ABC,
    // A->B->C->D to ABCD}, since the first gets applied first, the final
    // graph will be ABC->D.

    //
    // Add rules to fuse sequences such as "Transpose (NCHW -> NHWC) + Conv2D
    // (NHWC) + Transpose (NHWC->
    // NCHW)" into "Conv2D (NCHW)". Such patterns occur frequently in Keras.
    // Note: we use the term "merge" to combine (exactly) 2 nodes into one,
    // while "fusion" is for 3+ nodes situation.
    //

    // Transpose + Conv2d + Transpose:
    std::vector<int> transpose_to_nhwc = {NCHW::dim::N, NCHW::dim::H,
                                          NCHW::dim::W, NCHW::dim::C};
    std::vector<int> transpose_to_nchw = {NHWC::dim::N, NHWC::dim::C,
                                          NHWC::dim::H, NHWC::dim::W};
    auto CheckForTransposeToNHWC =
        std::bind(CheckForTranspose, std::placeholders::_1, transpose_to_nhwc);
    auto CheckForConv2dOp =
        std::bind(CheckForMklOp, std::placeholders::_1, csinfo_.conv2d);
    auto CheckForTransposeToNCHW =
        std::bind(CheckForTranspose, std::placeholders::_1, transpose_to_nchw);
    auto FuseConv2D =
        std::bind(FuseTransposeMklOpTranspose, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, "NCHW");
    finfo_.push_back(
        {"transpose-elimination for Conv2D",
         {CheckForTransposeToNHWC, CheckForConv2dOp, CheckForTransposeToNCHW},
         // CheckForMklOp
         FuseConv2D,
         CopyAttrsConv});

    // Transpose + Conv3d + Transpose:
    std::vector<int> transpose_to_ndhwc = {NCDHW::dim::N, NCDHW::dim::D,
                                           NCDHW::dim::H, NCDHW::dim::W,
                                           NCDHW::dim::C};
    std::vector<int> transpose_to_ncdhw = {NDHWC::dim::N, NDHWC::dim::C,
                                           NDHWC::dim::D, NDHWC::dim::H,
                                           NDHWC::dim::W};

    auto CheckForTransposeToNDHWC =
        std::bind(CheckForTranspose, std::placeholders::_1, transpose_to_ndhwc);
    auto CheckForConv3dOp =
        std::bind(CheckForMklOp, std::placeholders::_1, csinfo_.conv3d);
    auto CheckForTransposeToNCDHW =
        std::bind(CheckForTranspose, std::placeholders::_1, transpose_to_ncdhw);
    auto FuseConv3D =
        std::bind(FuseTransposeMklOpTranspose, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, "NCDHW");

    finfo_.push_back(
        {"transpose-elimination for Conv3D",
         {CheckForTransposeToNDHWC, CheckForConv3dOp, CheckForTransposeToNCDHW},
         // CheckForMklOp
         FuseConv3D,
         CopyAttrsConv});

    auto CheckForMaxPool3DOp =
        std::bind(CheckForMklOp, std::placeholders::_1, csinfo_.max_pool3d);
    auto FuseMaxPool3D =
        std::bind(FuseTransposeMklOpTranspose, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3, "NCDHW");
    finfo_.push_back({"transpose-elimination for MaxPool3D",
                      {CheckForTransposeToNDHWC, CheckForMaxPool3DOp,
                       CheckForTransposeToNCDHW},
                      // CheckForMklOp
                      FuseMaxPool3D,
                      CopyAttrsPooling});
  }

  // Standard interface to run pass
  Status Run(const GraphOptimizationPassOptions& options);

  // Helper function which does most of heavy lifting for rewriting
  // Mkl nodes to propagate Mkl tensor as additional output
  //
  // Extracts common functionality between Run public interface and
  // test interface.
  //
  // @return true, if and only if graph is mutated; false otherwise.
  bool RunPass(std::unique_ptr<Graph>* g);

  /// Cause for rewrite
  /// Currently, we only support 2 causes - either for Mkl layout propagation
  /// which is the most common case, or for just a name change (used in case
  /// of ops like MatMul, Transpose, which do not support Mkl layout)
  enum RewriteCause { kRewriteForLayoutPropagation, kRewriteForOpNameChange };

  /// Structure to specify the name of an original node, its new name after
  /// rewrite, the number of inputs to the original node, the function to
  /// be used to copy attributes for the op, and the rule (if any) which
  /// must hold for rewriting the node
  typedef struct {
    string name;      // Original name of op of the node in the graph
    string new_name;  // New name of the op of the node in the graph
    // A function handler to copy attributes from an old node to a new node.
    std::function<void(const Node*, NodeBuilder*, bool)> copy_attrs;
    // A rule under which to rewrite this node
    std::function<bool(const Node*)> rewrite_rule;
    // Why are we rewriting?
    RewriteCause rewrite_cause;
  } RewriteInfo;

  /// Structure to specify a forward op, a backward op, and the slot numbers
  /// in the forward and backward ops where we will add a workspace edge.
  typedef struct {
    string fwd_op;    // Name of a forward op in the graph
    string bwd_op;    // Name of a backward op in the graph
    int fwd_slot;     // Output slot in the forward op node where actual
                      // output tensor resides
    int bwd_slot;     // Input slot in the backward op node where actual
                      // input tensor resides
    int ws_fwd_slot;  // Output slot in the forward op node where workspace
                      // edge is added
    int ws_bwd_slot;  // Input slot in the backward op node where workspace
                      // edge is added
  } WorkSpaceInfo;

  /// Structure to specify information used in node merge of 2 operators
  typedef struct {
    string op1;       // Node string for one operator.
    string op2;       // Node string for second operator.
    string new_node;  // Name of the node after merge
    // Function that enables user of the node merger to specify how to find
    // second operator given the first operator.
    std::function<Node*(const Node*)> get_node_to_be_merged;
  } MergeInfo;

  // Structure to specify information used in node fusion of 3+ operators
  typedef struct {
    std::string pattern_name;  // Name to describe this pattern, such as
                               // "Transpose_Mklop_Transpose".
    std::vector<std::function<bool(const Node*)> >
        node_checkers;  // Extra restriction checker for these ops
    std::function<Status(
        std::unique_ptr<Graph>*, std::vector<Node*>&,
        std::function<void(const Node*, NodeBuilder* nb, bool)>)>
        fuse_func;
    std::function<void(const Node*, NodeBuilder* nb, bool)> copy_attrs;
  } FusionInfo;

  //
  // Dimension indices for 2D tensor.
  //
  struct NCHW {
    enum dim { N = 0, C = 1, H = 2, W = 3 };
  };

  struct NHWC {
    enum dim { N = 0, H = 1, W = 2, C = 3 };
  };

  //
  // dimension indices for 3D tensor.
  //
  struct NCDHW {
    enum dim { N = 0, C = 1, D = 2, H = 3, W = 4 };
  };

  struct NDHWC {
    enum dim { N = 0, D = 1, H = 2, W = 3, C = 4 };
  };

  /// Structure to store all constant strings
  /// NOTE: names are alphabetically sorted.
  typedef struct {
    string addn;
    string add;
    string add_v2;
    string avg_pool;
    string avg_pool_grad;
    string avg_pool3d;
    string avg_pool3d_grad;
    string batch_matmul;
    string batch_matmul_v2;
    string bias_add;
    string bias_add_grad;
    string concat;
    string concatv2;
    string conjugate_transpose;
    string conv2d;
    string conv2d_with_bias;
    string conv2d_grad_input;
    string conv2d_grad_filter;
    string conv2d_grad_filter_with_bias;
    string conv3d;
    string conv3d_grad_input;
    string conv3d_grad_filter;
    string depthwise_conv2d;
    string depthwise_conv2d_grad_input;
    string depthwise_conv2d_grad_filter;
    string dequantize;
    string fused_batch_norm;
    string fused_batch_norm_grad;
    string fused_batch_norm_v2;
    string fused_batch_norm_grad_v2;
    string fused_batch_norm_v3;
    string fused_batch_norm_grad_v3;
    string fused_conv2d;
    string fused_depthwise_conv2d;
    string fused_matmul;
    string identity;
    string leakyrelu;
    string leakyrelu_grad;
    string lrn;
    string lrn_grad;
    string matmul;
    string max_pool;
    string max_pool_grad;
    string max_pool3d;
    string max_pool3d_grad;
    string maximum;
    string mkl_conv2d;
    string mkl_conv2d_grad_input;
    string mkl_conv2d_grad_filter;
    string mkl_conv2d_grad_filter_with_bias;
    string mkl_conv2d_with_bias;
    string mkl_depthwise_conv2d_grad_input;
    string mkl_depthwise_conv2d_grad_filter;
    string mkl_fused_conv2d;
    string mkl_fused_depthwise_conv2d;
    string mkl_fused_matmul;
    string mkl_pad_with_conv2d;
    string mkl_pad_with_fused_conv2d;
    string mul;
    string pad;
    string pad_with_conv2d;
    string pad_with_fused_conv2d;
    string quantized_avg_pool;
    string quantized_conv2d;
    string quantized_conv2d_per_channel;
    string quantized_conv2d_with_requantize;
    string quantized_conv2d_with_bias;
    string quantized_conv2d_with_bias_and_requantize;
    string quantized_conv2d_and_relu;
    string quantized_conv2d_and_relu_and_requantize;
    string quantized_conv2d_with_bias_and_relu;
    string quantized_conv2d_with_bias_and_relu_and_requantize;
    string quantized_concatv2;
    string quantized_max_pool;
    string quantized_conv2d_with_bias_sum_and_relu;
    string quantized_conv2d_with_bias_sum_and_relu_and_requantize;
    string quant_conv2d_with_bias_signed_sum_and_relu_and_requantize;
    string quantized_matmul_with_bias;
    string quantized_matmul_with_bias_and_relu;
    string quantized_matmul_with_bias_and_relu_and_requantize;
    string quantized_matmul_with_bias_and_requantize;
    string quantized_matmul_with_bias_and_dequantize;
    string quantized_depthwise_conv2d;
    string quantized_depthwise_conv2d_with_bias;
    string quantized_depthwise_conv2d_with_bias_and_relu;
    string quantized_depthwise_conv2d_with_bias_and_relu_and_requantize;
    string quantize_v2;
    string relu;
    string relu_grad;
    string relu6;
    string relu6_grad;
    string requantize;
    string tanh;
    string tanh_grad;
    string transpose;
    string reshape;
    string slice;
    string softmax;
    string split;
    string squared_difference;
    string sub;
  } ConstStringsInfo;

 private:
  /// Maintain info about nodes to rewrite
  std::vector<RewriteInfo> rinfo_;

  /// Maintain info about nodes to add workspace edge
  std::vector<WorkSpaceInfo> wsinfo_;

  /// Maintain info about nodes to be merged
  std::vector<MergeInfo> minfo_;

  /// Maintain info about nodes to be fused
  std::vector<FusionInfo> finfo_;

  /// Maintain structure of constant strings
  static ConstStringsInfo csinfo_;

 private:
  // Is OpDef::ArgDef a list type? It could be N * T or list(type).
  // Refer to opdef.proto for details of list type.
  inline bool ArgIsList(const OpDef::ArgDef& arg) const {
    return !arg.type_list_attr().empty() || !arg.number_attr().empty();
  }

  // Get length of a list in 'n' if 'arg' is of list type. Refer to
  // description of ArgIsList for definition of list type.
  inline int GetTensorListLength(const OpDef::ArgDef& arg, const Node* n) {
    CHECK_EQ(ArgIsList(arg), true);
    int N = 0;
    const string attr_name = !arg.type_list_attr().empty()
                                 ? arg.type_list_attr()
                                 : arg.number_attr();
    if (!arg.type_list_attr().empty()) {
      std::vector<DataType> value;
      TF_CHECK_OK(GetNodeAttr(n->def(), attr_name, &value));
      N = value.size();
    } else {
      TF_CHECK_OK(GetNodeAttr(n->def(), attr_name, &N));
    }
    return N;
  }

  // Can op represented by node 'n' run on DEVICE_CPU?
  // Op can run on CPU with MKL if the runtime assigned device or the
  // user requested device contains device CPU, or both are empty.
  bool CanOpRunOnCPUDevice(const Node* n) {
    bool result = true;
    string reason;

    // Substring that should be checked for in device name for CPU device.
    const char* const kCPUDeviceSubStr = "CPU";

    // If Op has been specifically assigned to a non-CPU device, then No.
    if (!n->assigned_device_name().empty() &&
        !absl::StrContains(n->assigned_device_name(), kCPUDeviceSubStr)) {
      result = false;
      reason = "Op has been assigned a runtime device that is not CPU.";
    }

    // If user has specifically assigned this op to a non-CPU device, then No.
    if (!n->def().device().empty() &&
        !absl::StrContains(n->def().device(), kCPUDeviceSubStr)) {
      result = false;
      reason = "User has assigned a device that is not CPU.";
    }

    if (result == false) {
      VLOG(1) << "MklLayoutRewritePass: Skipping rewriting of the node "
              << n->type_string() << ", reason: " << reason;
    }

    // Otherwise Yes.
    return result;
  }

  // Return a node that can be merged with input node 'n'
  //
  // @return pointer to the node if we can find such a
  // node. Otherwise, it returns nullptr.
  Node* CheckForNodeMerge(const Node* n) const;

  // Merge node 'm' with node 'n'.
  // Currently, we merge (1) Conv2D with BiasAdd, and (2) BiasAddGrad with
  // Conv2DBackpropFilter.
  //
  // Input nodes m and n may be deleted if the call to
  // this function is successful. Attempt to use the pointers
  // after the call to function may result in undefined behaviors.
  //
  // @input g - input graph, m - graph node, n - graph node to be merged with m
  // @return Status::OK(), if merging is successful and supported.
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case nodes are merged. Otherwise, it is
  //         not updated.
  Status MergeNode(std::unique_ptr<Graph>* g, Node* m, Node* n);

  // Helper function to merge different nodes
  Status MergeConv2DWithBiasAdd(std::unique_ptr<Graph>* g, Node* m, Node* n);
  Status MergePadWithConv2D(std::unique_ptr<Graph>* g, Node* m, Node* n);
  Status MergeConv2DBackpropFilterWithBiasAddGrad(std::unique_ptr<Graph>* g,
                                                  Node* m, Node* n);

  // Find BiasAdd or Conv2D node that can be merged with input node 'm'.
  // If input 'm' is BiasAdd, then check if there exists Conv2D node that can be
  // merged with 'm'. If input 'm' is Conv2D, then check if there exists BiasAdd
  // node that can be merged with 'm'.
  static Node* GetConv2DOrBiasAdd(const Node* m) {
    CHECK_NOTNULL(m);
    Node* n = nullptr;

    DataType T_m;
    TF_CHECK_OK(GetNodeAttr(m->def(), "T", &T_m));

    // Don't try to merge if datatype is not DT_FLOAT
    if (T_m != DT_FLOAT) return n;

    if (m->type_string() == csinfo_.bias_add) {
      // If a is BiasAdd, then Conv2D is 0th input of BiasAdd.
      TF_CHECK_OK(m->input_node(0, &n));
    } else {
      CHECK_EQ(m->type_string(), csinfo_.conv2d);
      // Go over all output edges and search for BiasAdd Node.
      // 0th input of BiasAdd is Conv2D.
      for (const Edge* e : m->out_edges()) {
        if (!e->IsControlEdge() &&
            e->dst()->type_string() == csinfo_.bias_add &&
            e->dst_input() == 0) {
          n = e->dst();
          break;
        }
      }
    }

    if (n == nullptr) {
      VLOG(1) << "MklLayoutRewritePass: Could not find matching "
              << "Conv2D and BiasAdd node for merging. Input node: "
              << m->DebugString();
    }

    return n;
  }

  // Find Pad or Conv2D node that can be merged with input node 'm'.
  // If input 'm' is Pad, then check if there exists Conv2D node that can be
  // merged with 'm'. If input 'm' is Conv2D, then check if there exists Pad
  // node that can be merged with 'm'.
  static Node* GetPadOrConv2D(const Node* m) {
    DCHECK(m);
    Node* n = nullptr;

    DataType T_m;
    TF_CHECK_OK(GetNodeAttr(m->def(), "T", &T_m));

    // Don't try to merge if datatype is not DT_FLOAT
    if (T_m != DT_FLOAT) return n;

    const Node* conv_node;
    if (m->type_string() == csinfo_.pad) {
      // If m is Pad, then Conv2D is the output of Pad.
      for (const Edge* e : m->out_edges()) {
        if (!e->IsControlEdge() && e->dst()->type_string() == csinfo_.conv2d) {
          n = e->dst();
          conv_node = n;
          break;
        }
      }
    } else {
      DCHECK_EQ(m->type_string(), csinfo_.conv2d);
      // If m is conv2D, Go over all input edges
      // and search for Pad  Node.
      for (const Edge* e : m->in_edges()) {
        if (!e->IsControlEdge() && e->src()->type_string() == csinfo_.pad) {
          n = e->src();
          conv_node = m;
          break;
        }
      }
    }
    // Check if only VALID type of padding is used
    // or not.
    if (n != nullptr) {
      string padding;
      TF_CHECK_OK(GetNodeAttr(conv_node->def(), "padding", &padding));
      if (padding != "VALID")
        // Then do not merge.
        // Only VALID type of padding in conv op can be
        // merged with Pad op.
        n = nullptr;
    } else {
      VLOG(1) << "MklLayoutRewritePass: Could not find matching "
              << "Pad and Conv2D node for merging. Input node: "
              << m->DebugString();
    }

    return n;
  }

  // Find Pad or _FusedConv2D node that can be merged with input node 'm'.
  // If input 'm' is Pad, then check if there exists _FusedConv2D node that can
  // be merged with 'm'. If input 'm' is _FusedConv2D, then check if there
  // exists Pad node that can be merged with 'm'.
  static Node* GetPadOrFusedConv2D(const Node* m) {
    DCHECK(m);
    Node* n = nullptr;

    const Node* conv_node;
    if (m->type_string() == csinfo_.pad) {
      // If m is Pad, then _FusedConv2D is the output of Pad.
      for (const Edge* e : m->out_edges()) {
        if (!e->IsControlEdge() &&
            e->dst()->type_string() == csinfo_.fused_conv2d) {
          n = e->dst();
          conv_node = n;
          break;
        }
      }
    } else {
      DCHECK_EQ(m->type_string(), csinfo_.fused_conv2d);
      // If m is _FusedConv2D, Go over all input edges
      // and search for Pad node.
      for (const Edge* e : m->in_edges()) {
        if (!e->IsControlEdge() && e->src()->type_string() == csinfo_.pad) {
          n = e->src();
          conv_node = m;
          break;
        }
      }
    }
    // Check if only VALID type of padding is used or not.
    if (n != nullptr) {
      string padding;
      TF_CHECK_OK(GetNodeAttr(conv_node->def(), "padding", &padding));
      if (padding != "VALID") {
        // Then do not merge.
        n = nullptr;
        VLOG(1) << "MklLayoutRewritePass: Could match Pad and _FusedConv2D "
                << "nodes but cannot merge them. Only conv ops with padding "
                << "type VALID can be merged with Pad op Input node: "
                << m->DebugString();
      }
    } else {
      VLOG(1) << "MklLayoutRewritePass: Could not find matching "
              << "Pad and _FusedConv2D node for merging. Input node: "
              << m->DebugString();
    }

    return n;
  }

  // Find Conv2DBackpropFilter or BiasAddGrad node that can be merged with input
  // node 'm'. If input 'm' is Conv2DBackpropFilter, then check if there exists
  // BiasAddGrad node that can be merged with 'm'. If input 'm' is BiasAddGrad,
  // then check if there exists Conv2DBackpropFilter node that can be merged
  // with 'm'.
  //
  // Graph that will allow us to connect Conv2DBackpropFilter with BiasAddGrad
  // would look like:
  //
  // _ = Conv2DBackpropFilter(F, _, G)
  // _ = BiasAddGrad(G)
  //
  // So 1st input of BiasAddGrad connects with 3rd input of
  // Conv2DBackpropFilter and vice versa.
  static Node* GetConv2DBackpropFilterOrBiasAddGrad(const Node* m) {
    CHECK_NOTNULL(m);
    Node* n = nullptr;

    DataType T_m;
    TF_CHECK_OK(GetNodeAttr(m->def(), "T", &T_m));

    // Don't try to merge if datatype is not DT_FLOAT
    if (T_m != DT_FLOAT) return n;

    if (m->type_string() == csinfo_.bias_add_grad) {
      // Get 1st input 'g' of BiasAddGrad.
      Node* g = nullptr;
      TF_CHECK_OK(m->input_node(0, &g));
      // Now traverse all outgoing edges from g that have destination node as
      // Conv2DBackpropFilter.
      for (const Edge* e : g->out_edges()) {
        if (!e->IsControlEdge() &&
            e->dst()->type_string() == csinfo_.conv2d_grad_filter &&
            e->dst_input() == 2 /* 3rd input of BackpropFilter */) {
          n = e->dst();
          break;
        }
      }
    } else {
      CHECK_EQ(m->type_string(), csinfo_.conv2d_grad_filter);
      // Get 3rd input 'g' of Conv2DBackpropFilter.
      Node* g = nullptr;
      TF_CHECK_OK(m->input_node(2, &g));
      // Now traverse all outgoing edges from g that have destination node as
      // BiasAddGrad.
      for (const Edge* e : g->out_edges()) {
        if (!e->IsControlEdge() &&
            e->dst()->type_string() == csinfo_.bias_add_grad &&
            e->dst_input() == 0 /* 1st input of BiasAddGrad */) {
          n = e->dst();
          break;
        }
      }
    }

    if (n == nullptr) {
      VLOG(1) << "MklLayoutRewritePass: Could not find matching "
              << "Conv2DBackpropFilter and BiasAddGrad node for merging. "
              << "Input node: " << m->DebugString();
    }
    return n;
  }

  // Return a node that can be fused with input node 'n'
  //
  // @return tuple. If we can find such nodes, the first
  // element of the tuple is a true. Otherwise, it's false.
  std::tuple<bool, std::vector<Node*>, const MklLayoutRewritePass::FusionInfo>
  CheckForNodeFusion(Node* n) const;

  // Fuse nodes in the vector "nodes"
  Status FuseNode(std::unique_ptr<Graph>* g, std::vector<Node*>& nodes,
                  const MklLayoutRewritePass::FusionInfo fi);

  // Fuse transpose(to "NHWC") + mklop("NHWC") + transpose(to "NCHW") into
  // mklop("NCHW").
  // Here "mklop" can be any MKL-DNN supported op, such as Conv2D.
  static Status FuseTransposeMklOpTranspose(
      std::unique_ptr<Graph>* g, std::vector<Node*>& nodes,
      std::function<void(const Node*, NodeBuilder* nb, bool)> copy_attrs,
      string data_format);

  static bool CheckForTranspose(const Node* node, std::vector<int> perm) {
    // Check if node's type is "Transpose"
    if (node->type_string() != "Transpose") return false;

    // If "Transpose" has multiple output data edges, also don't fuse it.
    if (node->num_outputs() > 1 || node->out_edges().size() > 1) return false;

    // Check if has out control edge. If true, this is a training graph.
    // Currently we focus on inference and do no fusion in training.
    // Note: this constraint will eventually be removed, if we enabled this
    // fusion for training
    // in the future.
    for (const Edge* e : node->out_edges()) {
      if (e->IsControlEdge()) {
        return false;
      }
    }

    // If "Transpose" has input control edges, don't fuse on it.
    for (const Edge* e : node->in_edges()) {
      if (e->IsControlEdge()) {
        return false;
      }
    }

    // We compared the tensor containing the permutation order ("perm_node")
    // with our desired order ("perm"). If they're exactly match, this check
    // succeed and returns true.
    for (const Edge* e : node->in_edges()) {
      if (!e->IsControlEdge()) {
        const Node* perm_node = e->src();

        const int kPermTensorIndex = 1;
        if (perm_node->type_string() == "Const" &&
            e->dst_input() == kPermTensorIndex) {
          // we find the "perm" node, now try to retrieve its value.
          const TensorProto* proto = nullptr;
          TF_CHECK_OK(GetNodeAttr(perm_node->def(), "value", &proto));

          DataType type;
          TF_CHECK_OK(GetNodeAttr(perm_node->def(), "dtype", &type));

          Tensor tensor;
          if (!tensor.FromProto(*proto)) {
            TF_CHECK_OK(errors::InvalidArgument(
                "Could not construct Tensor from TensorProto in node: ",
                node->name()));
            return false;
          }
          // Current fusion only supports 4D or 5D tensors according to `perm`
          // vector, return false otherwise.
          if (tensor.dim_size(0) != perm.size()) return false;
          DCHECK_EQ(tensor.dims(), 1);
          if (type == DT_INT32) {
            const auto tensor_content = tensor.flat<int>().data();
            for (int i = 0; i < perm.size(); ++i)
              if (tensor_content[i] != perm[i]) return false;
            return true;
          } else if (type == DT_INT64) {
            const auto tensor_content = tensor.flat<int64>().data();
            for (int i = 0; i < perm.size(); ++i)
              if (tensor_content[i] != perm[i]) return false;
            return true;
          }
          return false;
        }
      }
    }
    return false;
  }

  static bool CheckForMklOp(const Node* node, string name = "") {
    if (node == nullptr) return false;

    if (!name.empty() && node->type_string() != name) {
      return false;
    }

    // if mklop has multiple outputs, don't fuse it.
    if (node->num_outputs() > 1) return false;

    if (node->out_edges().size() > 1) return false;

    DataType T;
    TF_CHECK_OK(GetNodeAttr(node->def(), "T", &T));
    return mkl_op_registry::IsMklLayoutDependentOp(
        mkl_op_registry::GetMklOpName(node->type_string()), T);
  }

  // Check if the node 'n' has any applicable rewrite rule
  // We check for 2 scenarios for rewrite.
  //
  // @return RewriteInfo* for the applicable rewrite rule
  const RewriteInfo* CheckForNodeRewrite(const Node* n) const;
  const RewriteInfo* CheckForQuantizedNodeRewrite(const Node* n) const;

  // Default rewrite rule to be used in scenario 1 for rewrite.
  // @return - true (since we want to always rewrite)
  static bool AlwaysRewrite(const Node* n) { return true; }

  // Rewrite rule which considers "context" of the current node to decide if we
  // should rewrite. By "context" we currently mean all the inputs of current
  // node. The idea is if none of the inputs of current node are not MKL nodes,
  // then rewriting current node to MKL node _may not_ offer any performance
  // improvement.
  //
  // One such case is element-wise ops. For such ops, we reuse the Eigen
  // implementation and pass the MKL metadata tensor through so we can avoid
  // conversions. However, if all incoming edges are in TF format, we don't
  // need all this overhead, so replace the elementwise node only if at least
  // one of its parents is a MKL node.
  //
  // More generally, all memory- or IO-bound ops (such as Identity) may fall
  // under this category.
  //
  // @input - Input graph node to be rewritten
  // @return - true if node is to be rewritten as MKL node; false otherwise.
  static bool RewriteIfAtleastOneMklInput(const Node* n) {
    DataType T;
    if (GetNodeAttr(n->def(), "T", &T).ok() &&
        mkl_op_registry::IsMklOp(
            mkl_op_registry::GetMklOpName(n->type_string()), T)) {
      for (auto e : n->in_edges()) {
        if (e->IsControlEdge()) continue;
        if (mkl_op_registry::IsMklOp(e->src())) {
          return true;
        }
      }
    }
    return false;
  }

  static bool DequantizeRewrite(const Node* n) {
    DCHECK(n);
    Node* input = nullptr;
    TF_CHECK_OK(n->input_node(0, &input));
    string mode_string;
    TF_CHECK_OK(GetNodeAttr(n->def(), "mode", &mode_string));
    if (mode_string != "SCALED") {
      VLOG(1) << "DequantizeRewrite: Mode is not SCALED. "
              << "This case is not optimized by Intel MKL kernel, thus using "
                 "Eigen op for Dequantize op.";
      return false;
    }
    if (input->IsConstant()) {
      VLOG(1) << "DequantizeRewrite: Trying to dequantize a Const node which "
              << "could possibly be a filter. "
              << "This case is not supported by Intel MKL kernel, thus using "
                 "Eigen op for Dequantize op.";
      return false;
    }
    return true;
  }

  // Rewrite rule for _FusedMatMul.
  // @return - true (no transpose attribute for input 1);
  //           false otherwise.
  static bool FusedMatMulRewrite(const Node* n) {
    bool trans_a;

    // Do not rewrite with transpose attribute because reorder has performance
    // impact.
    TF_CHECK_OK(GetNodeAttr(n->def(), "transpose_a", &trans_a));

    return !trans_a;
  }

  // Check if we are performing pooling on depth or batch. If it is, then we
  // do not rewrite MaxPool node to Mkl version.
  // @return - true (if it is not a depth/batch wise pooling case);
  //           false otherwise.
  static bool NonDepthBatchWisePoolRewrite(const Node* n) {
    CHECK_NOTNULL(n);

    string data_format_str;
    TensorFormat data_format;
    std::vector<int32> ksize, strides;
    TF_CHECK_OK(GetNodeAttr(n->def(), "ksize", &ksize));
    TF_CHECK_OK(GetNodeAttr(n->def(), "strides", &strides));
    TF_CHECK_OK(GetNodeAttr(n->def(), "data_format", &data_format_str));
    bool result = FormatFromString(data_format_str, &data_format);
    DCHECK(result);

    // Condition that specifies non-batch-wise and non-depth-wise pooling.
    if (GetTensorDim(ksize, data_format, 'N') == 1 &&
        GetTensorDim(strides, data_format, 'N') == 1 &&
        GetTensorDim(ksize, data_format, 'C') == 1 &&
        GetTensorDim(strides, data_format, 'C') == 1) {
      return true;
    }

    return false;
  }

  // If the depth_radius of LRN is not 2, then MKL DNN takes unoptimized
  // path. The unoptimized path is slow. Thus we dont rewrite the node
  // and use default Eigen. But for depth_radius=2, MKL DNN optimized
  // path is taken, i.e., eigen node is rewritten by MKl DNN node.
  static bool LrnRewrite(const Node* n) {
    CHECK_NOTNULL(n);

    int depth_radius;
    TF_CHECK_OK(GetNodeAttr(n->def(), "depth_radius", &depth_radius));

    // if the depth_radius of LRN is not 2, don't rewrite the node by MKL DNN
    // and use eigen node instead
    if (depth_radius == 2) {
      return true;
    }
    VLOG(1) << "LrnRewrite: The model sets depth_radius as not 2 which"
            << "case is not optimized by Intel MKL, thus using Eigen op"
            << "for LRN ";

    return false;
  }

  static bool LrnGradRewrite(const Node* n) {
    CHECK_NOTNULL(n);
    bool do_rewrite = false;

    for (const Edge* e : n->in_edges()) {
      // Rewrite only if there is corresponding LRN, i.e workspace is available
      if (e->dst()->type_string() == csinfo_.lrn_grad && e->dst_input() == 2 &&
          e->src()->type_string() ==
              mkl_op_registry::GetMklOpName(csinfo_.lrn) &&
          e->src_output() == 0) {
        do_rewrite = true;
        break;
      }
    }
    return do_rewrite;
  }

  // MKL-DNN's LeakyRelu(feature) = feature          (if feature > 0), or
  //                                feature * alpha  (otherwise),
  // while TensorFlow's LeakyRelu(feature) = max(feature, feature * alpha).
  // These two algorithms are not consistent when alpha > 1,
  // so we only rewrite LeakyRelu to MKL OP when alpha <= 1.
  static bool LeakyReluRewrite(const Node* n) {
    DCHECK(n);

    float alpha;
    bool has_attr = TryGetNodeAttr(n->def(), "alpha", &alpha);
    DCHECK(has_attr);

    // If the alpha of LeakyRelu is less than 1, rewrite the node.
    // Otherwise eigen node is used instead.
    if (alpha <= 1) {
      return true;
    }
    VLOG(1) << "LeakyReluRewrite: The model sets alpha is greater than 1 "
            << "which case is not optimized by Intel MKL, thus using Eigen op"
            << "for LeakyRelu ";

    return false;
  }

  static bool QuantizeOpRewrite(const Node* n) {
    DCHECK(n);
    Node* filter_node = nullptr;
    TF_CHECK_OK(n->input_node(0, &filter_node));
    bool narrow_range = false;
    int axis = -1;
    string mode_string;
    string round_mode_string;
    DataType type;
    TryGetNodeAttr(n->def(), "narrow_range", &narrow_range);
    TryGetNodeAttr(n->def(), "axis", &axis);
    TF_CHECK_OK(GetNodeAttr(n->def(), "mode", &mode_string));
    TF_CHECK_OK(GetNodeAttr(n->def(), "round_mode", &round_mode_string));
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &type));

    if (narrow_range) {
      VLOG(1) << "QuantizeOpRewrite: narrow range is enabled for quantization."
              << "This case is not optimized by Intel MKL, "
              << "thus using Eigen op for Quantize op ";
      return false;
    }
    if (axis != -1) {
      VLOG(1) << "QuantizeOpRewrite: dimension is specified for "
              << "per slice quantization."
              << "This case is not optimized by Intel MKL, "
              << "thus using Eigen op for Quantize op ";
      return false;
    }
    if (!((mode_string == "SCALED" && round_mode_string == "HALF_TO_EVEN") ||
          (mode_string == "MIN_FIRST"))) {
      VLOG(1) << "QuantizeOpRewrite: Mode is not SCALED or MIN_FIRST and/or"
              << "rounding mode is not HALF_TO_EVEN. "
              << "This case is not optimized by Intel MKL, thus using Eigen op"
              << "for Quantize op ";
      return false;
    }
    if (filter_node->IsConstant()) {
      VLOG(1) << "QuantizeOpRewrite: Trying to quantize a node which "
              << "is a constant. "
              << "This case is not supported by the kernel, thus using Eigen op"
              << "for Quantize op ";

      return false;
    }
    if (mode_string == "MIN_FIRST") {
      if (type != DT_QUINT8) {
        VLOG(1) << "QuantizeOpRewrite: For MIN_FIRST mode the data type is "
                << "not DT_UINT8. This case is not optimized by Intel MKL, "
                << "thus using Eigen op for Quantize op ";
        return false;
      }
    }
    return true;
  }
  static bool MaxpoolGradRewrite(const Node* n) {
    CHECK_NOTNULL(n);
    bool do_rewrite = false;
    for (const Edge* e : n->in_edges()) {
      // Rewrite only if there is corresponding Maxpool, i.e workspace is
      // available
      if (e->dst()->type_string() == csinfo_.max_pool_grad &&
          e->dst_input() == 1 &&
          e->src()->type_string() ==
              mkl_op_registry::GetMklOpName(csinfo_.max_pool) &&
          e->src_output() == 0) {
        do_rewrite = true;
        break;
      }
    }
    return do_rewrite;
  }

  static bool FusedConv2DRewrite(const Node* n) {
    // MKL DNN currently doesn't support all fusions that grappler fuses
    // together with Conv2D (ex. batchnorm). We rewrite _FusedConv2D only if
    // it includes those we support.
    DataType T;
    if (!TryGetNodeAttr(n->def(), "T", &T) ||
        !mkl_op_registry::IsMklLayoutDependentOp(csinfo_.mkl_fused_conv2d, T)) {
      return false;
    }

    std::vector<string> fused_ops;
    TF_CHECK_OK(GetNodeAttr(n->def(), "fused_ops", &fused_ops));
    return (fused_ops == std::vector<string>{"BiasAdd"} ||
            fused_ops == std::vector<string>{"Relu"} ||
            fused_ops == std::vector<string>{"Relu6"} ||
            fused_ops == std::vector<string>{"Elu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Elu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"});
  }

  static bool FusedDepthwiseConv2DRewrite(const Node* n) {
    // MKL DNN currently doesn't support all fusions that grappler fuses
    // together with DepthwiseConv2D (ex. batchnorm). We rewrite
    // _FusedDepthwiseConv2DNative only if it includes those we support.
    DataType T;
    if (!TryGetNodeAttr(n->def(), "T", &T) ||
        !mkl_op_registry::IsMklLayoutDependentOp(
            csinfo_.mkl_fused_depthwise_conv2d, T)) {
      return false;
    }

    std::vector<string> fused_ops;
    TF_CHECK_OK(GetNodeAttr(n->def(), "fused_ops", &fused_ops));
    return (fused_ops == std::vector<string>{"BiasAdd"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Elu"});
  }

  // Rewrites input node to a new node specified by its matching rewrite info.
  //
  // Method first searches matching rewrite info for input node and then
  // uses that info to rewrite.
  //
  // Input node may be deleted in case of rewrite. Attempt to use the node
  // after the call can result in undefined behaviors.
  //
  // @input  g - input graph, n - Node to be rewritten,
  //         ri - matching rewriteinfo
  // @return Status::OK(), if the input node is rewritten;
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case the input node is rewritten.
  //         Otherwise, it is not updated.
  Status RewriteNode(std::unique_ptr<Graph>* g, Node* n, const RewriteInfo* ri);

  // Rewrites input node to just change its operator name. The number of
  // inputs to the node and the number of outputs remain the same. Attributes
  // of the new node could be copied from attributes of the old node or
  // modified. copy_attrs field of RewriteInfo controls this.
  //
  // Conceptually, it allows us to rewrite:
  //
  //        f[a=v1,b=v2](x,y) -> g[a'=v3,b'=v4](x,y)
  //
  // Attributes can be altered without any restrictions --- they could be
  // copied, modified, or deleted completely.
  //
  // @input  g - input graph, orig_node - Node to be rewritten,
  //         ri - matching rewriteinfo
  // @output new_node - points to newly created node
  // @return Status::OK(), if the input node is rewritten;
  //         Returns appropriate Status error code otherwise.
  //         Graph is only updated when the input node is rewritten.
  Status RewriteNodeForJustOpNameChange(std::unique_ptr<Graph>* g,
                                        const Node* orig_node, Node** new_node,
                                        const RewriteInfo* ri);

  // Rewrites input node to enable MKL layout propagation. Please also refer to
  // documentation for the function RewriteNodeForJustOpNameChange() to
  // understand what it means.
  //
  // @input  g - input graph, orig_node - Node to be rewritten,
  //         ri - matching rewriteinfo
  // @output new_node - points to newly created node
  // @return Status::OK(), if the input node is rewritten;
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case the input node is rewritten.
  //         Otherwise, it is not updated.
  Status RewriteNodeForLayoutPropagation(std::unique_ptr<Graph>* g,
                                         const Node* orig_node, Node** new_node,
                                         const RewriteInfo* ri);

  // Get nodes that will feed a list of TF tensors to the new
  // node that we are constructing.
  //
  // @input g - input graph,
  // @input inputs - inputs to old node that we are using for constructing
  //                 new inputs,
  // @input input_idx - the index in the 'inputs' vector pointing to the
  //                    current input that we have processed so far
  // @output input_idx - index will be incremented by the number of nodes
  //                     from 'inputs' that are processed
  // @input list_length - The expected length of list of TF tensors
  // @output output_nodes - the list of new nodes creating TF tensors
  //
  // @return None
  void GetNodesProducingTFTensorList(
      const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
      int* input_idx, int list_length,
      std::vector<NodeBuilder::NodeOut>* output_nodes);

  // Get nodes that will feed a list of Mkl tensors to the new
  // node that we are constructing.
  //
  // @input g - input graph,
  // @input orig_node - Original node that we are rewriting
  // @input inputs - inputs to old node that we are using for constructing
  //                 new inputs,
  // @input input_idx - the index in the 'inputs' vector pointing to the
  //                    current input that we have processed so far
  // @output input_idx - index will be incremented by the number of nodes
  //                     from 'inputs' that are processed
  // @input list_length - The expected length of list of Mkl tensors
  // @output output_nodes - the list of new nodes creating Mkl tensors
  //
  // @return None
  void GetNodesProducingMklTensorList(
      std::unique_ptr<Graph>* g, const Node* orig_node,
      const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
      int* input_idx, int list_length,
      std::vector<NodeBuilder::NodeOut>* output_nodes);

  // Get a node that will feed an Mkl tensor to the new
  // node that we are constructing. The output node could be (1) 'n'
  // if it is Mkl layer, or (2) a dummy node producing dummy Mkl tensor
  // if 'n' is not an Mkl layer.
  //
  // @input g - input graph,
  // @input orig_node - Original node that we are rewriting,
  // @input n - Node based on which we are creating Mkl node,
  // @input n_output_slot - the output slot of node 'n'
  //            which is feeding to the node that we are constructing
  // @output mkl_node - the new node that will feed Mkl tensor
  // @output mkl_node_output_slot - the slot number of mkl_node that
  //                                will feed the tensor
  // @return None
  void GetNodeProducingMklTensor(std::unique_ptr<Graph>* g,
                                 const Node* orig_node, Node* n,
                                 int n_output_slot, Node** mkl_node,
                                 int* mkl_node_output_slot);

  // Setup new inputs using old inputs 'inputs' for the rewritten node in 'nb'
  // in graph 'g'. Original node is input in 'old_node'. Inputs to 'nb' are
  // set up in contiguous fashion. 'workspace_tensors' carry graph nodes
  // producing workspace edges if 'are_workspace_tensors_available' is true.
  // Otherwise, 'workspace_tensors' is empty vector.
  //
  // For details, refer to 'Ordering of inputs after rewriting' section in the
  // documentation above.
  //
  // Returns Status::OK() if setting up inputs is successful, otherwise
  // returns appropriate status code.
  int SetUpContiguousInputs(
      std::unique_ptr<Graph>* g,
      const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
      NodeBuilder* nb, const Node* old_node,
      std::vector<NodeBuilder::NodeOut>* workspace_tensors,
      bool are_workspace_tensors_available);

  // Setup new inputs using old inputs 'inputs' for the rewritten node in 'nb'
  // in graph 'g'. Original node is input in 'orig_node'.
  //
  // For details, refer to 'Ordering of Tensorflow tensors and Mkl tensors'
  // section in the documentation above.
  //
  // Returns Status::OK() if setting up inputs is successful, otherwise
  // returns appropriate status code.
  Status SetUpInputs(std::unique_ptr<Graph>* g,
                     const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
                     NodeBuilder* nb, const Node* orig_node);

  // Create new inputs by copying old inputs 'inputs' for the rewritten node
  // in 'nb' in graph 'g'. Original node is input in 'orig_node'. This is mostly
  // used in the context of rewrite for just operator name change in which
  // inputs of old operator and new operator are same.
  //
  // Returns Status::OK() if setting up inputs is successful, otherwise
  // returns appropriate status code.
  Status CopyInputs(const Node* orig_node,
                    const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
                    NodeBuilder* nb);

  // Add workspace edge on the input or output side of Node 'orig_node' by using
  // NodeBuilder 'nb' for the new node provided. If 'orig_node' does not dictate
  // adding workspace edge then do not add it. Workspace Tensorflow and Mkl
  // tensors, if they need to be added, will be set into these tensors.
  // If we set workspace tensors, then are_ws_tensors_added should be true.
  void AddWorkSpaceEdgeIfNeeded(std::unique_ptr<Graph>* g,
                                const Node* orig_node, NodeBuilder* nb,
                                std::vector<NodeBuilder::NodeOut>* ws_tensors,
                                bool* are_ws_tensors_added);

  // Helper function used by FixMklMetaDataEdges. Fixes the metadata edge
  // pointed by 'e_metadata' corresponding to the data edge 'e_data' in graph
  // 'g'. Returns true is fixup was done; otherwise, it returns false.
  bool FixMklMetaDataEdgeIfNeeded(std::unique_ptr<Graph>* g, const Edge* e_data,
                                  const Edge* e_metadata);

  // Are the input Mkl metadata edges for node 'n' in graph 'g' correctly
  // connected? If not, then fix them. This is needed because a graph may have
  // some input Mkl metadata edges incorrectly setup after node merge and
  // rewrite passes. This could happen because GetReversePostOrder function may
  // not provide topologically sorted order if a graph contains cycles. The
  // function returns true if at least one Mkl metadata edge for node 'n' was
  // fixed. Otherwise, it returns false.
  //
  // Example:
  //
  // X = MklConv2D(_, _, _)
  // Y = MklConv2DWithBias(_, _, _, _, _, _)
  // Z = MklAdd(X, Y, DummyMklTensor, Y:1)
  //
  // For a graph such as shown above, note that 3rd argument of MklAdd contains
  // DummyMklTensor. Actually, it should be getting the Mkl metadata from
  // MklConv2D op (specifically, X:2). This incorrect plumbing could be possible
  // (although rare) if the Mkl NodeMerge + NodeRewrite passes visit Z before X
  // (possible if X, Y, Z are part of a loop.) This function fixes the Mkl
  // metadata edges only - it does not rewrite nodes nor does it modify the Mkl
  // data edges (1st and 2nd arguments of MklAdd).
  bool FixMklMetaDataEdges(std::unique_ptr<Graph>* g, Node* n);

  // Functions specific to operators to copy attributes
  // We need operator-specific function to copy attributes because the framework
  // does not provide any generic function for it.
  // NOTE: names are alphabetically sorted.
  static void CopyAttrsAll(const Node* orig_node, NodeBuilder* nb,
                           bool change_format = false);
  static void CopyAttrsAllCheckConstFilter(const Node* orig_node,
                                           NodeBuilder* nb,
                                           bool change_format = false);

  static void CopyAttrsConv(const Node* orig_node, NodeBuilder* nb,
                            bool change_format = false);
  static void CopyAttrsConv2DDepthwiseCheckConstFilter(
      const Node* orig_node, NodeBuilder* nb, bool change_format = false);
  static void CopyAttrsConvCheckConstFilter(const Node* orig_node,
                                            NodeBuilder* nb,
                                            bool change_format = false);
  static void CopyAttrsFusedConv2D(const Node* orig_node, NodeBuilder* nb,
                                   bool change_format = false);
  static void CopyAttrsPadWithConv2D(const Node* orig_node, NodeBuilder* nb,
                                     bool change_format = false);
  static void CopyAttrsPadWithFusedConv2D(const Node* orig_node,
                                          NodeBuilder* nb,
                                          bool change_format = false);
  static void CopyAttrsFromPadAndConv2D(const Node* orig_node1,
                                        const Node* orig_node2, NodeBuilder* nb,
                                        bool change_format = false);
  static void CopyAttrsFromPadAndFusedConv2D(const Node* orig_node1,
                                             const Node* orig_node2,
                                             NodeBuilder* nb,
                                             bool change_format = false);
  static void CopyAttrsQuantizedConv2D(const Node* orig_node, NodeBuilder* nb,
                                       bool change_format = false);
  static void CopyFormatAttrsConv(const Node* orig_node, NodeBuilder* nb,
                                  const std::vector<int32>& strides,
                                  const std::vector<int32>& dilations,
                                  bool change_format = false);

  static void CopyAttrsQuantizedMatMulWithBias(const Node* orig_node,
                                               NodeBuilder* nb,
                                               bool change_format = false);
  static void CopyAttrsQuantizedMatMulWithBiasAndDequantize(
      const Node* orig_node, NodeBuilder* nb, bool change_format = false);
  static void CopyAttrsPooling(const Node* orig_node, NodeBuilder* nb,
                               bool change_format = false);

  // Generate a graph node in graph 'g' representing a dummy Mkl tensor node,
  // using node for original node 'orig_node' and return it in '*out'.
  // TODO(nhasabni) We should move this to mkl_util.h
  void GetDummyMklTensorNode(std::unique_ptr<Graph>* g, Node** out,
                             const Node* orig_node);
  void GetDummyWorkspaceTensorNode(std::unique_ptr<Graph>* g, Node** out,
                                   const Node* orig_node);
};

MklLayoutRewritePass::ConstStringsInfo MklLayoutRewritePass::csinfo_;

// We register Mkl rewrite pass for phase 1 in post partitioning group.
// We register it here so that we get a complete picture of all users of Mkl
// nodes. Do not change the ordering of the Mkl passes.
const OptimizationPassRegistry::Grouping kMklLayoutRewritePassGroup =
    OptimizationPassRegistry::POST_PARTITIONING;
#ifdef ENABLE_MKL
REGISTER_OPTIMIZATION(kMklLayoutRewritePassGroup, 1, MklLayoutRewritePass);
#endif  // ENABLE_MKL

//////////////////////////////////////////////////////////////////////////
//           Helper functions for creating new node
//////////////////////////////////////////////////////////////////////////

static void FillInputs(const Node* n,
                       gtl::InlinedVector<Node*, 4>* control_edges,
                       gtl::InlinedVector<std::pair<Node*, int>, 4>* in) {
  control_edges->clear();
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      control_edges->push_back(e->src());
    } else {
      (*in)[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  std::sort(control_edges->begin(), control_edges->end());
}

void MklLayoutRewritePass::GetNodesProducingTFTensorList(
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs, int* input_idx,
    int list_length, std::vector<NodeBuilder::NodeOut>* output_nodes) {
  CHECK_LT(*input_idx, inputs.size());
  CHECK_GT(list_length, 0);
  CHECK_NOTNULL(output_nodes);
  output_nodes->reserve(list_length);

  while (list_length != 0) {
    CHECK_GT(list_length, 0);
    CHECK_LT(*input_idx, inputs.size());
    Node* n = inputs[*input_idx].first;
    int slot = inputs[*input_idx].second;
    // If input node 'n' is just producing a single tensor at
    // output slot 'slot' then we just add that single node.
    output_nodes->push_back(NodeBuilder::NodeOut(n, slot));
    (*input_idx)++;
    list_length--;
  }
}

// TODO(nhasabni) We should move this to mkl_util.h.
void MklLayoutRewritePass::GetDummyMklTensorNode(std::unique_ptr<Graph>* g,
                                                 Node** out,
                                                 const Node* orig_node) {
  // We use a tensor of shape {8} and value 0,0,0,0,0,0,0,0 to represent
  // dummy Mkl tensor. 8 = 2*size_t.
  const DataType dt = DataTypeToEnum<uint8>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  uint8 zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  proto.set_tensor_content(string(reinterpret_cast<char*>(&zero), 8));
  TensorShape dummy_shape({8});
  dummy_shape.AsProto(proto.mutable_tensor_shape());
  TF_CHECK_OK(NodeBuilder((*g)->NewName("DMT"), "Const")
                  .Attr("value", proto)
                  .Attr("dtype", dt)
                  .Device(orig_node->def().device())  // We place this node on
                                                      // the same device as the
                                                      // device of the original
                                                      // node.
                  .Finalize(&**g, out));
  CHECK_NOTNULL(*out);  // Make sure we got a valid object before using it

  // If number of inputs to the original node is > 0, then we add
  // control dependency between 1st input (index 0) of the original node and
  // the dummy Mkl node. This is needed because control-flow ops such as Enter,
  // Merge, etc, require frame_name of the dummy Mkl node to be same as the
  // rewritten node. Adding control edge between 1st input of the original node
  // and the dummy Mkl node ensures that the dummy node is in the same frame
  // as the original node. Choosing 1st input is not necessary - any input of
  // the original node is fine because all the inputs of a node are always in
  // the same frame.
  if (orig_node->num_inputs() > 0) {
    Node* orig_input0 = nullptr;
    TF_CHECK_OK(
        orig_node->input_node(0, const_cast<const Node**>(&orig_input0)));
    auto edge = (*g)->AddControlEdge(orig_input0, *out, false);
    DCHECK(edge != nullptr || DoesControlEdgeExist(orig_input0, *out));
  }

  (*out)->set_assigned_device_name(orig_node->assigned_device_name());
}

void MklLayoutRewritePass::GetNodesProducingMklTensorList(
    std::unique_ptr<Graph>* g, const Node* orig_node,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs, int* input_idx,
    int list_length, std::vector<NodeBuilder::NodeOut>* output_nodes) {
  CHECK_LT(*input_idx, inputs.size());
  CHECK_GT(list_length, 0);
  CHECK_NOTNULL(output_nodes);
  output_nodes->reserve(list_length);

  while (list_length != 0) {
    CHECK_GT(list_length, 0);
    CHECK_LT(*input_idx, inputs.size());
    Node* n = inputs[*input_idx].first;
    int slot = inputs[*input_idx].second;
    // If 'n' is producing a single tensor, then create a single Mkl tensor
    // node.
    Node* mkl_node = nullptr;
    int mkl_node_output_slot = 0;
    GetNodeProducingMklTensor(g, orig_node, n, slot, &mkl_node,
                              &mkl_node_output_slot);
    output_nodes->push_back(
        NodeBuilder::NodeOut(mkl_node, mkl_node_output_slot));
    (*input_idx)++;
    list_length--;
  }
}

// Get an input node that will feed Mkl tensor to the new
// node that we are constructing. An input node could be (1) 'n'
// if it is Mkl layer, or (2) a dummy node producing dummy Mkl tensor
// if 'n' is not an Mkl layer.
void MklLayoutRewritePass::GetNodeProducingMklTensor(
    std::unique_ptr<Graph>* g, const Node* orig_node, Node* n,
    int n_output_slot, Node** mkl_node, int* mkl_node_output_slot) {
  CHECK_NOTNULL(n);
  CHECK_NOTNULL(mkl_node);
  CHECK_NOTNULL(mkl_node_output_slot);

  // If this is an MKL op, then it will create extra output for MKL layout.
  DataType T;
  if (TryGetNodeAttr(n->def(), "T", &T) &&
      mkl_op_registry::IsMklLayoutDependentOp(n->type_string(), T)) {
    // If this is an MKL op, then it will generate an edge that will receive
    // Mkl tensor from a node.
    // output slot number for Mkl tensor would be N+slot number of TensorFlow
    // tensor, where N is total number of TensorFlow tensors.
    *mkl_node = n;
    *mkl_node_output_slot =
        GetTensorMetaDataIndex(n_output_slot, n->num_outputs());
  } else {
    // If we have not visited the node and rewritten it, then we need
    // to create a dummy node that will feed a dummy Mkl tensor to this node.
    // DummyMklTensor node has no input and generates only 1 output
    // (dummy Mkl tensor) as output slot number 0.
    GetDummyMklTensorNode(g, mkl_node, orig_node);
    CHECK_NOTNULL(*mkl_node);
    *mkl_node_output_slot = 0;
  }
}

int MklLayoutRewritePass::SetUpContiguousInputs(
    std::unique_ptr<Graph>* g,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
    NodeBuilder* nb, const Node* old_node,
    std::vector<NodeBuilder::NodeOut>* workspace_tensors,
    bool are_workspace_tensors_available) {
  CHECK_NOTNULL(workspace_tensors);
  CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);

  // TODO(nhasabni): Temporary solution to connect filter input of
  // BackpropInput with the converted filter from Conv2D.
  bool do_connect_conv2d_backprop_input_filter = false;
  Node* conv2d_node = nullptr;
  // Filter node is 2nd input (slot index 1) of Conv2D.
  int kConv2DFilterInputSlotIdx = 1;
  int kConv2DBackpropInputFilterInputSlotIdx = 1;
  int kConv2DFilterOutputSlotIdx = 1;
  if (old_node->type_string() == csinfo_.conv2d_grad_input) {
    // We need to find Conv2D node from Conv2DBackpropInput.
    // For that let's first find filter node that is 2nd input (slot 1)
    // of BackpropInput.
    Node* filter_node = nullptr;
    TF_CHECK_OK(old_node->input_node(kConv2DBackpropInputFilterInputSlotIdx,
                                     &filter_node));
    CHECK_NOTNULL(filter_node);

    // Now check which nodes receive from filter_node. Filter feeds as
    // 2nd input (slot 1) of _MklConv2D, _MklConv2DWithBias, and
    // _MklFusedConv2D.
    for (const Edge* e : filter_node->out_edges()) {
      if ((e->dst()->type_string() == csinfo_.mkl_conv2d ||
           e->dst()->type_string() == csinfo_.mkl_pad_with_conv2d ||
           e->dst()->type_string() == csinfo_.mkl_pad_with_fused_conv2d ||
           e->dst()->type_string() == csinfo_.mkl_conv2d_with_bias ||
           e->dst()->type_string() == csinfo_.mkl_fused_conv2d) &&
          e->dst_input() == kConv2DFilterInputSlotIdx
          /* filter is 2nd input of Conv2D and _MklConv2D. */) {
        if (conv2d_node != nullptr) {
          VLOG(1) << "MklLayoutRewritePass: unusual case of same filter"
                  << " feeding multiple Conv2D nodes: "
                  << filter_node->DebugString();
          // We will not connect filter input of Conv2DBackpropInput
          // to be safe here.
          do_connect_conv2d_backprop_input_filter = false;
          break;
        } else {
          conv2d_node = e->dst();
          do_connect_conv2d_backprop_input_filter = true;
        }
      }
    }
  }

  // Number of input slots to original op
  // Input slots are represented by .Input() calls in REGISTER_OP.
  int old_node_input_slots = old_node->op_def().input_arg_size();
  // Actual number of inputs can be greater than or equal to number
  // of Input slots because inputs of type list could be unfolded.
  CHECK_GE(old_node_inputs.size(), old_node_input_slots);
  int nn_slot_idx = 0;  // slot index for inputs of new node

  // Let's copy all inputs (TF tensors) of original node to new node.
  int iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    CHECK_LT(iidx, old_node_inputs.size());
    const OpDef::ArgDef& arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int N = GetTensorListLength(arg, old_node);
      GetNodesProducingTFTensorList(old_node_inputs, &iidx, N,
                                    &new_node_inputs);
      nb->Input(new_node_inputs);
      nn_slot_idx++;
    } else {
      // Special case for connecting filter input of Conv2DBackpropInput
      if (do_connect_conv2d_backprop_input_filter &&
          iidx == kConv2DBackpropInputFilterInputSlotIdx) {
        nb->Input(conv2d_node, kConv2DFilterOutputSlotIdx);
      } else {
        nb->Input(old_node_inputs[iidx].first, old_node_inputs[iidx].second);
      }
      iidx++;
      nn_slot_idx++;
    }
  }

  // If workspace tensors are available for this op and we are using
  // contiguous ordering then we need to add Tensorflow tensor for
  // workspace here because Tensorflow tensor for workspace is the
  // last tensor in the list of Tensorflow tensors.
  if (are_workspace_tensors_available) {
    CHECK_EQ(workspace_tensors->size(), 2);
    // Tensorflow tensor
    nb->Input((*workspace_tensors)[0].node, (*workspace_tensors)[0].index);
    nn_slot_idx++;
  }

  // Let's now setup all Mkl inputs to a new node.
  // Number of Mkl inputs must be same as number of TF inputs.
  iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    CHECK_LT(iidx, old_node_inputs.size());
    const OpDef::ArgDef& arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int N = GetTensorListLength(arg, old_node);
      GetNodesProducingMklTensorList(g, old_node, old_node_inputs, &iidx, N,
                                     &new_node_inputs);
      nb->Input(new_node_inputs);
      nn_slot_idx++;
    } else {
      Node* mkl_node = nullptr;
      int mkl_node_output_slot = 0;
      // Special case for connecting filter input of Conv2DBackpropInput
      if (do_connect_conv2d_backprop_input_filter &&
          iidx == kConv2DBackpropInputFilterInputSlotIdx) {
        GetNodeProducingMklTensor(g, old_node, conv2d_node,
                                  kConv2DFilterOutputSlotIdx, &mkl_node,
                                  &mkl_node_output_slot);
      } else {
        GetNodeProducingMklTensor(g, old_node, old_node_inputs[iidx].first,
                                  old_node_inputs[iidx].second, &mkl_node,
                                  &mkl_node_output_slot);
      }
      nb->Input(mkl_node, mkl_node_output_slot);
      iidx++;
      nn_slot_idx++;
    }
  }

  // If workspace tensors are available for this op and we are using
  // contiguous ordering then we need to add Mkl tensor for
  // workspace here because Mkl tensor for workspace is the
  // last tensor in the list of Mkl tensors.
  if (are_workspace_tensors_available) {
    CHECK_EQ(workspace_tensors->size(), 2);
    // Mkl tensor
    nb->Input((*workspace_tensors)[1].node, (*workspace_tensors)[1].index);
    nn_slot_idx++;
  }

  return nn_slot_idx;
}

Status MklLayoutRewritePass::SetUpInputs(
    std::unique_ptr<Graph>* g,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
    NodeBuilder* nb, const Node* old_node) {
  // Let's check if we need to add workspace tensors for this node.
  // We add workspace edge only for MaxPool, LRN and BatchNorm.
  std::vector<NodeBuilder::NodeOut> workspace_tensors;
  bool are_workspace_tensors_available = false;

  // Avoid workspace check for QuantizedConv2D and the fused
  // Ops as they don't have attribute: "T".
  std::vector<string> quant_ops{
      "Dequantize",
      "QuantizeV2",
      "QuantizedConv2D",
      "QuantizedConv2DWithBias",
      "QuantizedConv2DAndRelu",
      "QuantizedConv2DWithBiasAndRelu",
      "QuantizedConv2DWithBiasSumAndRelu",
      "QuantizedConv2DPerChannel",
      "QuantizedConv2DAndRequantize",
      "QuantizedConv2DWithBiasAndRequantize",
      "QuantizedConv2DAndReluAndRequantize",
      "QuantizedConv2DWithBiasAndReluAndRequantize",
      "QuantizedConv2DWithBiasSumAndReluAndRequantize",
      "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
      "QuantizedMatMulWithBias",
      "QuantizedMatMulWithBiasAndRequantize",
      "QuantizedMatMulWithBiasAndDequantize",
      "QuantizedMatMulWithBiasAndRelu",
      "QuantizedMatMulWithBiasAndReluAndRequantize",
      "QuantizedDepthwiseConv2D",
      "QuantizedDepthwiseConv2DWithBias",
      "QuantizedDepthwiseConv2DWithBiasAndRelu",
      "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize"};
  bool should_check_workspace =
      std::find(std::begin(quant_ops), std::end(quant_ops),
                old_node->type_string()) == std::end(quant_ops);
  if (should_check_workspace)
    AddWorkSpaceEdgeIfNeeded(g, old_node, nb, &workspace_tensors,
                             &are_workspace_tensors_available);

  int new_node_input_slots = 0;
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    // TODO(nhasabni): implement this function just for same of completion.
    // We do not use interleaved ordering right now.
    return Status(
        error::Code::UNIMPLEMENTED,
        "Interleaved ordering of tensors is currently not supported.");
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    new_node_input_slots = SetUpContiguousInputs(
        g, old_node_inputs, nb, old_node, &workspace_tensors,
        are_workspace_tensors_available);
  }

  // Sanity check
  int old_node_input_slots = old_node->op_def().input_arg_size();
  if (!are_workspace_tensors_available) {
    // If we are not adding workspace tensors for this op, then the total
    // number of input slots to the new node _must_ be 2 times the number
    // of input slots to the original node: N original Tensorflow tensors and
    // N for Mkl tensors corresponding to each Tensorflow tensors.
    CHECK_EQ(new_node_input_slots, old_node_input_slots * 2);
  } else {
    // If we are adding workspace tensors for this op, then the total
    // The total number of input slots to new node _must_ be 2 times the number
    // of input slots to the original node: N original Tensorflow tensors and
    // N for Mkl tensors corresponding to each Tensorflow tensors plus 2
    // (for workspace Tensorflow tensor and workspace Mkl tensor).
    CHECK_EQ(new_node_input_slots, old_node_input_slots * 2 + 2);
  }

  return Status::OK();
}

Status MklLayoutRewritePass::CopyInputs(
    const Node* old_node,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
    NodeBuilder* nb) {
  // Number of input slots to old node
  // Input slots are represented by .Input() calls in REGISTER_OP.
  int old_node_input_slots = old_node->op_def().input_arg_size();
  // Actual number of inputs can be greater than or equal to number
  // of Input slots because inputs of type list could be unfolded.
  auto old_node_input_size = old_node_inputs.size();
  DCHECK_GE(old_node_input_size, old_node_input_slots);

  // Let's copy all inputs of old node to new node.
  int iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    DCHECK_LT(iidx, old_node_input_size);
    const OpDef::ArgDef& arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int N = GetTensorListLength(arg, old_node);
      GetNodesProducingTFTensorList(old_node_inputs, &iidx, N,
                                    &new_node_inputs);
      nb->Input(new_node_inputs);
    } else {
      nb->Input(old_node_inputs[iidx].first, old_node_inputs[iidx].second);
      iidx++;
    }
  }
  return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions related to workspace pass
//////////////////////////////////////////////////////////////////////////

// TODO(nhasabni) We should move this to mkl_util.h.
void MklLayoutRewritePass::GetDummyWorkspaceTensorNode(
    std::unique_ptr<Graph>* g, Node** out, const Node* orig_node) {
  // We use uint8 tensor of shape 8 with content {0,0,0,0,0,0,0,0} to represent
  // workspace tensor.
  GetDummyMklTensorNode(g, out, orig_node);
}

void MklLayoutRewritePass::AddWorkSpaceEdgeIfNeeded(
    std::unique_ptr<Graph>* g, const Node* orig_node, NodeBuilder* nb,
    std::vector<NodeBuilder::NodeOut>* ws_tensors, bool* are_ws_tensors_added) {
  bool workspace_edge_added = false;  // Default initializer
  CHECK_NOTNULL(are_ws_tensors_added);
  *are_ws_tensors_added = false;  // Default initializer

  DataType T;
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  for (auto ws : wsinfo_) {
    if (orig_node->type_string() == ws.fwd_op &&
        mkl_op_registry::IsMklLayoutDependentOp(
            mkl_op_registry::GetMklOpName(orig_node->type_string()), T)) {
      // If this op is a fwd op, then we need to check if there is an
      // edge from this node's fwd_slot to bwdop's bwd_slot. If there is
      // an edge, then we just add an attribute on this node for setting
      // workspace_passed to true. We don't add actual workspace edge
      // in this node. Actual workspace edge gets added in the backward
      // op for this node.
      for (const Edge* e : orig_node->out_edges()) {
        if (e->src_output() == ws.fwd_slot &&
            e->dst()->type_string() == ws.bwd_op &&
            e->dst_input() == ws.bwd_slot) {
          nb->Attr("workspace_enabled", true);
          VLOG(1) << "MklLayoutRewritePass: workspace_enabled for "
                  << orig_node->type_string();
          workspace_edge_added = true;
          // We found the edge that we were looking for, so break.
          break;
        }
      }

      if (!workspace_edge_added) {
        // If we are here, then we did not find backward operator for this
        // node.
        nb->Attr("workspace_enabled", false);
      }
    } else if (orig_node->type_string() == ws.bwd_op &&
               mkl_op_registry::IsMklLayoutDependentOp(
                   mkl_op_registry::GetMklOpName(orig_node->type_string()),
                   T)) {
      // If this op is a bwd op, then we need to add workspace edge and
      // it's Mkl tensor edge between its corresponding fwd op and this
      // op. Corresponding fwd op is specified in 'fwd_op' field of
      // workspace info. fwd_slot and bwd_slot in workspace info specify
      // an edge between which slots connect forward and backward op.
      // Once all these criteria match, we add a workspace edge between
      // ws_fwd_slot and ws_bwd_slot. Its corresponding Mkl tensor is
      // determined by interleaved/contiguous ordering. Function
      // DataIndexToMetaDataIndex tells us the location of Mkl tensor
      // from the location of the Tensorflow tensor.
      for (const Edge* e : orig_node->in_edges()) {
        if (e->src_output() == ws.fwd_slot &&
            // We would have rewritten the forward op, so we need to use
            // GetMklOpName call to get its Mkl name.
            e->src()->type_string() ==
                mkl_op_registry::GetMklOpName(ws.fwd_op) &&
            e->dst_input() == ws.bwd_slot) {
          nb->Attr("workspace_enabled", true);
          CHECK_NOTNULL(ws_tensors);
          // Add workspace edge between fwd op and bwd op.
          ws_tensors->push_back(NodeBuilder::NodeOut(e->src(), ws.ws_fwd_slot));
          // Add Mkl tensor edge for workspace edge between fwd op and bwd op.
          ws_tensors->push_back(NodeBuilder::NodeOut(
              e->src(), DataIndexToMetaDataIndex(ws.ws_fwd_slot,
                                                 e->src()->num_outputs())));
          *are_ws_tensors_added = true;
          // In terms of input ordering, we add these calls to add Input
          // here because workspace edge (and its Mkl tensor) is the last
          // edge in the fwdop and bwdop. So all inputs before workspace
          // tensor have been added by SetUpInputs function.
          VLOG(1) << "MklLayoutRewritePass: workspace_enabled for "
                  << orig_node->type_string();
          workspace_edge_added = true;
          // We found the edge that we were looking for, so break.
          break;
        }
      }

      // If we are here means we did not find fwd op that feeds to this
      // bwd op. So in this case, we need to generate dummy tensors for
      // workspace input and Mkl tensor for workspace, and set
      // workspace_enabled to false.
      if (!workspace_edge_added) {
        nb->Attr("workspace_enabled", false);
        Node* dmt_ws = nullptr;      // Dummy tensor for workspace
        Node* dmt_mkl_ws = nullptr;  // Dummy Mkl tensor for workspace
        GetDummyWorkspaceTensorNode(g, &dmt_ws, orig_node);
        GetDummyMklTensorNode(g, &dmt_mkl_ws, orig_node);
        CHECK_NOTNULL(dmt_ws);
        CHECK_NOTNULL(dmt_mkl_ws);
        CHECK_NOTNULL(ws_tensors);
        // We add dummy tensor as workspace tensor.
        ws_tensors->push_back(NodeBuilder::NodeOut(dmt_ws, 0));
        // We add dummy tensor as Mkl tensor for workspace tensor.
        ws_tensors->push_back(NodeBuilder::NodeOut(dmt_mkl_ws, 0));
        *are_ws_tensors_added = true;
        VLOG(1) << "MklLayoutRewritePass: dummy workspace_enabled for "
                << orig_node->type_string();
      }
    } else {
      // If this node does not match any workspace info, then we do not
      // do anything special for workspace propagation for it.
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

// Generic function to copy all attributes from original node to target.
void MklLayoutRewritePass::CopyAttrsAll(const Node* orig_node, NodeBuilder* nb,
                                        bool change_format) {
  string name;
  AttrSlice attr_list(orig_node->def());

  auto iter = attr_list.begin();
  while (iter != attr_list.end()) {
    name = iter->first;
    auto attr = iter->second;
    nb->Attr(name, attr);
    ++iter;
  }
}

// Generic function to copy all attributes and check if filter is const.
void MklLayoutRewritePass::CopyAttrsAllCheckConstFilter(const Node* orig_node,
                                                        NodeBuilder* nb,
                                                        bool change_format) {
  CopyAttrsAll(orig_node, nb, change_format);

  // Check and set filter attribute.
  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));
  nb->Attr("is_filter_const", filter_node->IsConstant());
}

void MklLayoutRewritePass::CopyAttrsConvCheckConstFilter(const Node* orig_node,
                                                         NodeBuilder* nb,
                                                         bool change_format) {
  DataType T;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));

  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("padding", padding);
  nb->Attr("is_filter_const", filter_node->IsConstant());

  // Add attributes related to `data_format`.
  CopyFormatAttrsConv(orig_node, nb, strides, dilations, change_format);
}

void MklLayoutRewritePass::CopyAttrsConv(const Node* orig_node, NodeBuilder* nb,
                                         bool change_format) {
  DataType T;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("padding", padding);

  // Add attributes related to `data_format`.
  CopyFormatAttrsConv(orig_node, nb, strides, dilations, change_format);
}

// Used in rinfo when replacing __MklDummyPadWithConv2D by _MklPadWithConv2D
void MklLayoutRewritePass::CopyAttrsPadWithConv2D(const Node* orig_node,
                                                  NodeBuilder* nb,
                                                  bool change_format) {
  DataType Tpaddings;
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  bool use_cudnn_on_gpu;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(
      GetNodeAttr(orig_node->def(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tpaddings", &Tpaddings));

  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("dilations", dilations);
  nb->Attr("padding", padding);
  nb->Attr("is_filter_const", filter_node->IsConstant());
  nb->Attr("data_format", data_format);
  nb->Attr("use_cudnn_on_gpu", use_cudnn_on_gpu);
  nb->Attr("Tpaddings", Tpaddings);
}

void MklLayoutRewritePass::CopyAttrsPadWithFusedConv2D(const Node* orig_node,
                                                       NodeBuilder* nb,
                                                       bool change_format) {
  DataType Tpaddings;

  CopyAttrsFusedConv2D(orig_node, nb, change_format);

  // Get attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tpaddings", &Tpaddings));
  // Check if filter is a constant.
  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("Tpaddings", Tpaddings);
  nb->Attr("is_filter_const", filter_node->IsConstant());
}

// Used with MergePadWithConv2D
void MklLayoutRewritePass::CopyAttrsFromPadAndConv2D(const Node* orig_node1,
                                                     const Node* orig_node2,
                                                     NodeBuilder* nb,
                                                     bool change_format) {
  DataType Tpaddings;
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  bool use_cudnn_on_gpu;

  // Get all attributes from old node 1.
  TF_CHECK_OK(GetNodeAttr(orig_node1->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node1->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node1->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node1->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node1->def(), "data_format", &data_format));
  TF_CHECK_OK(
      GetNodeAttr(orig_node1->def(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  // Get all attributes from old node 2.
  TF_CHECK_OK(GetNodeAttr(orig_node2->def(), "Tpaddings", &Tpaddings));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("dilations", dilations);
  nb->Attr("padding", padding);
  nb->Attr("data_format", data_format);
  nb->Attr("use_cudnn_on_gpu", use_cudnn_on_gpu);
  nb->Attr("Tpaddings", Tpaddings);
}

void MklLayoutRewritePass::CopyAttrsFromPadAndFusedConv2D(
    const Node* fused_conv2d, const Node* pad, NodeBuilder* nb,
    bool change_format) {
  DataType T;
  int num_args;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  float epsilon;
  std::vector<string> fused_ops;
  DataType Tpaddings;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "num_args", &num_args));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "fused_ops", &fused_ops));
  TF_CHECK_OK(GetNodeAttr(fused_conv2d->def(), "epsilon", &epsilon));
  TF_CHECK_OK(GetNodeAttr(pad->def(), "Tpaddings", &Tpaddings));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("num_args", num_args);
  nb->Attr("strides", strides);
  nb->Attr("padding", padding);
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);
  nb->Attr("epsilon", epsilon);
  nb->Attr("Tpaddings", Tpaddings);
  nb->Attr("fused_ops", fused_ops);
}

void MklLayoutRewritePass::CopyAttrsConv2DDepthwiseCheckConstFilter(
    const Node* orig_node, NodeBuilder* nb, bool change_format) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));

  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("dilations", dilations);
  nb->Attr("padding", padding);
  nb->Attr("is_filter_const", filter_node->IsConstant());
  nb->Attr("data_format", data_format);
}

void MklLayoutRewritePass::CopyAttrsQuantizedConv2D(const Node* orig_node,
                                                    NodeBuilder* nb,
                                                    bool change_format) {
  DataType Tinput, Tfilter, out_type;
  string padding;
  string data_format("NHWC");
  std::vector<int32> strides, dilations, padding_list;
  bool has_padding_list = HasNodeAttr(orig_node->def(), "padding_list");

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tinput", &Tinput));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tfilter", &Tfilter));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "out_type", &out_type));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  if (has_padding_list) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding_list", &padding_list));
  }

  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("Tinput", Tinput);
  nb->Attr("Tfilter", Tfilter);
  nb->Attr("out_type", out_type);
  nb->Attr("padding", padding);
  nb->Attr("is_filter_const", filter_node->IsConstant());
  nb->Attr("strides", strides);
  nb->Attr("dilations", dilations);
  nb->Attr("T", out_type);  // added "T" for facilitating MklToTf conversion.
  nb->Attr("data_format", data_format);
  if (has_padding_list) {
    nb->Attr("padding_list", padding_list);
  }

  // Requantization attr Tbias.
  DataType Tbias;
  Status bias_status = GetNodeAttr(orig_node->def(), "Tbias", &Tbias);
  if (bias_status.ToString() == "OK") nb->Attr("Tbias", Tbias);
}

void MklLayoutRewritePass::CopyAttrsQuantizedMatMulWithBiasAndDequantize(
    const Node* orig_node, NodeBuilder* nb, bool change_format) {
  DataType T1, T2, Toutput;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T1", &T1));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T2", &T2));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Toutput", &Toutput));

  // Add attributes to new node.
  nb->Attr("T1", T1);
  nb->Attr("T2", T2);
  nb->Attr("Toutput", Toutput);
  nb->Attr("T", T1);  // added "T" for facilitating MklToTf conversion.

  // Requantization attr Tbias
  DataType Tbias;
  Status bias_status = GetNodeAttr(orig_node->def(), "Tbias", &Tbias);
  if (bias_status.ToString() == "OK") nb->Attr("Tbias", Tbias);
}

void MklLayoutRewritePass::CopyAttrsQuantizedMatMulWithBias(
    const Node* orig_node, NodeBuilder* nb, bool change_format) {
  DataType T1, T2, Toutput;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T1", &T1));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T2", &T2));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Toutput", &Toutput));

  Node* weight_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &weight_node));

  // Add attributes to new node.
  nb->Attr("T1", T1);
  nb->Attr("T2", T2);
  nb->Attr("Toutput", Toutput);
  nb->Attr("is_weight_const", weight_node->IsConstant());
  nb->Attr("T", Toutput);  // added "T" for facilitating MklToTf conversion.

  // Requantization attr Tbias
  DataType Tbias;
  Status bias_status = GetNodeAttr(orig_node->def(), "Tbias", &Tbias);
  if (bias_status.ToString() == "OK") nb->Attr("Tbias", Tbias);
}

void MklLayoutRewritePass::CopyFormatAttrsConv(
    const Node* orig_node, NodeBuilder* nb, const std::vector<int32>& strides,
    const std::vector<int32>& dilations, bool change_format) {
  string data_format;

  if (!change_format) {
    nb->Attr("strides", strides);
    nb->Attr("dilations", dilations);

    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
    nb->Attr("data_format", data_format);
  } else {
    std::vector<int32> new_strides;
    std::vector<int32> new_dilations;
    if (strides.size() == 5) {
      // `strides` and `dilations` also need to be changed according to
      // `data_format`. In this case, from `NDHWC` to `NCDHW`.
      new_strides = {strides[NDHWC::dim::N], strides[NDHWC::dim::C],
                     strides[NDHWC::dim::D], strides[NDHWC::dim::H],
                     strides[NDHWC::dim::W]};

      new_dilations = {dilations[NDHWC::dim::N], dilations[NDHWC::dim::C],
                       dilations[NDHWC::dim::D], dilations[NDHWC::dim::H],
                       dilations[NDHWC::dim::W]};
    } else {
      // `strides` and `dilations` also need to be changed according to
      // `data_format`. In this case, from `NHWC` to `NCHW`.

      new_strides = {strides[NHWC::dim::N], strides[NHWC::dim::C],
                     strides[NHWC::dim::H], strides[NHWC::dim::W]};

      new_dilations = {dilations[NHWC::dim::N], dilations[NHWC::dim::C],
                       dilations[NHWC::dim::H], dilations[NHWC::dim::W]};
    }
    nb->Attr("strides", new_strides);
    nb->Attr("dilations", new_dilations);
  }
}

void MklLayoutRewritePass::CopyAttrsFusedConv2D(const Node* orig_node,
                                                NodeBuilder* nb,
                                                bool change_format) {
  DataType T;
  int num_args;
  float epsilon;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<string> fused_ops;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "num_args", &num_args));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "fused_ops", &fused_ops));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "epsilon", &epsilon));

  Node* filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("num_args", num_args);
  nb->Attr("strides", strides);
  nb->Attr("padding", padding);
  nb->Attr("is_filter_const", filter_node->IsConstant());
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);
  nb->Attr("fused_ops", fused_ops);
  nb->Attr("epsilon", epsilon);
}

void MklLayoutRewritePass::CopyAttrsPooling(const Node* orig_node,
                                            NodeBuilder* nb,
                                            bool change_format) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> ksize, strides;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "ksize", &ksize));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("padding", padding);

  if (!change_format) {
    nb->Attr("strides", strides);
    nb->Attr("ksize", ksize);

    nb->Attr("data_format", data_format);
  } else {
    std::vector<int32> new_strides;
    std::vector<int32> new_ksize;
    if (strides.size() == 5) {
      DCHECK(data_format == "NCDHW");
      // `strides` and `ksize` also need to be changed according to
      // `data_format`. In this case, from `NDHWC` to `NCDHW`.
      new_strides = {strides[NDHWC::dim::N], strides[NDHWC::dim::C],
                     strides[NDHWC::dim::D], strides[NDHWC::dim::H],
                     strides[NDHWC::dim::W]};

      new_ksize = {ksize[NDHWC::dim::N], ksize[NDHWC::dim::C],
                   ksize[NDHWC::dim::D], ksize[NDHWC::dim::H],
                   ksize[NDHWC::dim::W]};

    } else {
      // `strides` and `ksize` also need to be changed according to
      // `data_format`. In this case, from `NHWC` to `NCHW`.
      DCHECK(data_format == "NCHW");
      new_strides = {strides[NHWC::dim::N], strides[NHWC::dim::C],
                     strides[NHWC::dim::H], strides[NHWC::dim::W]};

      new_ksize = {ksize[NHWC::dim::N], ksize[NHWC::dim::C],
                   ksize[NHWC::dim::H], ksize[NHWC::dim::W]};
    }
    nb->Attr("strides", new_strides);
    nb->Attr("ksize", new_ksize);
  }
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions related to node merge pass
//////////////////////////////////////////////////////////////////////////

Node* MklLayoutRewritePass::CheckForNodeMerge(const Node* a) const {
  // TODO(nhasabni) Add check for type of node similar to CheckForNodeRewrite
  // once we support BiasAddGrad as Mkl layer.

  // Search for all matching mergeinfo.
  // We allow more than one match for extensibility.
  std::vector<const MergeInfo*> matching_mi;
  for (auto mi = minfo_.cbegin(); mi != minfo_.cend(); ++mi) {
    if (a->type_string() == mi->op1 || a->type_string() == mi->op2) {
      matching_mi.push_back(&*mi);
    }
  }

  for (const MergeInfo* mi : matching_mi) {
    // Get the operand with which 'a' can be merged.
    Node* b = nullptr;
    if ((b = mi->get_node_to_be_merged(a)) == nullptr) {
      continue;
    }

    // Get the control edges and input of node
    const int N_in = a->num_inputs();
    gtl::InlinedVector<Node*, 4> a_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> a_in(N_in);
    FillInputs(a, &a_control_edges, &a_in);

    const int B_in = b->num_inputs();
    gtl::InlinedVector<Node*, 4> b_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> b_in(B_in);
    FillInputs(b, &b_control_edges, &b_in);

    // Shouldn't merge if a and b have different control edges.
    if (a_control_edges != b_control_edges) {
      continue;
    } else {
      // We found a match.
      return b;
    }
  }

  return nullptr;
}

Status MklLayoutRewritePass::MergeConv2DWithBiasAdd(std::unique_ptr<Graph>* g,
                                                    Node* m, Node* n) {
  CHECK_EQ(((m->type_string() == csinfo_.bias_add &&
             n->type_string() == csinfo_.conv2d)) ||
               ((n->type_string() == csinfo_.bias_add &&
                 m->type_string() == csinfo_.conv2d)),
           true);

  // If 'm' is BiasAdd, then 'n' is Conv2D. Since Conv2D feeds BiasAdd,
  // BiasAdd is successor node, and Conv2D predecessor node.
  Node* pred = m->type_string() == csinfo_.bias_add ? n : m;
  Node* succ = m->type_string() == csinfo_.bias_add ? m : n;

  // 1. Get all attributes from input nodes.
  DataType T_pred, T_succ;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  string data_format_pred, data_format_succ;
  bool use_cudnn_on_gpu;
  TF_CHECK_OK(GetNodeAttr(pred->def(), "T", &T_pred));
  TF_CHECK_OK(GetNodeAttr(succ->def(), "T", &T_succ));
  TF_CHECK_OK(GetNodeAttr(pred->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(pred->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(pred->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(pred->def(), "data_format", &data_format_pred));
  TF_CHECK_OK(GetNodeAttr(succ->def(), "data_format", &data_format_succ));
  TF_CHECK_OK(GetNodeAttr(pred->def(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  // We check to ensure that data formats of both succ and pred are same.
  // We expect them to be same, so we can enforce this as assert.
  // But assert can be too strict, so we enforce this as a check.
  // If the check fails, then we do not merge two nodes.
  // We also do same check for devices.
  if (data_format_pred != data_format_succ || T_pred != T_succ ||
      pred->assigned_device_name() != succ->assigned_device_name() ||
      pred->def().device() != succ->def().device()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "data_format or T attribute or devices of Conv2D and "
                  "BiasAdd do not match. Will skip node merge optimization");
  }

  const int succ_num = succ->num_inputs();
  gtl::InlinedVector<Node*, 4> succ_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> succ_in(succ_num);
  FillInputs(succ, &succ_control_edges, &succ_in);

  const int pred_num = pred->num_inputs();
  gtl::InlinedVector<Node*, 4> pred_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> pred_in(pred_num);
  FillInputs(pred, &pred_control_edges, &pred_in);

  // We need to ensure that Conv2D only feeds to BiasAdd (some other operator is
  // not expecting output of Conv2D). If this is not the case, then we cannot
  // merge Conv2D with BiasAdd.
  const int kFirstOutputSlot = 0;
  for (const Edge* e : pred->out_edges()) {
    if (e->src_output() == kFirstOutputSlot && e->dst() != succ) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "Conv2D does not feed to BiasAdd, or "
                    "it feeds BiasAdd but has multiple outputs. "
                    "Will skip node merge optimization");
    }
  }

  // 2. Get inputs from both the nodes.
  // Find the 2 inputs from the conv and the bias from the add Bias.
  // Get operand 0, 1 of conv2D.
  CHECK_EQ(pred->in_edges().size(), 2);  // Conv2D must have 2 inputs.
  // Get operand 1 of add_bias
  // BiasAdd must have 2 inputs: Conv, bias
  CHECK_EQ(succ->in_edges().size(), 2);

  // We will use the node name of BiasAdd as the name of new node
  // Build new node. We use same name as original node, but change the op
  // name.
  NodeBuilder nb(succ->name(), csinfo_.conv2d_with_bias);
  nb.Input(pred_in[0].first, pred_in[0].second);  // In1 of Conv2D
  // pred_in[1] will be 2nd Tensorflow tensor for Conv2D.
  nb.Input(pred_in[1].first, pred_in[1].second);  // In2 of Conv2D
  // In1 of BiasAdd is same as output of Conv2D.
  nb.Input(succ_in[1].first, succ_in[1].second);  // In2 of BiasAdd

  // Copy attributes from Conv2D to Conv2DWithBias.
  CopyAttrsConvCheckConstFilter(const_cast<const Node*>(pred), &nb);

  // Copy the device assigned to old node to new node.
  nb.Device(succ->def().device());

  // Create node.
  Node* new_node;
  TF_CHECK_OK(nb.Finalize(&**g, &new_node));

  // In the following code of this function, an unsorted set is used to make
  // sure no duplicated edges be added into the new node. Therefore, we can
  // pass allow_duplicates = true in AddControlEdge call to skip the O(#edges)
  // check in the routine.

  // Incoming data edges from 'pred' node and 'succ' node to new 'new_node'
  // node are already copied in BuildNode. We handle control edges now.
  std::unordered_set<Node*> unique_node;
  for (const Edge* e : pred->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), new_node, true);
      }
    }
  }
  unique_node.clear();

  for (const Edge* e : succ->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), new_node, true);
      }
    }
  }
  unique_node.clear();

  // Incoming edges are fixed, we will fix the outgoing edges now.
  // First, we will fix outgoing control edges from 'pred' node.
  for (const Edge* e : pred->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(new_node, e->dst(), true);
      }
    }
  }
  unique_node.clear();

  // Second, we will fix outgoing control and data edges from 'succ' node.
  for (const Edge* e : succ->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(new_node, e->dst(), true);
      }
    } else {
      // BiasAdd has only 1 output (at slot 0) and merged node also has only 1
      // output (at slot 0).
      const int kConv2DWithBiasOutputSlot = 0;
      CHECK_NOTNULL((*g)->AddEdge(new_node, kConv2DWithBiasOutputSlot, e->dst(),
                                  e->dst_input()));
    }
  }

  // Copy device assigned to old node to new node.
  // It's ok to use pred or succ as we have enforced a check that
  // both have same device assigned.
  new_node->set_assigned_device_name(pred->assigned_device_name());

  VLOG(1) << "MklLayoutRewritePass: Merged old node:" << pred->DebugString()
          << ", and node: " << succ->DebugString()
          << ", into node:" << new_node->DebugString();

  (*g)->RemoveNode(succ);
  (*g)->RemoveNode(pred);

  return Status::OK();
}

Status MklLayoutRewritePass::MergePadWithConv2D(std::unique_ptr<Graph>* g,
                                                Node* m, Node* n) {
  DCHECK((m->type_string() == csinfo_.pad &&
          (n->type_string() == csinfo_.conv2d ||
           n->type_string() == csinfo_.fused_conv2d)) ||
         (n->type_string() == csinfo_.pad &&
          (m->type_string() == csinfo_.conv2d ||
           m->type_string() == csinfo_.fused_conv2d)));

  bool is_fused_conv2d = n->type_string() == csinfo_.fused_conv2d ||
                         m->type_string() == csinfo_.fused_conv2d;
  // Conv2D is successor node, and Pad predecessor node.
  Node* pred = m->type_string() == csinfo_.pad ? m : n;
  Node* succ = m->type_string() == csinfo_.pad ? n : m;

  // 1. Get all attributes from input nodes.
  DataType T_pred, T_succ;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  string data_format_pred, data_format_succ;

  TF_CHECK_OK(GetNodeAttr(pred->def(), "T", &T_pred));
  TF_CHECK_OK(GetNodeAttr(succ->def(), "T", &T_succ));
  TF_CHECK_OK(GetNodeAttr(succ->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(succ->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(succ->def(), "dilations", &dilations));
  // Check if the devices of both succ and pred are the same.
  // Assert is not used because it can be too strict.
  // Don't need to check for data formats because it is not available in Pad.
  if (T_pred != T_succ ||
      pred->assigned_device_name() != succ->assigned_device_name() ||
      pred->def().device() != succ->def().device()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "T attribute or devices of Conv2D and "
                  "Pad do not match. Will skip node merge optimization");
  }

  const int succ_num = succ->num_inputs();
  gtl::InlinedVector<Node*, 4> succ_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> succ_in(succ_num);
  FillInputs(succ, &succ_control_edges, &succ_in);

  const int pred_num = pred->num_inputs();
  gtl::InlinedVector<Node*, 4> pred_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> pred_in(pred_num);
  FillInputs(pred, &pred_control_edges, &pred_in);

  // We need to ensure that Pad only feeds to Conv2D (some other operator is
  // not expecting output of Pad). If this is not the case, then we cannot
  // merge Conv2D with Pad.
  const int kFirstOutputSlot = 0;
  for (const Edge* e : pred->out_edges()) {
    if (e->src_output() == kFirstOutputSlot && e->dst() != succ) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "Pad does not feed to Conv2D, or "
                    "it feeds Conv2D but has multiple outputs. "
                    "Will skip node merge optimization");
    }
  }

  // 2. Get inputs from both the nodes.

  // Pad must have 2 data inputs: "input" and paddings.
  int PadDataInputEdges = 0;
  for (const Edge* e : pred->in_edges()) {
    if (!e->IsControlEdge()) {
      PadDataInputEdges++;
    }
  }
  DCHECK_EQ(PadDataInputEdges, 2);

  // Conv2D must have 2 data inputs: Pad output and Filter
  // FusedConv2D have 3 data inputs: Pad output, Filter and Args;
  int ConvDataInputEdges = 0;
  for (const Edge* e : succ->in_edges()) {
    if (!e->IsControlEdge()) {
      ConvDataInputEdges++;
    }
  }

  DCHECK_EQ(ConvDataInputEdges, is_fused_conv2d ? 3 : 2);

  // We will use the node name of Conv2D as the name of new node
  // Build new node. We use same name as original node, but change the op
  // name.

  NodeBuilder nb(succ->name(), is_fused_conv2d ? csinfo_.pad_with_fused_conv2d
                                               : csinfo_.pad_with_conv2d);
  nb.Input(pred_in[0].first, pred_in[0].second);  // In1 (input data)  of Pad
  // pred_in[1] will be 2nd Tensorflow tensor for Conv2D.
  nb.Input(succ_in[1].first, succ_in[1].second);  // In2 (filter) of conv2d
  // In1 of Conv2D is same as output of Pad.
  // Thus, only need to add In2 of Conv2D

  if (is_fused_conv2d) {
    // FusedConv2D has one additional input, args
    std::vector<NodeBuilder::NodeOut> args;
    args.emplace_back(succ_in[2].first, succ_in[2].second);
    nb.Input(gtl::ArraySlice<NodeBuilder::NodeOut>{
        args});                                     // In3 (args) of FusedConv2D
    nb.Input(pred_in[1].first, pred_in[1].second);  // In2 (paddings) of Pad
    // Copy attributes from Pad and FusedConv2D to PadWithFusedConv2D.
    CopyAttrsFromPadAndFusedConv2D(const_cast<const Node*>(succ),
                                   const_cast<const Node*>(pred), &nb);
  } else {
    nb.Input(pred_in[1].first, pred_in[1].second);  // In2 (paddings) of Pad
    // Copy attributes from Pad and conv2D to PadWithConv2D.
    CopyAttrsFromPadAndConv2D(const_cast<const Node*>(succ),
                              const_cast<const Node*>(pred), &nb);
  }

  // Copy the device assigned to old node to new node.
  nb.Device(succ->def().device());

  // Create node.
  Node* new_node;
  TF_CHECK_OK(nb.Finalize(&**g, &new_node));
  // No need to check if new_node is null because it will be null only when
  // Finalize fails.

  // Incoming data edges from 'pred' node and 'succ' node to new 'new_node'
  // node are already copied in BuildNode.
  // We handle control edges now.
  for (const Edge* e : pred->in_edges()) {
    if (e->IsControlEdge()) {
      // Don't allow duplicate edge
      (*g)->AddControlEdge(e->src(), new_node, false);
    }
  }
  for (const Edge* e : succ->in_edges()) {
    if (e->IsControlEdge()) {
      // Don't allow duplicate edge
      (*g)->AddControlEdge(e->src(), new_node, false);
    }
  }

  // Incoming edges are fixed, we will fix the outgoing edges now.
  // First, we will fix outgoing control edges from 'pred' node.
  for (const Edge* e : pred->out_edges()) {
    if (e->IsControlEdge()) {
      // Don't allow duplicate edge
      (*g)->AddControlEdge(new_node, e->dst(), false);
    }
  }

  // Second, we will fix outgoing control and data edges from 'succ' node.
  for (const Edge* e : succ->out_edges()) {
    if (e->IsControlEdge()) {
      // Allow duplicate while adding control edge as it would fail (return
      // NULL) if we try to add duplicate edge.
      (*g)->AddControlEdge(new_node, e->dst(), false);
    } else {
      // Conv2D has only 1 output (at slot 0) and merged node also has only 1
      // output (at slot 0).
      const int kPadWithConv2DOutputSlot = 0;
      (*g)->AddEdge(new_node, kPadWithConv2DOutputSlot, e->dst(),
                    e->dst_input());
    }
  }

  // Copy device assigned to old node to new node.
  // It's ok to use pred or succ as we have enforced a check that
  // both have same device assigned.
  new_node->set_assigned_device_name(pred->assigned_device_name());

  VLOG(1) << "MklLayoutRewritePass: Merged old node:" << pred->DebugString()
          << ", and node: " << succ->DebugString()
          << ", into node:" << new_node->DebugString();

  (*g)->RemoveNode(succ);
  (*g)->RemoveNode(pred);

  return Status::OK();
}

Status MklLayoutRewritePass::MergeConv2DBackpropFilterWithBiasAddGrad(
    std::unique_ptr<Graph>* g, Node* m, Node* n) {
  CHECK_EQ(((m->type_string() == csinfo_.bias_add_grad &&
             n->type_string() == csinfo_.conv2d_grad_filter)) ||
               ((n->type_string() == csinfo_.bias_add_grad &&
                 m->type_string() == csinfo_.conv2d_grad_filter)),
           true);

  // If 'm' is BiasAddGrad, then 'n' is BackpropFilter.
  Node* badd = m->type_string() == csinfo_.bias_add_grad ? m : n;
  Node* fltr = m->type_string() == csinfo_.bias_add_grad ? n : m;

  // Sanity check for attributes from input nodes.
  DataType T_b, T_f;
  string data_format_b, data_format_f;
  TF_CHECK_OK(GetNodeAttr(badd->def(), "T", &T_b));
  TF_CHECK_OK(GetNodeAttr(fltr->def(), "T", &T_f));
  TF_CHECK_OK(GetNodeAttr(badd->def(), "data_format", &data_format_b));
  TF_CHECK_OK(GetNodeAttr(fltr->def(), "data_format", &data_format_f));
  if (data_format_b != data_format_f || T_b != T_f ||
      badd->assigned_device_name() != fltr->assigned_device_name() ||
      badd->def().device() != fltr->def().device()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "data_format or T attribute or devices of "
                  "Conv2DBackpropFilter and BiasAddGrad do not match. "
                  "Will skip node merge optimization");
  }

  // We will use the node name of Conv2DBackpropFilter as the name of new node.
  // This is because BackpropFilterWithBias is going to emit bias output also.
  NodeBuilder nb(fltr->name(), csinfo_.conv2d_grad_filter_with_bias);
  // Since Conv2DBackpropFilterWithBias has same number of inputs as
  // Conv2DBackpropFilter, we can just copy input edges directly. We dont need
  // to copy any data input of BiasAddGrad because that input also goes to
  // Conv2DBackpropFilter.
  const int fltr_ins = fltr->num_inputs();
  gtl::InlinedVector<Node*, 4> fltr_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> fltr_in_edges(fltr_ins);
  FillInputs(fltr, &fltr_control_edges, &fltr_in_edges);
  for (int idx = 0; idx < fltr_ins; idx++) {
    nb.Input(fltr_in_edges[idx].first, fltr_in_edges[idx].second);
  }

  // Copy attributes from Conv2DBackpropFilter.
  CopyAttrsConv(const_cast<const Node*>(fltr), &nb);

  // Copy the device assigned to old node to new node.
  nb.Device(fltr->def().device());

  // Create node.
  Node* new_node;
  TF_CHECK_OK(nb.Finalize(&**g, &new_node));

  // In the following code of this function, an unsorted set is used to make
  // sure no duplicated edges be added into the new node. Therefore, we can
  // pass allow_duplicates = true in AddControlEdge call to skip the O(#edges)
  // check in the routine.

  // Incoming data edges from BiasAddGrad node and Conv2DBackpropFilter node to
  // new 'new_node' node are already copied in BuildNode. We handle control
  // edges now.
  std::unordered_set<Node*> unique_node;
  for (const Edge* e : badd->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), new_node, true);
      }
    }
  }
  unique_node.clear();
  for (const Edge* e : fltr->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), new_node, true);
      }
    }
  }
  unique_node.clear();

  // Incoming edges are fixed, we will fix the outgoing edges now.
  // First, we will fix outgoing control edges from 'badd' node.
  // Conv2DBackpropFilter has 1 output -- filter_grad.
  // Conv2DBackpropFilterWithBias has 2 outputs -- filter_grad and
  // bias_grad. But filter_grad is at same slot number (0) in both the
  // nodes. bias_grad is at slot number 1 in Conv2DBackpropFilterWithBias, while
  // it is at slot number 0 in BiasAddGrad.
  const int kMergedNodeFilterGradOutputIdx = 0;
  const int kMergedNodeBiasGradOutputIdx = 1;

  for (const Edge* e : badd->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(new_node, e->dst(), true);
      }
    } else {
      CHECK_NOTNULL((*g)->AddEdge(new_node, kMergedNodeBiasGradOutputIdx,
                                  e->dst(), e->dst_input()));
    }
  }
  unique_node.clear();

  // Second, we will fix outgoing control and data edges from 'fltr' node.
  for (const Edge* e : fltr->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(new_node, e->dst(), true);
      }
    } else {
      CHECK_NOTNULL((*g)->AddEdge(new_node, kMergedNodeFilterGradOutputIdx,
                                  e->dst(), e->dst_input()));
    }
  }

  // Copy device assigned to old node to new node.
  // It's ok to use badd or fltr as we have enforced a check that
  // both have same device assigned.
  new_node->set_assigned_device_name(badd->assigned_device_name());

  VLOG(1) << "MklLayoutRewritePass: Merged old node:" << badd->DebugString()
          << ", and node: " << fltr->DebugString()
          << ", into node:" << new_node->DebugString();

  (*g)->RemoveNode(badd);
  (*g)->RemoveNode(fltr);

  return Status::OK();
}

Status MklLayoutRewritePass::MergeNode(std::unique_ptr<Graph>* g, Node* m,
                                       Node* n) {
  CHECK_NOTNULL(m);
  CHECK_NOTNULL(n);

  if (((m->type_string() == csinfo_.bias_add &&
        n->type_string() == csinfo_.conv2d)) ||
      ((n->type_string() == csinfo_.bias_add &&
        m->type_string() == csinfo_.conv2d))) {
    return this->MergeConv2DWithBiasAdd(g, m, n);
  }
  if ((m->type_string() == csinfo_.pad &&
       (n->type_string() == csinfo_.conv2d ||
        (n->type_string() == csinfo_.fused_conv2d && FusedConv2DRewrite(n)))) ||
      (n->type_string() == csinfo_.pad &&
       (m->type_string() == csinfo_.conv2d ||
        (m->type_string() == csinfo_.fused_conv2d && FusedConv2DRewrite(m))))) {
    return this->MergePadWithConv2D(g, m, n);
  }

  if (((m->type_string() == csinfo_.bias_add_grad &&
        n->type_string() == csinfo_.conv2d_grad_filter)) ||
      ((n->type_string() == csinfo_.bias_add_grad &&
        m->type_string() == csinfo_.conv2d_grad_filter))) {
    return this->MergeConv2DBackpropFilterWithBiasAddGrad(g, m, n);
  }

  return Status(error::Code::UNIMPLEMENTED,
                "Unimplemented case for node merge optimization.");
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions for node rewrite
//////////////////////////////////////////////////////////////////////////

Status MklLayoutRewritePass::RewriteNodeForLayoutPropagation(
    std::unique_ptr<Graph>* g, const Node* orig_node, Node** new_node,
    const RewriteInfo* ri) {
  // Get all data inputs.
  int num_data_inputs = orig_node->in_edges().size();
  // Drop count for control edges from inputs
  for (const Edge* e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      num_data_inputs--;
    }
  }

  gtl::InlinedVector<Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> inputs(num_data_inputs);
  FillInputs(orig_node, &control_edges, &inputs);

  // Build new node. We use same name as original node, but change the op name.
  NodeBuilder nb(orig_node->name().c_str(), ri->new_name.c_str());
  // Copy user-specified device assigned to original node to new node.
  nb.Device(orig_node->def().device());
  // Set up new inputs to the rewritten node.
  Status s = SetUpInputs(g, inputs, &nb, orig_node);
  if (s != Status::OK()) {
    return s;
  }

  const bool kPartialCopyAttrs = false;
  ri->copy_attrs(const_cast<const Node*>(orig_node), &nb, kPartialCopyAttrs);

  // Set the Mkl layer label for this op.
  if (DataTypeIsQuantized(orig_node->input_type(0)) ||
      DataTypeIsQuantized(orig_node->output_type(0))) {
    nb.Attr("_kernel", mkl_op_registry::kMklQuantizedOpLabel);
  } else {
    nb.Attr("_kernel", mkl_op_registry::kMklLayoutDependentOpLabel);
  }
  // Finalize graph and get new node.
  s = nb.Finalize(&**g, new_node);
  if (s != Status::OK()) {
    return s;
  }

  // In the following code of this function, an unsorted set is used to make
  // sure no duplicated edges be added into the new node. Therefore, we can
  // pass allow_duplicates = true in AddControlEdge call to skip the O(#edges)
  // check in the routine.

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  std::unordered_set<Node*> unique_node;
  for (const Edge* e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), *new_node, true);
      }
    }
  }
  unique_node.clear();

  // Copy outgoing edges from 'orig_node' node to new
  // 'new_node' node, since the output also follows same ordering among
  // Tensorflow tensors and Mkl tensors. We need to connect Tensorflow
  // tensors appropriately. Specifically, nth output of the original node
  // will become 2*nth output of the Mkl node for the interleaved ordering
  // of the tensors. For the contiguous ordering of the tensors, it will be n.
  // GetTensorDataIndex provides this mapping function.
  for (const Edge* e : orig_node->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(*new_node, e->dst(), true);
      }
    } else {
      CHECK_NOTNULL((*g)->AddEdge(
          *new_node,
          GetTensorDataIndex(e->src_output(), e->src()->num_outputs()),
          e->dst(), e->dst_input()));
    }
  }
  return Status::OK();
}

Status MklLayoutRewritePass::RewriteNodeForJustOpNameChange(
    std::unique_ptr<Graph>* g, const Node* orig_node, Node** new_node,
    const RewriteInfo* ri) {
  // Get all data inputs.
  int num_data_inputs = orig_node->in_edges().size();
  // Drop count for control edges from inputs
  for (const Edge* e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      num_data_inputs--;
    }
  }
  gtl::InlinedVector<Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> inputs(num_data_inputs);
  FillInputs(orig_node, &control_edges, &inputs);

  // Build new node. We use same name as original node, but change the op name.
  NodeBuilder nb(orig_node->name().c_str(), ri->new_name.c_str());
  // Copy user-specified device assigned to original node to new node.
  nb.Device(orig_node->def().device());

  Status s = CopyInputs(orig_node, inputs, &nb);
  if (s != Status::OK()) {
    return s;
  }

  ri->copy_attrs(const_cast<const Node*>(orig_node), &nb, true);
  nb.Attr("_kernel", mkl_op_registry::kMklNameChangeOpLabel);

  // Finalize graph and get new node.
  s = nb.Finalize(&**g, new_node);
  if (s != Status::OK()) {
    return s;
  }

  // In the following code of this function, an unsorted set is used to make
  // sure no duplicated edges be added into the new node. Therefore, we can
  // pass allow_duplicates = true in AddControlEdge call to skip the O(#edges)
  // check in the routine.

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  std::unordered_set<Node*> unique_node;
  for (const Edge* e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), *new_node, true);
      }
    }
  }
  unique_node.clear();

  // Transfer outgoing edges from 'orig_node' node to new 'new_node' node.
  for (const Edge* e : orig_node->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(*new_node, e->dst(), true);
      }
    } else {
      auto result =
          (*g)->AddEdge(*new_node, e->src_output(), e->dst(), e->dst_input());
      DCHECK(result != nullptr);
    }
  }

  return Status::OK();
}

Status MklLayoutRewritePass::RewriteNode(std::unique_ptr<Graph>* g,
                                         Node* orig_node,
                                         const RewriteInfo* ri) {
  DCHECK(ri != nullptr);
  DCHECK(orig_node != nullptr);

  VLOG(1) << "MklLayoutRewritePass: Original node:" << orig_node->DebugString();

  Status ret_status = Status::OK();
  Node* new_node = nullptr;
  if (ri->rewrite_cause == kRewriteForLayoutPropagation) {
    ret_status = RewriteNodeForLayoutPropagation(g, orig_node, &new_node, ri);
  } else if (ri->rewrite_cause == kRewriteForOpNameChange) {
    ret_status = RewriteNodeForJustOpNameChange(g, orig_node, &new_node, ri);
  } else {
    ret_status = Status(error::Code::INVALID_ARGUMENT,
                        "Unsupported rewrite cause found."
                        "RewriteNode will fail.");
  }
  TF_CHECK_OK(ret_status);

  // Copy the runtime device assigned from original code to new node.
  new_node->set_assigned_device_name(orig_node->assigned_device_name());

  // Delete original node and mark new node as rewritten.
  (*g)->RemoveNode(orig_node);

  VLOG(1) << "MklLayoutRewritePass: New node:" << new_node->DebugString();
  return ret_status;
}

// TODO(mdfaijul): Is there any other elegent way to check for quantized ops
// having attributes other than "T"?
// Current implementation reflects only QuantizedConv2D and its fused Ops.
const MklLayoutRewritePass::RewriteInfo*
MklLayoutRewritePass::CheckForQuantizedNodeRewrite(const Node* n) const {
  DataType T1, T2;
  DataType Tinput, Tfilter;
  bool type_attrs_present = false;

  if (TryGetNodeAttr(n->def(), "Tinput", &Tinput) &&
      TryGetNodeAttr(n->def(), "Tfilter", &Tfilter) &&
      mkl_op_registry::IsMklLayoutDependentOp(
          mkl_op_registry::GetMklOpName(n->type_string()), Tinput, Tfilter)) {
    type_attrs_present = true;
  } else if (TryGetNodeAttr(n->def(), "T1", &T1) &&
             TryGetNodeAttr(n->def(), "T2", &T2) &&
             mkl_op_registry::IsMklLayoutDependentOp(
                 mkl_op_registry::GetMklOpName(n->type_string()), T1, T2)) {
    type_attrs_present = true;
  }

  if (type_attrs_present) {
    for (auto ri = rinfo_.cbegin(); ri != rinfo_.cend(); ++ri) {
      if (n->type_string().compare(ri->name) == 0 && ri->rewrite_rule(n)) {
        return &*ri;
      }
    }
  }

  return nullptr;
}

const MklLayoutRewritePass::RewriteInfo*
MklLayoutRewritePass::CheckForNodeRewrite(const Node* n) const {
  CHECK_NOTNULL(n);

  // QuantizedOps may have attributes other than "T", so decoupled the check
  // with a function, CheckForQuantizedNodeRewrite(const Node*).
  const RewriteInfo* ri = CheckForQuantizedNodeRewrite(n);
  if (ri != nullptr) return ri;

  // First check if node along with its type is supported by MKL layer.
  // We do not want to rewrite an op into Mkl op if types are not supported.
  // E.g., MklRelu does not support INT32. So we cannot rewrite Relu to
  // MklRelu if type is INT32.
  DataType T;
  if (!TryGetNodeAttr(n->def(), "T", &T)) {
    return nullptr;
  }

  // We make an exception for Conv2D, as the corresponding MKL ops
  // currently do not support the case of padding == EXPLICIT yet.
  if (n->type_string() == csinfo_.conv2d ||
      n->type_string() == csinfo_.conv2d_grad_input ||
      n->type_string() == csinfo_.conv2d_grad_filter) {
    string padding;
    TF_CHECK_OK(GetNodeAttr(n->def(), "padding", &padding));
    if (padding == "EXPLICIT") return nullptr;
  }

  // We make an exception for __MklDummyConv2DWithBias,
  // __MklConv2DBackpropFilterWithBias, and __MklDummyPadWithConv2D since their
  // names do not match Mkl node names.
  if (n->type_string() != csinfo_.conv2d_with_bias &&
      n->type_string() != csinfo_.pad_with_conv2d &&
      n->type_string() != csinfo_.pad_with_fused_conv2d &&
      n->type_string() != csinfo_.conv2d_grad_filter_with_bias &&
      n->type_string() != csinfo_.fused_conv2d &&
      n->type_string() != csinfo_.fused_depthwise_conv2d &&
      n->type_string() != csinfo_.fused_matmul &&
      !mkl_op_registry::IsMklOp(mkl_op_registry::GetMklOpName(n->type_string()),
                                T)) {
    return nullptr;
  }

  // We now check if rewrite rule applies for this op. If rewrite rule passes
  // for this op, then we rewrite it to Mkl op.
  // Find matching RewriteInfo and then check that rewrite rule applies.
  for (auto ri = rinfo_.cbegin(); ri != rinfo_.cend(); ++ri) {
    if (n->type_string().compare(ri->name) == 0 && ri->rewrite_rule(n)) {
      return &*ri;
    }
  }

  // Else return not found.
  return nullptr;
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions for node fusion
//////////////////////////////////////////////////////////////////////////
Status MklLayoutRewritePass::FuseTransposeMklOpTranspose(
    std::unique_ptr<Graph>* g, std::vector<Node*>& nodes,
    std::function<void(const Node*, NodeBuilder* nb, bool)> copy_attrs,
    string data_format) {
  Node* transpose_to_nhwc = nodes[0];
  Node* mklop = nodes[1];
  Node* transpose_to_nchw = nodes[2];

  const int transpose_nhwc_num_inputs = transpose_to_nhwc->num_inputs();
  gtl::InlinedVector<Node*, 4> transpose_nhwc_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> transpose_nhwc_in(
      transpose_nhwc_num_inputs);
  FillInputs(transpose_to_nhwc, &transpose_nhwc_control_edges,
             &transpose_nhwc_in);

  const int mklop_num_inputs = mklop->num_inputs();
  gtl::InlinedVector<Node*, 4> mklop_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> mklop_in(mklop_num_inputs);
  FillInputs(mklop, &mklop_control_edges, &mklop_in);

  const int transpose_nchw_num_inputs = transpose_to_nchw->num_inputs();
  gtl::InlinedVector<Node*, 4> transpose_nchw_control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> transpose_nchw_in(
      transpose_nchw_num_inputs);
  FillInputs(transpose_to_nchw, &transpose_nchw_control_edges,
             &transpose_nchw_in);

  // We use same name as original node, but change the op
  // type.
  NodeBuilder nb(mklop->name(), mklop->type_string());

  // Storing the output slots of the input nodes.
  for (int i = 0; i < mklop_num_inputs; i++) {
    if (mklop_in[i].first == transpose_to_nhwc) {
      // Fill "x":
      nb.Input(transpose_nhwc_in[0].first, transpose_nhwc_in[0].second);
    } else {
      // Fill inputs other than "x":
      nb.Input(mklop_in[i].first, mklop_in[i].second);
    }
  }

  copy_attrs(const_cast<const Node*>(mklop), &nb, true);
  nb.Attr("data_format", data_format);

  // Copy the device assigned to old node to new node.
  nb.Device(mklop->def().device());

  // Create node.
  Node* new_node;
  TF_CHECK_OK(nb.Finalize(&**g, &new_node));
  // No need to check if new_node is null because it will be null only when
  // Finalize fails.

  // Fill outputs.
  for (const Edge* e : transpose_to_nchw->out_edges()) {
    if (!e->IsControlEdge()) {
      const int kTransposeWithMklOpOutputSlot = 0;
      auto new_edge = (*g)->AddEdge(new_node, kTransposeWithMklOpOutputSlot,
                                    e->dst(), e->dst_input());
      DCHECK(new_edge);
    }
  }

  // Copy device assigned to old node to new node.
  new_node->set_assigned_device_name(mklop->assigned_device_name());

  // Copy requested_device and assigned_device_name_index
  new_node->set_requested_device(mklop->requested_device());
  new_node->set_assigned_device_name_index(mklop->assigned_device_name_index());

  (*g)->RemoveNode(transpose_to_nhwc);
  (*g)->RemoveNode(mklop);
  (*g)->RemoveNode(transpose_to_nchw);

  return Status::OK();
}

Status MklLayoutRewritePass::FuseNode(
    std::unique_ptr<Graph>* g, std::vector<Node*>& nodes,
    const MklLayoutRewritePass::FusionInfo fi) {
  return fi.fuse_func(g, nodes, fi.copy_attrs);
}

std::tuple<bool, std::vector<Node*>, const MklLayoutRewritePass::FusionInfo>
MklLayoutRewritePass::CheckForNodeFusion(Node* a) const {
  // Stores matched nodes, in the same order as node_checkers.
  std::vector<Node*> nodes;

  for (auto fi = finfo_.begin(); fi != finfo_.end(); ++fi) {
    //
    // Make sure node "a" and its succeding nodes (b, c ...), match the pattern
    // defined in fusion info (ops[0], ops[1], ...),
    // a.k.a. "a->b->c" matches "op1->op2->op3"
    //

    // Stores the first unvisited outgoing edge of each matched node in "nodes".
    std::stack<EdgeSet::const_iterator> current_neighbor_stack;
    nodes.clear();

    auto node_checker = fi->node_checkers.begin();
    if (a != nullptr && (*node_checker)(a)) {
      nodes.push_back(a);
      current_neighbor_stack.push(a->out_edges().begin());
      ++node_checker;
    }

    while (!nodes.empty()) {
      auto& current_neighbor_iter = current_neighbor_stack.top();

      if (current_neighbor_iter != nodes.back()->out_edges().end()) {
        // Found an unvisited edge. Goes through the edge to get the neighbor.
        Node* neighbor_node = (*current_neighbor_iter)->dst();
        ++current_neighbor_stack.top();  // Retrieves the next unvisited edge.

        if ((*node_checker)(neighbor_node)) {
          // Found a match. Stores the node and moves to the next checker.
          nodes.push_back(neighbor_node);
          current_neighbor_stack.push(neighbor_node->out_edges().begin());
          if (++node_checker == fi->node_checkers.end()) {
            return make_tuple(true, nodes, *fi);
          }
        }
      } else {
        // Removes the current node since none of its neighbor leads to a
        // further match.
        nodes.pop_back();
        current_neighbor_stack.pop();
        --node_checker;
      }
    }
  }

  return make_tuple(false, std::vector<Node*>(), FusionInfo());
}

///////////////////////////////////////////////////////////////////////////////
//              Post-rewrite Mkl metadata fixup pass
///////////////////////////////////////////////////////////////////////////////
bool MklLayoutRewritePass::FixMklMetaDataEdgeIfNeeded(std::unique_ptr<Graph>* g,
                                                      const Edge* e_data,
                                                      const Edge* e_metadata) {
  if (g == nullptr || e_data == nullptr || e_metadata == nullptr) {
    return false;
  }

  Node* n_data = e_data->src();
  int n_data_op_slot = e_data->src_output();
  int n_metadata_op_slot =
      GetTensorMetaDataIndex(n_data_op_slot, n_data->num_outputs());

  // If the source of meta edge is a constant node (producing dummy Mkl metadata
  // tensor), then we will need to fix.
  if (IsConstant(e_metadata->src())) {
    Node* e_metadata_dst = e_metadata->dst();
    int e_metadata_in_slot = e_metadata->dst_input();
    CHECK_NOTNULL((*g)->AddEdge(n_data, n_metadata_op_slot, e_metadata_dst,
                                e_metadata_in_slot));

    (*g)->RemoveEdge(e_metadata);
    return true;
  }

  return false;
}

bool MklLayoutRewritePass::FixMklMetaDataEdges(std::unique_ptr<Graph>* g,
                                               Node* n) {
  bool result = false;

  // If graph node is not Mkl node, then return.
  DataType T = DT_INVALID;
  if (!TryGetNodeAttr(n->def(), "T", &T) ||
      !mkl_op_registry::IsMklLayoutDependentOp(n->type_string(), T)) {
    return result;
  }

  // If it is Mkl node, then check if the input edges to this node that carry
  // Mkl metadata are linked up correctly with the source node.

  // For Mkl nodes, we generate twice the number of input tensors (n for Mkl
  // data tensors + n for Mkl metadata tensors). We need to check for correct
  // connection of n metadata tensors only.
  int num_data_inputs = n->num_inputs() / 2;
  for (int idx = 0; idx < num_data_inputs; idx++) {
    // Get the edge connecting input slot with index (idx).
    const Edge* e = nullptr;
    TF_CHECK_OK(n->input_edge(idx, &e));

    // If e is control edge, then skip.
    if (e->IsControlEdge()) {
      continue;
    }

    // Check that the source node for edge 'e' is Mkl node. If it is not an Mkl
    // node, then we don't need to do anything.
    Node* e_src = e->src();
    if (TryGetNodeAttr(e_src->def(), "T", &T) &&
        mkl_op_registry::IsMklLayoutDependentOp(e_src->type_string(), T)) {
      // Source node for edge 'e' is Mkl node.
      // Destination node and destination input slot of e is node 'n' and 'idx'
      // resp.
      CHECK_EQ(e->dst(), n);
      CHECK_EQ(e->dst_input(), idx);

      // Let's get edge that carries Mkl metadata corresponding to Mkl data edge
      // 'e'. For that, let's first get the input slot of 'n' where the meta
      // edge will feed the value.
      int e_meta_in_slot =
          GetTensorMetaDataIndex(e->dst_input(), n->num_inputs());
      const Edge* e_meta = nullptr;
      TF_CHECK_OK(n->input_edge(e_meta_in_slot, &e_meta));

      // Let's check if we need to fix this meta edge.
      if (FixMklMetaDataEdgeIfNeeded(g, e, e_meta)) {
        result = true;
      }
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
///////////////////////////////////////////////////////////////////////////////

bool MklLayoutRewritePass::RunPass(std::unique_ptr<Graph>* g) {
  bool result = false;
  CHECK_NOTNULL(g);

  DumpGraph("Before running MklLayoutRewritePass", &**g);

  std::vector<Node*> order;
  GetReversePostOrder(**g, &order);  // This will give us topological sort.
  for (Node* n : order) {
    // If node is not an op or it cannot run on CPU device, then skip.
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }

    Node* m = nullptr;
    if ((m = CheckForNodeMerge(n)) != nullptr && CanOpRunOnCPUDevice(m)) {
      // Check if the node 'n' can be merged with any other node. If it can
      // be 'm' contains the node with which it can be merged.
      string n1_name = n->name();
      string n2_name = m->name();

      VLOG(1) << "MklLayoutRewritePass: Scheduled nodes " << n1_name << " and "
              << n2_name << " for merging";

      if (MergeNode(g, n, m) == Status::OK()) {
        VLOG(1) << "MklLayoutRewritePass: Merged nodes " << n1_name << " and "
                << n2_name;
        result = true;
      }
    }
  }

  DumpGraph("After running MklLayoutRewritePass(NodeMerge)", &**g);

  order.clear();
  GetReversePostOrder(**g, &order);  // This will give us topological sort.
  for (Node* n : order) {
    // If node is not an op or it cannot run on CPU device, then skip.
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }

    auto check_result = CheckForNodeFusion(n);
    bool found_pattern = std::get<0>(check_result);
    std::vector<Node*> nodes = std::get<1>(check_result);
    const FusionInfo fi = std::get<2>(check_result);

    // if "found_pattern" is true, we can do the fusion.
    if (found_pattern) {
      if (FuseNode(g, nodes, fi) == Status::OK()) {
        result = true;
      }
    }
  }
  DumpGraph("After running MklLayoutRewritePass(NodeFusion)", &**g);

  order.clear();
  GetReversePostOrder(**g, &order);  // This will give us topological sort.
  for (Node* n : order) {
    // If node is not an op or it cannot run on CPU device, then skip.
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }

    const RewriteInfo* ri = nullptr;
    // We will first search if node is to be rewritten.
    if ((ri = CheckForNodeRewrite(n)) != nullptr) {
      string node_name = n->name();
      string op_name = n->type_string();

      VLOG(1) << "MklLayoutRewritePass: Scheduled node " << node_name
              << " with op " << op_name << " for rewrite using"
              << " layout optimization.";

      if (RewriteNode(g, n, ri) == Status::OK()) {
        VLOG(1) << "MklLayoutRewritePass: rewrote node " << node_name
                << " with op " << op_name << " for Mkl layout optimization.";
        result = true;
      }
    }
  }

  DumpGraph("After running MklLayoutRewritePass(NodeMerge+Rewrite)", &**g);

  order.clear();
  GetReversePostOrder(**g, &order);  // This will give us topological sort.
  for (Node* n : order) {
    // If node is not an op or it cannot run on CPU device, then skip.
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }
    if (FixMklMetaDataEdges(g, n)) {
      string node_name = n->name();
      string op_name = n->type_string();

      VLOG(1) << "MklLayoutRewritePass: fixed metadata edges for node "
              << node_name << " with op " << op_name;
      result = true;
    }
  }
  DumpGraph("After running MklLayoutRewritePass(NodeMerge+Rewrite+Fixup)",
            &**g);

  return result;
}

bool RunMklLayoutRewritePass(std::unique_ptr<Graph>* g) {
  return MklLayoutRewritePass().RunPass(g);
}

Status MklLayoutRewritePass::Run(const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return Status::OK();
  }
  if (DisableMKL()) {
    VLOG(2) << "TF-MKL: Disabling MKL";
    return Status::OK();
  }

  auto process_graph = [&](std::unique_ptr<Graph>* g) {
    // Get the ownership of a graph
    std::unique_ptr<Graph>* ng = std::move(g);
    RunPass(ng);
    // Return the ownership of a graph back
    g->reset(ng->release());
  };

  if (kMklLayoutRewritePassGroup !=
      OptimizationPassRegistry::POST_PARTITIONING) {
    // For any pre-partitioning phase, a graph is stored in options.graph.
    process_graph(options.graph);
  } else {
    // For post partitioning phase, graphs are stored in
    // options.partition_graphs.
    for (auto& pg : *options.partition_graphs) {
      process_graph(&pg.second);
    }
  }

  return Status::OK();
}

}  // namespace tensorflow

#endif
