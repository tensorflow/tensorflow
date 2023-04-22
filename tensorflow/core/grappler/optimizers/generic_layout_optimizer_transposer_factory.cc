/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"

#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow {
namespace grappler {

std::shared_ptr<Transposer> TransposerFactory::GetTransposer(
    const NodeDef& node) {
  // Check layout sensitive ops.
  if (IsDefaultLayoutSensitiveOp(node)) {
    return GetOrCreateIfNotFound<DefaultLayoutSensitiveOpTransposer>(
        "DefaultLayoutSensitiveOp");
  }
  if (IsAvgPoolGrad(node)) {
    return GetOrCreateIfNotFound<AvgPoolGradTransposer>("AvgPoolGrad");
  }
  if (IsBiasAddV2(node)) {
    return GetOrCreateIfNotFound<BiasAddTransposer>("BiasAdd");
  }
  if (IsBiasAddGrad(node)) {
    return GetOrCreateIfNotFound<BiasAddGradTransposer>("BiasAddGrad");
  }
  if (IsConv2DBackpropFilter(node) ||
      IsDepthwiseConv2dNativeBackpropFilter(node)) {
    return GetOrCreateIfNotFound<Conv2DBackpropFilterTransposer>(
        "Conv2DBackpropFilter");
  }
  if (IsConv2DBackpropInput(node) ||
      IsDepthwiseConv2dNativeBackpropInput(node)) {
    return GetOrCreateIfNotFound<Conv2DBackpropInputTransposer>(
        "Conv2DBackpropInput");
  }
  if (IsConv3D(node)) {
    return GetOrCreateIfNotFound<Conv3DTransposer>("Conv3D");
  }
  if (IsConv3DBackpropInputV2(node)) {
    return GetOrCreateIfNotFound<Conv3DBackpropInputTransposer>(
        "Conv3DBackpropInput");
  }
  if (IsConv3DBackpropFilterV2(node)) {
    return GetOrCreateIfNotFound<Conv3DBackpropFilterTransposer>(
        "Conv3DBackpropFilter");
  }
  if (IsFusedBatchNormEx(node)) {
    return GetOrCreateIfNotFound<FusedBatchNormExTransposer>(
        "FusedBatchNormEx");
  }
  if (IsFusedBatchNormGrad(node)) {
    return GetOrCreateIfNotFound<FusedBatchNormGradTransposer>(
        "FusedBatchNormGrad");
  }
  if (IsMaxPoolV2(node)) {
    return GetOrCreateIfNotFound<MaxPoolV2Transposer>("MaxPoolV2");
  }
  if (IsMaxPoolGrad(node) || IsMaxPoolGradGradV1(node)) {
    return GetOrCreateIfNotFound<MaxPoolGradTransposer>("MaxPoolGrad");
  }
  if (IsMaxPoolGradV2(node) || IsMaxPoolGradGradV2(node)) {
    return GetOrCreateIfNotFound<MaxPoolGradV2Transposer>("MaxPoolGradV2");
  }
  // Check layout agnostic ops.
  if (IsDefaultLayoutAgnosticOp(node)) {
    return GetOrCreateIfNotFound<DefaultLayoutAgnosticOpTransposer>(
        "DefaultLayoutAgnosticOp");
  }
  if (IsAddN(node)) {
    return GetOrCreateIfNotFound<AddNTransposer>("AddN");
  }
  if (IsBinaryOp(node)) {
    return GetOrCreateIfNotFound<BinaryOpTransposer>("BinaryOp");
  }
  if (IsConcat(node)) {
    return GetOrCreateIfNotFound<ConcatOpTransposer>("Concat");
  }
  if (IsFill(node)) {
    return GetOrCreateIfNotFound<FillOpTransposer>("Fill");
  }
  if (IsIdentityN(node)) {
    return GetOrCreateIfNotFound<IdentityNTransposer>("IdentityN");
  }
  if (IsMerge(node)) {
    return GetOrCreateIfNotFound<MergeTransposer>("Merge");
  }
  if (IsMirrorPad(node) || IsMirrorPadGrad(node) || IsPad(node)) {
    return GetOrCreateIfNotFound<PadTransposer>("Pad");
  }
  if (IsReduceOp(node)) {
    return GetOrCreateIfNotFound<ReduceTransposer>("ReduceOp");
  }
  if (IsReverseV2(node)) {
    return GetOrCreateIfNotFound<ReverseV2Transposer>("ReverseV2");
  }
  if (IsSelect(node)) {
    return GetOrCreateIfNotFound<SelectTransposer>("Select");
  }
  if (IsShape(node)) {
    return GetOrCreateIfNotFound<ShapeTransposer>("Shape");
  }
  if (IsShapeN(node)) {
    return GetOrCreateIfNotFound<ShapeNTransposer>("ShapeN");
  }
  if (IsSlice(node)) {
    return GetOrCreateIfNotFound<SliceTransposer>("Slice");
  }
  if (IsSplit(node)) {
    return GetOrCreateIfNotFound<SplitTransposer>("Split");
  }
  if (IsSplitV(node)) {
    return GetOrCreateIfNotFound<SplitVTransposer>("SplitV");
  }
  if (IsSqueeze(node)) {
    return GetOrCreateIfNotFound<SqueezeTransposer>("Squeeze");
  }
  if (IsStridedSlice(node)) {
    return GetOrCreateIfNotFound<StridedSliceTransposer>("StridedSlice");
  }
  if (IsSwitch(node)) {
    return GetOrCreateIfNotFound<SwitchTransposer>("Switch");
  }
  if (IsTernaryOp(node)) {
    return GetOrCreateIfNotFound<TernaryOpTransposer>("TernaryOp");
  }
  if (IsTile(node)) {
    return GetOrCreateIfNotFound<TileTransposer>("Tile");
  }
  if (IsUnaryGrad(node)) {
    return GetOrCreateIfNotFound<UnaryGradTransposer>("UnaryGrad");
  }
  return nullptr;
}

}  // namespace grappler
}  // namespace tensorflow
