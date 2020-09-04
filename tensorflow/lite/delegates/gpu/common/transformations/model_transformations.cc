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

#include "tensorflow/lite/delegates/gpu/common/transformations/model_transformations.h"

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/custom_transformations.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_quant_adjustments.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/make_fully_connected.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/make_padding.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/remove_noop.h"

namespace tflite {
namespace gpu {

namespace {

bool ApplyGeneralTransformations(ModelTransformer* transformer) {
  // whenever any of these transforms return false, that means that a graph
  // is in the broken state and processing should not continue.
  return transformer->Apply("add_quant_adjustments",
                            NewAddQuantAdjustments().get()) &&
         transformer->Apply("remove_degenerate_upsampling",
                            NewRemoveDegenerateUpsampling().get()) &&
         transformer->Apply("remove_single_input_add",
                            NewRemoveSingleInputAdd().get()) &&
         transformer->Apply("remove_single_input_concat",
                            NewRemoveSingleInputConcat().get()) &&
         transformer->Apply("remove_identity_reshape",
                            NewRemoveIdentityReshape().get()) &&
         transformer->Apply("make_padding_from_concat",
                            NewMakePaddingFromConcat().get()) &&
         transformer->Apply("make_fully_connected_from_convolution",
                            NewMakeFullyConnectedFromConvolution().get()) &&
         transformer->Apply("merge_padding_with_convolution",
                            NewMergePaddingWithConvolution2D().get()) &&
         transformer->Apply("merge_padding_with_pooling",
                            NewMergePaddingWithPooling().get()) &&
         transformer->Apply("merge_padding_with_depthwise_convolution",
                            NewMergePaddingWithDepthwiseConvolution().get()) &&
         transformer->Apply("merge_convolution_with_mul",
                            NewMergeConvolutionWithMul().get()) &&
         transformer->Apply("merge_convolution_with_add",
                            NewMergeConvolutionWithAdd().get()) &&
         transformer->Apply("merge_mul_with_convolution",
                            NewMergeMulWithConvolution().get());
}

}  // namespace

bool ApplyModelTransformations(ModelTransformer* transformer) {
  return ApplyCustomTransformations(transformer) &&
         ApplyGeneralTransformations(transformer);
}

}  // namespace gpu
}  // namespace tflite
