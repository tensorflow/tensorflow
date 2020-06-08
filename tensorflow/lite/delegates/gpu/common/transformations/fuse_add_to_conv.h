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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_FUSE_ADD_TO_CONV_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_FUSE_ADD_TO_CONV_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Fuse Add Scalar or Add Broadcast after Convolution(Convolution2D,
// DepthWise, TransposedConvolution, FullyConnected) into biases of
// convolution.
std::unique_ptr<SequenceTransformation> NewMergeConvolutionWithAdd();

// Fuse Add Scalar or Add Broadcast before Convolution(Convolution2D,
// DepthWise, FullyConnected) into biases of
// convolution.
std::unique_ptr<SequenceTransformation> NewMergeAddWithConvolution();

// Modify Convolution2DAttributes so that after making convolution with
// modified attributes we will have the same result as convolution
// with old attributes and following add operation.
void FuseConvolution2DWithAdd(const AddAttributes& add_attr,
                              Convolution2DAttributes* attr);

// Modify DepthwiseConvolution2DAttributes so that after making depth wise
// convolution with modified attributes we will have the same result as depth
// wise convolution with old attributes and following add operation.
void FuseDepthwiseConvolution2DWithAdd(const AddAttributes& add_attr,
                                       DepthwiseConvolution2DAttributes* attr);

// Modify ConvolutionTransposedAttributes so that after making convolution
// transposed with modified attributes we will have the same result as
// convolution transposed with old attributes and following add operation.
void FuseConvolutionTransposedWithAdd(const AddAttributes& add_attr,
                                      ConvolutionTransposedAttributes* attr);

// Modify FullyConnectedAttributes so that after making fully connected with
// modified attributes we will have the same result as fully connected
// with old attributes and following add operation.
void FuseFullyConnectedWithAdd(const AddAttributes& add_attr,
                               FullyConnectedAttributes* attr);

// Modify Convolution2DAttributes so that after making convolution with
// modified attributes we will have the same result as add operation and
// convolution with old attributes
void FuseAddWithConvolution2D(const AddAttributes& add_attr,
                              Convolution2DAttributes* attr);

// Modify DepthwiseConvolution2DAttributes so that after making depth wise
// convolution with modified attributes we will have the same result as add
// operation and depth wise convolution with old attributes
void FuseAddWithDepthwiseConvolution2D(const AddAttributes& add_attr,
                                       DepthwiseConvolution2DAttributes* attr);

// Modify FullyConnectedAttributes so that after making fully connected
// with modified attributes we will have the same result as add operation and
// fully connected with old attributes
void FuseAddWithFullyConnected(const AddAttributes& add_attr,
                               FullyConnectedAttributes* attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TRANSFORMATIONS_FUSE_ADD_TO_CONV_H_
