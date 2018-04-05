/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SYMBOLIC_SHAPES_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SYMBOLIC_SHAPES_H_

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"

namespace tensorflow {
namespace grappler {

bool IsKnown(const TensorShapeProto::Dim& dim);
bool IsKnownSymbolically(const TensorShapeProto::Dim& dim);
bool IsUnknown(const TensorShapeProto::Dim& dim);

// Shape is symbolically defined, if it has a known rank, and each dimension is
// known (dim_size >= 0), or is a symbolic dimension size (dim_size <= -2).
bool ShapeIsSymbolicallyDefined(const TensorShapeProto& shape);
bool ShapeIsSymbolicallyDefined(const OpInfo::TensorProperties& properties);

// Shapes are symbolically equal, if they have the same rank, they are known or
// symbolically defined, and have matching dimensions.
bool ShapesSymbolicallyEqual(const TensorShapeProto& left,
                             const TensorShapeProto& right);
bool ShapesSymbolicallyEqual(const OpInfo::TensorProperties& left,
                             const OpInfo::TensorProperties& right);

// Check if two shapes can be broadcasted to each other. Both shapes must be at
// least symbolically defined, and the have valid BCast instance.
bool ShapesBroadcastable(const TensorShapeProto& left,
                         const TensorShapeProto& right);
bool ShapesBroadcastable(const OpInfo::TensorProperties& left,
                         const OpInfo::TensorProperties& right);

// Return true if can prove, that tensor of size 'left' is smaller than tensor
// of size 'right'. Return false if it's larger or equal, or it's impossible to
// compare because of unknown dimensions, or mismatch in symbolic dimensions.
bool CompareSymbolicallyShapedTensorSizes(const TensorShapeProto& left,
                                          const TensorShapeProto& right);
bool CompareSymbolicallyShapedTensorSizes(
    const OpInfo::TensorProperties& left,
    const OpInfo::TensorProperties& right);

}  // namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_SYMBOLIC_SHAPES_H_
