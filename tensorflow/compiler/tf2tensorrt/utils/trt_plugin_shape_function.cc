/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/tf2tensorrt/utils/trt_plugin_shape_function.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferPlugin.h"

namespace tensorflow {
namespace tensorrt {
namespace shape_inference {
    using tensorflow::shape_inference::InferenceContext;
    using tensorflow::shape_inference::ShapeHandle;

nvinfer1::Dims ShapeHandleToTrtDims(const ShapeHandle& shape, InferenceContext *c){
    nvinfer1::Dims dim;
    int rank=InferenceContext::Rank(shape);
    for(int i=0;i<rank;++i){
        dim.d[i]=c->Value(c->Dim(shape,i));
    }
    dim.nbDims=rank;
    return std::move(dim);
}

Status TRTPluginShapeFunction(
    tensorflow::shape_inference::InferenceContext* c) {
  std::vector<tensorflow::shape_inference::ShapeHandle> input_shapes;
  std::vector<nvinfer1::Dims> plugin_shapes;
  for (int i = 0; i < c->num_inputs(); ++i) {
    input_shapes.emplace_back(c->input(i));
    // Require fully defined shapes for the time being
    if (!c->FullyDefined(input_shapes.back())) {
      LOG(ERROR)
          << "Need shapes to be fully defined for plugin op shape inference!";
      return tensorflow::shape_inference::UnknownShape(c);
    }
    if (c->Rank(input_shapes.back()) > nvinfer1::Dims::MAX_DIMS) {
      LOG(ERROR) << "Input rank seems to be greater than "
                 << nvinfer1::Dims::MAX_DIMS
                 << " TRT can not handle this graph!";
      return tensorflow::shape_inference::UnknownShape(c);
    }
    // we should not ignore batch dim since it may not have a meaning for a
    // plugin.
    plugin_shapes.emplace_back(
        ShapeHandleToTrtDims(input_shapes.back(),c));
  }
  InitializeTrtPlugins();
  const auto attr_slice = c->attrs();
  nvinfer1::IPluginV2* plugin_ptr = nullptr;
  Status plugin_status =
      ConstructPlugin(attr_slice, "For_shapeInference", plugin_ptr, false);
  if (!plugin_status.ok()) {
    LOG(ERROR)
        << "Plugin construction failed. Returning unknown shape. Error was "
        << plugin_status;
    return tensorflow::shape_inference::UnknownShape(c);
  }
  TrtUniquePtrType<nvinfer1::IPluginV2> plugin(plugin_ptr);
  std::vector<nvinfer1::Dims> output_dimensions;
  int num_outputs = plugin->getNbOutputs();
  std::vector<tensorflow::shape_inference::ShapeHandle> out_shapes;
  for (int i = 0; i < num_outputs; ++i) {
    output_dimensions.emplace_back(plugin->getOutputDimensions(
        i, &plugin_shapes[0], (int)plugin_shapes.size()));
    const nvinfer1::Dims& dim = output_dimensions[i];
    std::vector<tensorflow::shape_inference::DimensionHandle> dims;
    for (int k = 0; k < dim.nbDims; ++k) {
      dims.emplace_back(std::move(c->MakeDim(dim.d[k])));
    }
    auto shape = c->MakeShape(dims);
    c->set_output(i, shape);
  }

  return Status::OK();
}
}  // namespace shape_inference
}  // namespace tensorrt
}  // namespace tensorflow
#else
namespace tensorflow {
namespace tensorrt {
namespace shape_inference {
Status TRTPluginShapeFunction(
    tensorflow::shape_inference::InferenceContext* c) {
  return shape_inference::UnknownShape(c);
}
}  // namespace shape_inference
}  // namespace tensorrt
}  // namespace tensorflow

#endif
#else
namespace tensorflow {
namespace tensorrt {
namespace shape_inference {
Status TRTPluginShapeFunction(
    tensorflow::shape_inference::InferenceContext* c) {
  return shape_inference::UnknownShape(c);
}
}  // namespace shape_inference
}  // namespace tensorrt
}  // namespace tensorflow

#endif
