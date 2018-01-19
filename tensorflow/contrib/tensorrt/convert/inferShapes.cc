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
#include "tensorflow/contrib/tensorrt/convert/inferShapes.h"
#include <functional>
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#define _TF_LOG_DEBUG ::tensorflow::internal::LogMessage(__FILE__, __LINE__, -1)

namespace tensorflow {
namespace trt {
std::vector<tensorflow::DataType> getTypes(const tensorflow::OpDef& op,
                                           const tensorflow::NodeDef& nd,
                                           bool inp = true) {
  const auto& attrMap = nd.attr();
  auto getType = [&attrMap](decltype(
                     op.input_arg(0)) a) -> std::vector<tensorflow::DataType> {
    std::vector<tensorflow::DataType> tvec;
    if (!a.type_list_attr().empty()) {  // get the list types
      const auto& tl = attrMap.at(a.type_list_attr()).list();
      int tsize = tl.type_size();
      tvec.reserve(tsize);
      for (int t = 0; t < tsize; t++) {
        tvec.push_back(tl.type(t));
      }
      return tvec;
    }
    tensorflow::DataType cType = tensorflow::DT_INVALID;
    if (a.type() != tensorflow::DT_INVALID) {  // get defined types
      cType = a.type();
    } else if (!a.type_attr().empty()) {
      cType = attrMap.at(a.type_attr()).type();
    }
    if (!a.number_attr().empty()) {  // numbertypes
      int64 nTensors = attrMap.at(a.number_attr()).i();
      tvec = std::vector<tensorflow::DataType>(nTensors, cType);
      return tvec;
    }
    tvec.push_back(cType);
    return tvec;
  };
  std::vector<tensorflow::DataType> types;
  if (inp) {
    int n_inputs = op.input_arg_size();
    for (int i = 0; i < n_inputs; i++) {
      auto tout = getType(op.input_arg(i));
      LOG(DEBUG) << "Node= " << nd.name() << " #inputs" << tout.size();
      types.insert(types.end(), tout.begin(), tout.end());
    }
  } else {
    int n_outputs = op.output_arg_size();
    // types.resize(n_outputs);
    for (int i = 0; i < n_outputs; i++) {
      auto tout = getType(op.output_arg(i));
      LOG(DEBUG) << "Node= " << nd.name() << " #outputs" << tout.size();
      types.insert(types.end(), tout.begin(), tout.end());
    }
  }
  return types;
}

tensorflow::Status inferShapes(const tensorflow::GraphDef& graph_def,
                               const std::vector<std::string>& output_names,
                               ShapeMap& shapes) {
  tensorflow::Graph g(OpRegistry::Global());
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      tensorflow::GraphConstructorOptions(), graph_def, &g));
  std::vector<tensorflow::Node*> POnodes;
  tensorflow::GetPostOrder(g, &POnodes);
  tensorflow::ShapeRefiner refiner(graph_def.versions().producer(),
                                   OpRegistry::Global());
  for (auto n = POnodes.rbegin(); n != POnodes.rend(); ++n) {
    TF_CHECK_OK(refiner.AddNode(*n));
  }

  auto shape2PTS = [](tensorflow::shape_inference::InferenceContext* ic,
                      const tensorflow::shape_inference::ShapeHandle& sh)
      -> tensorflow::PartialTensorShape {
    std::vector<int64> dims;
    int64 rank = ic->Rank(sh);
    for (int64 i = 0; i < rank; i++) {
      auto dh = ic->Dim(sh, i);
      dims.push_back(ic->Value(dh));
    }
    return tensorflow::PartialTensorShape(dims);
  };
  for (const auto& n : POnodes) {
    auto ic = refiner.GetContext(n);
    if (ic) {
      int nOuts = ic->num_outputs();
      auto types = getTypes(n->op_def(), n->def(), false);
      std::vector<
          std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>
          SAT;
      for (int i = 0; i < nOuts; i++) {
        auto PTS = shape2PTS(ic, ic->output(i));
        SAT.push_back({PTS, types.at(i)});
      }
      shapes[n->name()] = SAT;
    } else {
      LOG(WARNING) << "Node " << n->name() << " doesn't have InferenceContext!";
    }
  }
  return tensorflow::Status::OK();
}
}  // namespace trt
}  // namespace tensorflow
