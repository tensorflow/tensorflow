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

#include "tensorflow/python/grappler/model_analyzer.h"

#include <iomanip>
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

ModelAnalyzer::ModelAnalyzer(const GrapplerItem& item) : item_(item) {}

Status ModelAnalyzer::GenerateReport(std::ostream& os) {
  GraphProperties properties(item_);
  TF_RETURN_IF_ERROR(properties.InferStatically());

  for (const auto& node : item_.MainOpsFanin()) {
    PrintNodeInfo(node, properties, os);
  }
  for (const auto& node : item_.EnqueueOpsFanin()) {
    PrintNodeInfo(node, properties, os);
  }

  return Status::OK();
}

void ModelAnalyzer::PrintNodeInfo(const NodeDef* node,
                                  const GraphProperties& properties,
                                  std::ostream& os) const {
  os << node->name() << " [" << node->op() << "]" << std::endl;
  if (properties.HasOutputProperties(node->name())) {
    std::vector<OpInfo::TensorProperties> props =
        properties.GetOutputProperties(node->name());
    for (int i = 0; i < props.size(); ++i) {
      const OpInfo::TensorProperties& prop = props[i];
      os << "\t"
         << "output " << i << " (" << DataTypeString(prop.dtype())
         << ") has shape ";
      if (prop.shape().unknown_rank()) {
        os << "?";
      } else {
        os << "[";
        for (int i = 0; i < prop.shape().dim_size(); ++i) {
          if (i > 0) {
            os << ", ";
          }
          if (prop.shape().dim(i).size() >= 0) {
            // Print the actual dimension.
            os << prop.shape().dim(i).size();
          } else if (prop.shape().dim(i).size() == -1) {
            // We don't know anything about the dimension.
            os << "?";
          } else {
            // Symbolic dimension.
            os << "x" << -prop.shape().dim(i).size();
          }
        }
        os << "]";
      }
      os << std::endl;
    }
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
