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

#ifndef TENSORFLOW_PYTHON_GRAPPLER_MODEL_ANALYZER_H_
#define TENSORFLOW_PYTHON_GRAPPLER_MODEL_ANALYZER_H_

#include <iostream>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace grappler {
struct GrapplerItem;
class GraphProperties;

// Generate a report detailing how much information is known statically for most
// operations in the model, including output data types and output shapes.
class ModelAnalyzer {
 public:
  explicit ModelAnalyzer(const GrapplerItem& item);
  Status GenerateReport(bool debug, bool assume_valid_feeds, std::ostream& os);

 private:
  void PrintNodeInfo(const NodeDef* node, const GraphProperties& properties,
                     bool debug, std::ostream& os) const;

  const GrapplerItem& item_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_PYTHON_GRAPPLER_MODEL_ANALYZER_H_
