/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COSTMODEL_MANAGER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COSTMODEL_MANAGER_H_

#include <unordered_map>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"

namespace tensorflow {

// Used to manage all the cost models for a session.
class CostModelManager {
 public:
  ~CostModelManager();

  typedef std::unordered_map<const Graph*, CostModel*> CostModelMap;
  typedef CostModelMap::iterator CostModelMapIter;

  gtl::iterator_range<CostModelMapIter> CostModels() {
    mutex_lock l(mu_);
    return gtl::make_range(cost_models_.begin(), cost_models_.end());
  }

  CostModel* FindOrCreateCostModel(const Graph* graph);
  Status BuildCostGraphDef(const Graph* graph, CostGraphDef* cost_graph);

 private:
  mutex mu_;
  CostModelMap cost_models_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COSTMODEL_MANAGER_H_
