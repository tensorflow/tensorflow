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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_

#include <atomic>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace grappler {

class Cluster;
struct GrapplerItem;

// An abstract interface for an algorithm for generating a candidate
// optimization of a GrapplerItem for running on a cluster.
class GraphOptimizer {
 public:
  GraphOptimizer() : is_cancelled_(false) {}
  virtual ~GraphOptimizer() {}

  virtual string name() const = 0;

  // Routine called to allow an algorithm to propose a rewritten graph
  // for the graph, feeds and fetches in "item" to run more efficiently
  // on "cluster".
  // Returns true iff it managed to generate a solution, false otherwise.
  virtual Status Optimize(Cluster* cluster, const GrapplerItem& item,
                          GraphDef* optimized_graph) = 0;

  // Method invoked by the framework so that it can provide feedback
  // on how well the "optimized_graph" (produced as *optimized_graph from a
  // call to Optimize) performed.  Lower "result" scores are better.
  virtual void Feedback(Cluster* cluster, const GrapplerItem& item,
                        const GraphDef& optimized_graph, double result) = 0;

  // Best effort cancellation. Sets is_cancelled to true and requests that the
  // optimizer returns as soon as possible from active calls to Optimize() or
  // FeedBack().
  void Cancel() { is_cancelled_ = true; }

  bool is_cancelled() const { return is_cancelled_; }

 private:
  std::atomic<bool> is_cancelled_;
};

#define GRAPPLER_RETURN_IF_CANCELLED()                                  \
  do {                                                                  \
    if (is_cancelled()) {                                               \
      return errors::DeadlineExceeded(this->name(), " was cancelled."); \
    }                                                                   \
  } while (0)

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_H_
