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

#ifndef TENSORFLOW_GRAPPLER_UTILS_H_
#define TENSORFLOW_GRAPPLER_UTILS_H_

#include <functional>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// A utility class to lookup a node and its outputs by node name.
class NodeMap {
 public:
  explicit NodeMap(GraphDef* graph);
  NodeDef* GetNode(const string& name);
  std::set<NodeDef*> GetOutputs(const string& node_name);
  // This method doesn't record the outputs of the added node; the outputs need
  // to be explictly added by the AddOutput method.
  void AddNode(const string& name, NodeDef* node);
  void AddOutput(const string& node, const string& output);
  void UpdateOutput(const string& node, const string& old_output,
                    const string& new_output);

 private:
  GraphDef* graph_;
  std::unordered_map<string, NodeDef*> nodes_;
  std::unordered_map<string, std::set<NodeDef*>> outputs_;
};

// Return the node name corresponding to 'name' if name is valid, or the empty
// string otherwise.
string NodeName(const string& name);

// Get the trailing position number ":{digits}" (if any) of a node name.
int NodePosition(const string& name);

// Add a prefix to a node name
string AddPrefixToNodeName(const string& name, const string& prefix);

// Executes a 'fn' in the 'thread_pool'. The method waits for the configured
// timeout (in milliseconds) for 'fn' to complete, before returning false.
//
// If returning false, the 'fn' may still continue to execute in the
// thread-pool. It is the responsibility of the caller to reset the thread-pool
// as appropriate.
bool ExecuteWithTimeout(std::function<void()> fn, int64 timeout_in_ms,
                        thread::ThreadPool* thread_pool);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_UTILS_H_
