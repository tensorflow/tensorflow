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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_IMPLEMENTATION_SELECTOR_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_IMPLEMENTATION_SELECTOR_H_

#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/function_api_info.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

// Motivation: To achieve the same high level functionality, the underlying
// implementations sometimes are different for various devices where the
// function runs. In order to achieve the correct result and best performance,
// the proper implementation needs to be picked dynamically.
//
// Currently there are two approaches to do this.
// (1) Utilize case op and dynamacically change the branch index.
// (2) Swap function implementation, it will be deprecated.
//
// Idea for approach 1.
// This transformation rewrites the DeviceIndex op with a Const op with value
// of the index of the device the associcated Case op runs.
// Example:
// def plus_one_gpu(x): return x + 1.0
// def plus_one_reference_implementation(x): return x + 1.0
// input = tf.constant(2.0, dtype=tf.float32)
// cpu_fn = lambda:plus_one_reference_implementation(input)
// gpu_fn = lambda:plus_one_gpu(input)
// control_flow_ops.execute_fn_for_device(
//  {"CPU": cpu_fn, "GPU":gpu_fn)}, default_fn=cpu_fn)
//
// Idea for approach 2.
// This transformation replaces function calls by the appropriate function
// definition based on properties of the runtime system. For instance,
// we may choose one implementation over another if we have a GPU with
// enough memory available.
//
// It is a way for the programmer to specify alternative implementations
// of the same functionality in the graph, and let TensorFlow pick the
// most appropriate one at runtime.
//
// For instance, the python code might specify:
// @Defun(tf.float32,
//        api_implements='plus_one',
//        api_preferred_device='GPU')
// def plus_one_gpu(x): return x + 1.0
//
// @Defun(tf.float32,
//        api_implements='plus_one')
// def plus_one_reference_implementation(x): return x + 1.0
// input = tf.constant(2.0, dtype=tf.float32)
//
// z = plus_one_reference_implementation(input)
// z = plus_one_gpu(input)
// print(sess.run(z))
//

// At runtime, we will select either `plus_one_gpu` or
// `plus_one_reference_implementation` based on the availability of the GPU.
//
// Available annotations:
//  - api_implements(string): all functions mapping to the same
//    string can be interchanged. For now, all functions must have the same
//    signature and overloads are not allowed. Defuns within defuns are
//    allowed.
//  - api_preferred_device(string): sets which device is preferred.
class ImplementationSelector : public CustomGraphOptimizer {
 public:
  ImplementationSelector() = default;
  ~ImplementationSelector() override = default;
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }
  string name() const override {
    return "implementation_selector";
  }

  bool UsesFunctionLibrary() const override { return false; }

  // This call is not thread-safe.
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  // Does not take any feedback.
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}

 private:
  Status LoadFunctions(const GraphDef& graph);
  Status MaybeOptimizeFunctionCall(utils::MutableNodeView* node_view) const;

  // Finds all call sites for functions, then replace with the appropriate
  // implementation.
  // There are two ways of calling functions:
  //  1. By specifying an op name as a function name, and
  //  2. Via the functional interface, where the function name appears as an
  //  Attr.
  //
  // There may be multiple call sites for a given function. The function body
  // may call into another function, so a function might have to be duplicated.
  // For simplicity, we do not change function bodies. Also, we do not change
  // gradients.
  Status SelectImplementation(GraphDef* graph) const;

  // Rewrites the DeviceIndex op with a Const op with value of the index of the
  // device the associcated Case op runs.

  // This function first looks up all the DeviceIndex ops.
  // Then for each of these ops, it finds the device of the
  // associated Case op that takes the DeviceIndex op as the input, and
  // caculates the index of the device in the device list of DeviceIndex op.
  // Lastly, it rewrites the DeviceIndex op with a Const op and sets the value
  // to be the index.
  //
  // Example input nodes:
  // node {
  //   name: "x"
  //   op: "DeviceIndex"
  //   device: "/device:CPU:0"
  //   attr {
  //     key: "device_names"
  //     value {
  //       list {
  //         s: "CPU"
  //         s: "TPU_REPLICATED_CORE"
  //         s: "GPU"
  //       }
  //     }
  //   }
  // }
  // node {
  //   name: "case"
  //   op: "Case"
  //   input: "x"
  //   device: "/device:GPU:0"
  //   ...
  // }
  // Example output nodes:
  //
  //  name: "x"
  //  op: "Const"
  //  device: "/device:CPU:0"
  //  attr {
  //    key: "dtype"
  //    value {
  //      type: DT_INT32
  //    }
  //  }
  //  attr {
  //    key: "value"
  //    value {
  //      tensor {
  //        dtype: DT_INT32
  //        int_val: 2
  //      }
  //    }
  //  }
  // node {
  //   name: "case"
  //   op: "Case"
  //   input: "x"
  //   device: "/device:GPU:0"
  //   ...
  // }
  Status SelectDeviceIndex(GraphDef* graph) const;

  std::unique_ptr<FunctionLibraryApiInfo> lib_info_;

  TF_DISALLOW_COPY_AND_ASSIGN(ImplementationSelector);
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_IMPLEMENTATION_SELECTOR_H_
