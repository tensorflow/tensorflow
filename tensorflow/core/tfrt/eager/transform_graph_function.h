/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_TRANSFORM_GRAPH_FUNCTION_H_
#define TENSORFLOW_CORE_TFRT_EAGER_TRANSFORM_GRAPH_FUNCTION_H_

#include <memory>

#include "tensorflow/core/common_runtime/function_body.h"
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tfrt {
class Device;
}  // namespace tfrt

namespace tensorflow {

class EagerContext;
class FunctionDef;
class FunctionLibraryDefinition;

// Run placer.
// When `enable_grappler` is true, also run grappler passes over
// the input function, which might add some entries to `func_lib_def`.
//
// TODO(tfrt-devs): Consider passing in a more expressive compiler options
// object such as TFRTCompilerOptions instead of `enable_grappler`, for caller
// to configure graph transformation behavior, such as the more granular options
// in RewriterConfig proto and even individual grappler pass options like
// grappler::ArithmeticOptimizerOptions.
Status TransformGraphFunction(const std::string& func_name,
                              const FunctionDef& fdef,
                              const std::string& device_name,
                              const tensorflow::DeviceSet& device_set,
                              EagerContext* eager_ctx, bool enable_grappler,
                              std::unique_ptr<FunctionBody>* fbody,
                              std::unique_ptr<Graph> graph,
                              tfrt::ArrayRef<const tfrt::Device*> input_devices,
                              FunctionLibraryDefinition* func_lib_def);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_EAGER_TRANSFORM_GRAPH_FUNCTION_H_
