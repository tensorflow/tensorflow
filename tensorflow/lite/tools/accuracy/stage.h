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

#ifndef TENSORFLOW_LITE_TOOLS_ACCURACY_STAGE_H_
#define TENSORFLOW_LITE_TOOLS_ACCURACY_STAGE_H_

#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {
namespace metrics {

// A stage in an evaluation pipeline.
// Each stage adds a subgraph to the pipeline. Stages can be chained
// together.
class Stage {
 public:
  Stage() = default;
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;

  Stage(const Stage&&) = delete;
  Stage& operator=(const Stage&&) = delete;

  // Adds a subgraph to given scope that takes in `input` as a parameter.
  virtual void AddToGraph(const Scope& scope, const Input& input) = 0;
  virtual ~Stage() {}

  // The name of the stage.
  // Can be used by derived classes for naming the subscope for the stage
  // graph.
  virtual string name() const = 0;

  // The name of the output for the stage.
  virtual string output_name() const = 0;

  const ::tensorflow::Output& Output() const { return stage_output_; }

 protected:
  ::tensorflow::Output stage_output_;
};
}  //  namespace metrics
}  //  namespace tensorflow

#endif  // TENSORFLOW_LITE_TOOLS_ACCURACY_STAGE_H_
