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

#ifndef TENSORFLOW_CORE_GRAPPLER_INPUTS_TRIVIAL_TEST_GRAPH_INPUT_YIELDER_H_
#define TENSORFLOW_CORE_GRAPPLER_INPUTS_TRIVIAL_TEST_GRAPH_INPUT_YIELDER_H_

#include <string>
#include <vector>
#include "tensorflow/core/grappler/inputs/input_yielder.h"

namespace tensorflow {
namespace grappler {

class Cluster;
struct GrapplerItem;

class TrivialTestGraphInputYielder : public InputYielder {
 public:
  TrivialTestGraphInputYielder(int num_stages, int width, int tensor_size,
                               bool insert_queue,
                               const std::vector<string>& device_names);
  bool NextItem(GrapplerItem* item) override;

 private:
  const int num_stages_;
  const int width_;
  const int tensor_size_;
  const bool insert_queue_;
  std::vector<string> device_names_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_INPUTS_TRIVIAL_TEST_GRAPH_INPUT_YIELDER_H_
