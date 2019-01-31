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

// An optimization pass that performs mixed-precision optimization

#ifndef TENSORFLOW_CORE_GRAPH_AUTO_MIXED_PRECISION_H_
#define TENSORFLOW_CORE_GRAPH_AUTO_MIXED_PRECISION_H_

namespace tensorflow {

class Graph;

// Returns true if and only if 'g' is mutated.
extern bool RunAutoMixedPrecision(Graph* g);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_AUTO_MIXED_PRECISION_H_
