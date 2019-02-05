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

#ifndef TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
#define TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// This analyzes a TensorFlow graph to identify nodes which may have partially
// dead inputs (i.e. these nodes may have some dead inputs and some alive
// inputs).
//
// For example, the ADD node in the following graph
//
//      V0  PRED0    V1  PRED1
//       |    |       |    |
//       v    v       v    v
//       SWITCH       SWITCH
//          |            |
//          +---+   + ---+
//              |   |
//              v   v
//               ADD
//
// can have its inputs independently dead or alive based on the runtime values
// of PRED0 and PRED1.
//
// It is tempting to call this a liveness analysis but I avoided that because
// "liveness" already has other connotations.
class DeadnessAnalysis {
 public:
  // Returns true if `node` may have some live inputs and some dead inputs.
  //
  // This is a conservatively correct routine -- if it returns false then `node`
  // is guaranteed to not have inputs with mismatching liveness, but not the
  // converse.
  //
  // REQUIRES: node is not a Merge operation.
  virtual bool HasInputsWithMismatchingDeadness(const Node& node) = 0;

  // Prints out the internal state of this instance.  For debugging purposes
  // only.
  virtual void Print() const = 0;
  virtual ~DeadnessAnalysis();

  // Run the deadness analysis over `graph` and returns an error or a populated
  // instance of DeadnessAnalysis in `result`.
  static Status Run(const Graph& graph,
                    std::unique_ptr<DeadnessAnalysis>* result);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
