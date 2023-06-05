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
  // An opaque representation of a predicate.  DeadnessPredicate
  // instances that compare equal via operator== represent predicates
  // that always evaluate to the same value.
  struct DeadnessPredicate {
   public:
    DeadnessPredicate(const DeadnessPredicate&) = default;
    DeadnessPredicate(DeadnessPredicate&&) = default;

    DeadnessPredicate& operator=(const DeadnessPredicate&) = default;
    DeadnessPredicate& operator=(DeadnessPredicate&&) = default;

    bool operator==(const DeadnessPredicate& other) const {
      return other.pred_ == pred_;
    }

    bool operator!=(const DeadnessPredicate& other) const {
      return other.pred_ != pred_;
    }

   private:
    explicit DeadnessPredicate(void* pred) : pred_(pred) {}

    // This is really a Predicate*, but we don't want to expose that
    // implementation detail to our clients.  `pred_` has pointer equality so we
    // can just compare the pointer in operator== and operator!=.
    void* pred_;

    friend class DeadnessAnalysis;
  };

  virtual tsl::StatusOr<DeadnessPredicate> GetPredicateFor(Node* n,
                                                           int oidx) const = 0;

  // Prints out the internal state of this instance.  For debugging purposes
  // only.
  virtual void Print() const = 0;
  virtual ~DeadnessAnalysis();

  string DebugString(DeadnessPredicate predicate) const;

  // Run the deadness analysis over `graph` and returns an error or a populated
  // instance of DeadnessAnalysis in `result`.
  static Status Run(const Graph& graph,
                    std::unique_ptr<DeadnessAnalysis>* result);

 protected:
  static DeadnessPredicate MakeDeadnessPredicate(void* pred) {
    return DeadnessPredicate(pred);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEADNESS_ANALYSIS_H_
