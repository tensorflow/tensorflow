/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CC_OPTIMIZER_H_
#define THIRD_PARTY_TENSORFLOW_CC_OPTIMIZER_H_

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

typedef std::vector<std::tuple<Output, Output>> GradAndVar;

class Optimizer {
    public:
        Optimizer(const string& name) : name_(name) {} 
        string name() const { return name_; }

        // Note: map the function AddSymbolicGradients, the parameters name have been changed
        // in order to be consistent with the python API (tensorflow/python/training/optimizer.py)
        //
        // Given initial gradients 'grad_loss' (which represent the symbolic partial
        // derivatives of some loss function 'L' w.r.t 'loss'), adds gradient nodes
        // to the graph associated with 'scope', which computes (and return in
        // 'grads_and_vars') the symbolic partial derivatives of 'L' w.r.t 'inputs'.
        // TODO(theflofly): Optional parameters + all options provided by the python counterpart 
        // (gate_gradients, aggregation_method, colocate_gradients_with_ops)
        void ComputeGradients(const Scope& scope,
                              const std::vector<Output>& loss,
                              const std::vector<Output>& var_list,
                              const std::vector<Output>& grad_loss,
                              GradAndVar* grads_and_vars);

        // Same as above but uses 'OnesLike' for all shapes in 'loss' as grad_loss.
        void ComputeGradients(const Scope& scope,
                              const std::vector<Output>& loss,
                              const std::vector<Output>& var_list,
                              GradAndVar* grads_and_vars);
        
        // Applies gradients to variables.
        // Return a vector of operations that update the var using the gradient.
        // TODO(theflofly): Optional parameters + all options provided by the python counterpart (global_step, name)
        std::vector<Output> ApplyGradients(const Scope& scope, 
                                           const GradAndVar& grads_and_vars);

        // Add ops to apply dense gradient to `var`.
        virtual Output ApplyDense(const Scope& scope, 
                                  const Output& grad, 
                                  const Output& var) const = 0;

        // Calls ComputeGradient and ApplyGradient.
        // Return a list of operations that update the Variables using the gradient.
        // TODO(theflofly): Optional parameters + all options provided by the python counterpart 
        // (gate_gradients, aggregation_method, colocate_gradients_with_ops, global_step, name)
        std::vector<Output> Minimize(const Scope& scope,
                                     const std::vector<Output>& loss,
                                     const std::vector<Output>& var_list);

        // same as above but the var_list is created using all the vars from the graph
        std::vector<Output> Minimize(const Scope& scope,
                                     const std::vector<Output>& loss);
        
    protected:
        string name_;
        static const string GRADIENTS_NAME;

        // make the necessary preparation regarding the Optimization used
        virtual void Prepare(const Scope& scope) = 0;
};

} // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_OPTIMIZER_H_