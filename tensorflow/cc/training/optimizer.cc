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

#include "tensorflow/cc/training/optimizer.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow {
namespace {

    using namespace tensorflow::ops;

    // processor abstract class
    class OptimizableVariable {
        public:
            virtual ~OptimizableVariable() {}
            virtual Output UpdateOp(const Scope& scope, const Optimizer& optimizer, const Output& grad) = 0;
            OptimizableVariable(const Output& v) : variable_(v) {}
            Output variable() { return variable_; }
        protected:
            Output variable_;
    };

    // processor for a Variable
    class RefVariableProcessor: public OptimizableVariable {
        public:
            ~RefVariableProcessor() {}
            RefVariableProcessor(const Output& v) : OptimizableVariable(v) {}
            Output UpdateOp(const Scope& scope, const Optimizer& optimizer, const Output& grad) override {
                return optimizer.ApplyDense(scope, grad, variable_);
            }
    };

    // return the processor regarding the type of the variable
    OptimizableVariable* GetProcessor(const Scope& scope, const Output& variable) {
        if(::tensorflow::grappler::IsVariable(variable.node()->def())) {
            return new RefVariableProcessor(variable);
        } else {
            scope.UpdateStatus(Status(::tensorflow::error::Code::INVALID_ARGUMENT, 
                "No processor yet for this kind of variable: " + variable.node()->def().op()));
        }
        return nullptr;
    }

} // namespace

const string Optimizer::GRADIENTS_NAME = "Gradients";

void Optimizer::ComputeGradients(const Scope& scope,
                                 const std::vector<Output>& loss,
                                 const std::vector<Output>& var_list,
                                 const std::vector<Output>& grad_loss,
                                 GradAndVar* grads_and_vars) {

    // if a ComputeGradients overload has been called, the scope is already GRADIENTS_NAME
    Scope scope_gradient = scope;
    if (scope.Name() != GRADIENTS_NAME) {
        scope_gradient = scope.NewSubScope(GRADIENTS_NAME);
    }

    // add the gradients node to the graph
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope_gradient, loss, var_list, grad_loss, &grad_outputs));
    
    // create a vector of pair (grad, var)
    for (int i = 0; i < var_list.size(); ++i) {
        if (::tensorflow::grappler::IsVariable(var_list[i].node()->def())) {
            grads_and_vars->push_back(std::make_tuple(grad_outputs[i], var_list[i]));
        }
    }

    // if we don't have the same number of pair (grad, var) as the number of vars
    // it means that at least one var from var_list is not a variable
    if (var_list.size() != grads_and_vars->size()) {
        scope.UpdateStatus(Status(::tensorflow::error::Code::INVALID_ARGUMENT, 
            "You are trying to compute the gradients of non Variable Output."));
    }

}

std::vector<Output> Optimizer::ApplyGradients(const Scope& scope,
                                              const GradAndVar& grads_and_vars) {
    // we'll return a list of update ops
    // TODO(theflofly): make a Group op and return one output
    std::vector<Output> update_ops;

    if (grads_and_vars.empty()) {
        scope.UpdateStatus(Status(::tensorflow::error::Code::INVALID_ARGUMENT, "No variable to update provided."));
    }

    Scope scope_optimizer = scope.NewSubScope(this->name_);

    // each instance can prepare itself using their own logic
    this->Prepare(scope_optimizer);

    // retrieve the processor for each (gradient, var) tuple
    for (int i = 0; i < grads_and_vars.size(); i++) {
    
        Output grad = std::get<0>(grads_and_vars[i]);
        Output var = std::get<1>(grads_and_vars[i]);
        OptimizableVariable *processor = GetProcessor(scope_optimizer, var);

        // no processor for this kind of variable, the scope.status() contains more details
        if (processor == nullptr) {
            return update_ops;
        }

        update_ops.push_back(processor->UpdateOp(scope_optimizer, *this, grad));
        delete processor;

    }

    return update_ops;
}

void Optimizer::ComputeGradients(const Scope& scope,
                                 const std::vector<Output>& loss,
                                 const std::vector<Output>& var_list,
                                 GradAndVar* grads_and_vars) {
    
    Scope scope_gradient = scope.NewSubScope(GRADIENTS_NAME);

    // fill grad_loss with 'OnesLike' for all shapes in 'loss'
    std::vector<Output> grad_loss;
    grad_loss.reserve(loss.size());

    for (const Output& loss_output : loss) {
        grad_loss.emplace_back(ops::OnesLike(scope_gradient, loss_output));
    }

    ComputeGradients(scope_gradient, loss, var_list, grad_loss, grads_and_vars);
}

std::vector<Output> Optimizer::Minimize(const Scope& scope,
                                        const std::vector<Output>& loss,
                                        const std::vector<Output>& var_list) {

    GradAndVar grad_and_var;
    ComputeGradients(scope, loss, var_list, &grad_and_var);
    return ApplyGradients(scope, grad_and_var);

}

std::vector<Output> Optimizer::Minimize(const Scope& scope,
                                        const std::vector<Output>& loss) {
    
    // retrieve all the variables from the graph
    std::vector<Output> var_list;

    for (Node* node : scope.graph()->nodes()) {
        if(::tensorflow::grappler::IsVariable(node->def())) {
            var_list.push_back(Output{node});
        }
    }

    GradAndVar grad_and_var;
    ComputeGradients(scope, loss, var_list, &grad_and_var);
    return ApplyGradients(scope, grad_and_var);

}

} /// namespace tensorflow
