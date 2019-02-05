////////// SWIG INCLUDE //////////

%include "net/proto/swig/protofunc.swig"

#ifndef ABSL_MUST_USE_RESULT
#error Use this file only as a %include or %import after google.swig.
#endif

%{
#include "third_party/tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
%}

PROTO_INPUT(tensorflow::decision_trees::DecisionTree, decision_tree);
PROTO_IN_OUT(tensorflow::decision_trees::DecisionTree, decision_tree);
