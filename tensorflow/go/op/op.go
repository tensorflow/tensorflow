// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package op defines functions for adding TensorFlow operations to a Graph.
//
// Functions for adding an operation to a graph take a Scope object as the
// first argument. The Scope object encapsulates a graph and a set of
// properties (such as a name prefix) for all operations being added
// to the graph.
//
// WARNING: The API in this package has not been finalized and can
// change without notice.
package op

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Const adds an operation to graph that produces value as output.
func Const(scope *Scope, value interface{}) (tf.Output, error) {
	if t, ok := value.(*tf.Tensor); ok {
		return makeConst(scope, t)
	}
	t, err := tf.NewTensor(value)
	if err != nil {
		return tf.Output{}, err
	}
	return makeConst(scope, t)
}

func makeConst(scope *Scope, t *tf.Tensor) (tf.Output, error) {
	op, err := scope.Graph().AddOperation(tf.OpSpec{
		Name: scope.opName("Const"),
		Type: "Const",
		Attrs: map[string]interface{}{
			"dtype": t.DataType(),
			"value": t,
		}})
	return op.Output(0), err
}
