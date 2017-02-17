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

package tensorflow

func Placeholder(g *Graph, name string, dt DataType) (Output, error) {
	op, err := g.AddOperation(OpSpec{
		Type: "Placeholder",
		Name: name,
		Attrs: map[string]interface{}{
			"dtype": dt,
		},
	})
	return op.Output(0), err
}

func Const(g *Graph, name string, value interface{}) (Output, error) {
	t, ok := value.(*Tensor)
	if !ok {
		var err error
		if t, err = NewTensor(value); err != nil {
			return Output{}, err
		}
	}
	op, err := g.AddOperation(OpSpec{
		Type: "Const",
		Name: name,
		Attrs: map[string]interface{}{
			"dtype": t.DataType(),
			"value": t,
		},
	})
	return op.Output(0), err
}

func Neg(g *Graph, name string, port Output) (Output, error) {
	op, err := g.AddOperation(OpSpec{
		Type:  "Neg",
		Name:  name,
		Input: []Input{port},
	})
	return op.Output(0), err
}

func Add(g *Graph, name string, x, y Output) (Output, error) {
	op, err := g.AddOperation(OpSpec{
		Type:  "Add",
		Name:  name,
		Input: []Input{x, y},
	})
	return op.Output(0), err
}
