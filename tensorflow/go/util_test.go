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
	b := newOpBuilder(g, "Placeholder", name)
	b.SetAttrType("dtype", dt)
	op, err := b.Build()
	if err != nil {
		return Output{}, err
	}
	return Output{op, 0}, nil
}

func Neg(g *Graph, name string, port Output) (Output, error) {
	b := newOpBuilder(g, "Neg", name)
	b.AddInput(port)
	op, err := b.Build()
	if err != nil {
		return Output{}, err
	}
	return Output{op, 0}, nil
}
