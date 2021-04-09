/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func hasOperations(g *Graph, ops ...string) error {
	var missing []string
	for _, op := range ops {
		if g.Operation(op) == nil {
			missing = append(missing, op)
		}
	}
	if len(missing) != 0 {
		return fmt.Errorf("Graph does not have the operations %v", missing)
	}

	inList := map[string]bool{}
	for _, op := range g.Operations() {
		inList[op.Name()] = true
	}

	for _, op := range ops {
		if !inList[op] {
			missing = append(missing, op)
		}
	}

	if len(missing) != 0 {
		return fmt.Errorf("Operations %v are missing from graph.Operations()", missing)
	}

	return nil
}

func TestGraphWriteToAndImport(t *testing.T) {
	// Construct a graph
	g := NewGraph()
	v, err := NewTensor(int64(1))
	if err != nil {
		t.Fatal(err)
	}
	input, err := Placeholder(g, "input", v.DataType())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Neg(g, "neg", input); err != nil {
		t.Fatal(err)
	}

	// Serialize the graph
	buf := new(bytes.Buffer)
	if _, err := g.WriteTo(buf); err != nil {
		t.Fatal(err)
	}

	// Import it into the same graph, with a prefix
	if err := g.Import(buf.Bytes(), "imported"); err != nil {
		t.Error(err)
	}
	if err := hasOperations(g, "input", "neg", "imported/input", "imported/neg"); err != nil {
		t.Error(err)
	}
}

func TestGraphInputMapping(t *testing.T) {
	// Construct a graph
	g := NewGraph()
	v, err := NewTensor(int64(1))
	if err != nil {
		t.Fatal(err)
	}
	input, err := Placeholder(g, "input", v.DataType())
	if err != nil {
		t.Fatal(err)
	}
	neg, err := Neg(g, "neg", input)
	if err != nil {
		t.Fatal(err)
	}

	// Serialize the graph
	buf := new(bytes.Buffer)
	if _, err := g.WriteTo(buf); err != nil {
		t.Fatal(err)
	}

	g = NewGraph()
	v, err = NewTensor(int64(1))
	if err != nil {
		t.Fatal(err)
	}

	replacement, err := Placeholder(g, "replacement", v.DataType())
	if err != nil {
		t.Fatal(err)
	}

	options := GraphImportOptions{
		Prefix: "imported",
	}
	options.AddInputMapping("input", 0, replacement)
	// Import it into the same graph, with a prefix and replacement
	if err := g.ImportWithOptions(buf.Bytes(), options); err != nil {
		t.Error(err)
	}
	if err := hasOperations(g, "replacement", "imported/neg"); err != nil {
		t.Error(err)
	}

	sess, err := NewSession(g, nil)
	if err != nil {
		t.Fatal(err)
	}

	neg = g.Operation("imported/neg").Output(0)

	outputs, err := sess.Run(
		map[Output]*Tensor{replacement: v},
		[]Output{neg},
		nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 1 {
		t.Fatal(len(outputs))
	}
	if outputs[0].Value().(int64) != -1 {
		t.Fatalf("Got %v, wanted int64 -1", outputs[0].Value())
	}
}

func TestGraphAddGradients(t *testing.T) {
	g := NewGraph()
	x1, err := Placeholder(g, "x1", Float)
	if err != nil {
		t.Fatal(err)
	}
	x2, err := Placeholder(g, "x2", Float)
	if err != nil {
		t.Fatal(err)
	}
	op0, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y0",
		Input: []Input{x1},
	})
	if err != nil {
		t.Fatal(err)
	}
	y0 := op0.Output(0)
	op1, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y1",
		Input: []Input{y0},
	})
	if err != nil {
		t.Fatal(err)
	}
	y1 := op1.Output(0)
	op2, err := g.AddOperation(OpSpec{
		Type:  "AddN",
		Input: []Input{OutputList([]Output{y0, x2})},
	})
	if err != nil {
		t.Fatal(err)
	}
	y2 := op2.Output(0)

	grads0, err := g.AddGradients("", []Output{y1}, []Output{x1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(grads0) != 1 {
		t.Fatal(len(grads0))
	}
	if grads0[0].DataType() != Float {
		t.Fatalf("Got DataType %v, wanted %v", grads0[0].DataType(), Float)
	}

	grads1, err := g.AddGradients("", []Output{y2}, []Output{x1, x2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(grads1) != 2 {
		t.Fatal(len(grads1))
	}
	if grads1[0].DataType() != Float {
		t.Fatalf("Got DataType %v, wanted %v", grads1[0].DataType(), Float)
	}
	if grads1[1].DataType() != Float {
		t.Fatalf("Got DataType %v, wanted %v", grads1[1].DataType(), Float)
	}

	sess, err := NewSession(g, nil)
	if err != nil {
		t.Fatal(err)
	}

	c1, _ := NewTensor(float32(3.0))
	c2, _ := NewTensor(float32(2.0))
	outputs, err := sess.Run(
		map[Output]*Tensor{x1: c1, x2: c2},
		[]Output{grads0[0], grads1[0], grads1[1]},
		nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(outputs) != 3 {
		t.Fatal(len(outputs))
	}
	if outputs[0].Value().(float32) != 108.0 {
		t.Fatalf("Got %v, wanted float 108.0", outputs[0].Value())
	}
	if outputs[1].Value().(float32) != 6.0 {
		t.Fatalf("Got %v, wanted float 6.0", outputs[1].Value())
	}
	if outputs[2].Value().(float32) != 1.0 {
		t.Fatalf("Got %v, wanted float 1.0", outputs[2].Value())
	}
}

func TestGraphAddGradientsSums(t *testing.T) {
	g := NewGraph()
	x, err := Placeholder(g, "x", Float)
	if err != nil {
		t.Fatal(err)
	}
	op0, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y0",
		Input: []Input{x},
	})
	if err != nil {
		t.Fatal(err)
	}
	y0 := op0.Output(0)
	op1, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y1",
		Input: []Input{y0},
	})
	y1 := op1.Output(0)

	grad, err := g.AddGradients("", []Output{y0, y1}, []Output{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(grad) != 1 {
		t.Fatal(len(grad))
	}
	if grad[0].DataType() != Float {
		t.Fatalf("Got DataType %v, wanted %v", grad[0].DataType(), Float)
	}

	sess, err := NewSession(g, nil)
	if err != nil {
		t.Fatal(err)
	}

	c, _ := NewTensor(float32(3.0))
	outputs, err := sess.Run(
		map[Output]*Tensor{x: c},
		[]Output{grad[0]},
		nil)
	if err != nil {
		t.Fatal(err)
	}
	if outputs[0].Value().(float32) != 114.0 {
		t.Fatalf("Got %v, wanted float 114.0", outputs[0].Value())
	}
}

func TestGraphAddGradientsWithInitialValues(t *testing.T) {
	g := NewGraph()
	x, err := Placeholder(g, "x", Float)
	op0, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y0",
		Input: []Input{x},
	})
	if err != nil {
		t.Fatal(err)
	}
	y0 := op0.Output(0)
	op1, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y1",
		Input: []Input{y0},
	})
	if err != nil {
		t.Fatal(err)
	}
	y1 := op1.Output(0)

	grads0, err := g.AddGradients("", []Output{y1}, []Output{y0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(grads0) != 1 {
		t.Fatal(len(grads0))
	}
	if grads0[0].DataType() != Float {
		t.Fatalf("Got DataType %v, wanted %v", grads0[0].DataType(), Float)
	}

	grads1, err := g.AddGradients("", []Output{y0}, []Output{x}, []Output{grads0[0]})
	if err != nil {
		t.Fatal(err)
	}
	if len(grads1) != 1 {
		t.Fatal(len(grads1))
	}
	if grads1[0].DataType() != Float {
		t.Fatalf("Got DataType %v, wanted %v", grads1[0].DataType(), Float)
	}

	sess, err := NewSession(g, nil)
	if err != nil {
		t.Fatal(err)
	}

	c, _ := NewTensor(float32(3.0))
	outputs, err := sess.Run(
		map[Output]*Tensor{x: c},
		[]Output{grads1[0]},
		nil)
	if err != nil {
		t.Fatal(err)
	}
	if outputs[0].Value().(float32) != 108.0 {
		t.Fatalf("Got %v, wanted float 108.0", outputs[0].Value())
	}
}

func TestGraphValidateGradientsNames(t *testing.T) {
	g := NewGraph()
	x, err := Placeholder(g, "x", Float)
	if err != nil {
		t.Fatal(err)
	}
	op0, err := g.AddOperation(OpSpec{
		Type:  "Square",
		Name:  "y0",
		Input: []Input{x},
	})
	if err != nil {
		t.Fatal(err)
	}
	y0 := op0.Output(0)

	grads0, err := g.AddGradients("", []Output{y0}, []Output{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(grads0[0].Op.Name(), "gradients/") {
		t.Fatalf("Got name %v, wanted started with gradients/", grads0[0].Op.Name())
	}

	grads1, err := g.AddGradients("", []Output{y0}, []Output{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(grads1[0].Op.Name(), "gradients_1/") {
		t.Fatalf("Got name %v, wanted started with gradients_1/", grads1[0].Op.Name())
	}

	grads2, err := g.AddGradients("more_gradients", []Output{y0}, []Output{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(grads2[0].Op.Name(), "more_gradients/") {
		t.Fatalf("Got name %v, wanted started with more_gradients/", grads2[0].Op.Name())
	}

	grads3, err := g.AddGradients("even_more_gradients", []Output{y0}, []Output{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(grads3[0].Op.Name(), "even_more_gradients/") {
		t.Fatalf("Got name %v, wanted started with even_more_gradients/", grads3[0].Op.Name())
	}

	_, err = g.AddGradients("even_more_gradients", []Output{y0}, []Output{x}, nil)
	if err == nil {
		t.Error("AddGradients should have failed if gradients name is already existing")
	}
}
