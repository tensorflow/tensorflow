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

package op

import (
	"fmt"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestScopeSubScope(t *testing.T) {
	var (
		root  = NewScope()
		sub1  = root.SubScope("x")
		sub2  = root.SubScope("x")
		sub1a = sub1.SubScope("y")
		sub2a = sub2.SubScope("y")
	)
	testdata := []struct {
		scope *Scope
		name  string
	}{
		{root, "Const"},
		{sub1, "x/Const"},
		{sub1a, "x/y/Const"},
		{sub2, "x_1/Const"},
		{sub2a, "x_1/y/Const"},
	}
	for _, test := range testdata {
		c := Const(test.scope, int64(1))
		if err := test.scope.Err(); err != nil {
			t.Fatalf("%q: %v", test.name, err)
		}
		if got := c.Op.Name(); got != test.name {
			t.Errorf("%q: Got %q", test.name, got)
		}
	}
}

func TestScopeSubScopeErrors(t *testing.T) {
	var (
		root = NewScope()
		sub  = root.SubScope("x")
	)
	// Error on the root, even after sub has been created should be propagated.
	// Force an error by creating a Const which has a type that does not
	// translate to the TensorFlow type system.
	Const(root, int(1))
	if err := root.Err(); err == nil {
		t.Fatal("Expected error")
	}
	if err := sub.Err(); err == nil {
		t.Errorf("Root scope had error [%v], but sub-scope did not", root.Err())
	}
}

func TestControlDependencies(t *testing.T) {
	var (
		s        = NewScope()
		zero     = Const(s.SubScope("zero"), int32(0))
		one      = Const(s.SubScope("one"), int32(1))
		variable = VarHandleOp(s, tf.Int32, tf.ScalarShape())
		init     = AssignVariableOp(s, variable, zero)
		update   = AssignAddVariableOp(s, variable, one)
		readDeps = []*tf.Operation{update}
	)
	// We intend for `read` to have a control dependency on `update`.
	s = s.WithControlDependencies(readDeps...)
	// Ensure that Scope.WithControlDependencies makes a copy of the underlying
	// array, rather than just holding a slice reference to the same user-supplied
	// underlying array.  If the copy is correctly performed, overwriting
	// readDeps[0] should have no effect on control dependencies for `read`.
	readDeps[0] = init
	read := ReadVariableOp(s, variable, tf.Int32)

	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = sess.Run(nil, nil, []*tf.Operation{init}); err != nil {
		t.Fatal(err)
	}
	// Without the control dependency, the read operation may not see the
	// update.
	for i := int32(0); i < 10; i++ {
		out, err := sess.Run(nil, []tf.Output{read}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := out[0].Value().(int32), i+1; got != want {
			t.Errorf("Got %d, want %d", got, want)
		}
	}
}

func TestDevice(t *testing.T) {
	s := NewScope()
	matrix := Const(s, [][]float32{{3.0}})
	s = s.WithDevice("/device:GPU:0")
	square := MatMul(s.SubScope("square"), matrix, matrix)
	s = s.WithDevice("")
	cube := MatMul(s.SubScope("cube"), square, matrix)
	if got, want := square.Op.Device(), "/device:GPU:0"; got != want {
		t.Errorf("Got %q, want %q", got, want)
	}
	if got, want := cube.Op.Device(), ""; got != want {
		t.Errorf("Got %q, want %q", got, want)
	}
}

func TestScopeFinalize(t *testing.T) {
	var (
		root = NewScope()
		sub1 = root.SubScope("x")
		sub2 = sub1.SubScope("y")
	)
	if _, err := sub1.Finalize(); err != nil {
		t.Fatal(err)
	}
	if err := root.Err(); err == nil {
		t.Error("Root scope's Err() should be non-nil once Finalize has been called")
	}
	if err := sub2.Err(); err == nil {
		t.Error("Sub scope's Err() should be non-nil once Finalize has been called")
	}
}

func TestMultipleGeneratedOps(t *testing.T) {
	s := NewScope()
	Placeholder(s.SubScope("x"), tf.Float)
	Placeholder(s.SubScope("y"), tf.Float)
	if _, err := s.Finalize(); err != nil {
		t.Fatal(err)
	}
}

func TestScopeWithGraph(t *testing.T) {
	s1 := NewScope()
	Const(s1, "hello")
	graph, err := s1.Finalize()
	if err != nil {
		t.Fatal(err)
	}

	s2 := NewScopeWithGraph(graph)
	Const(s2.SubScope("addition"), "world")
	if err := s2.Err(); err != nil {
		t.Fatal(err)
	}
}

func Example() {
	// This example creates a Graph that multiplies a constant matrix with
	// a matrix to be provided during graph execution (via
	// tensorflow.Session).
	s := NewScope()
	input := Placeholder(s, tf.Float) // Matrix to be provided to Session.Run
	output := MatMul(s,
		Const(s, [][]float32{{10}, {20}}), // Constant 2x1 matrix
		input,
		MatMulTransposeB(true))
	if s.Err() != nil {
		panic(s.Err())
	}
	// Shape of the product: The number of rows is fixed by m1, but the
	// number of columns will depend on m2, which is unknown.
	fmt.Println(output.Shape())
	// Output: [2, ?]
}

func ExampleScope_SubScope() {
	var (
		s  = NewScope()
		c1 = Const(s.SubScope("x"), int64(1))
		c2 = Const(s.SubScope("x"), int64(1))
	)
	if s.Err() != nil {
		panic(s.Err())
	}
	fmt.Println(c1.Op.Name(), c2.Op.Name())
	// Output: x/Const x_1/Const
}
