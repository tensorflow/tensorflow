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
