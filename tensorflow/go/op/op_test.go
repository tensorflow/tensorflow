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

// Tests for the generated code of some operations.

package op

import (
	"strings"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestPlaceholder(t *testing.T) {
	s := NewScope()
	Placeholder(s.SubScope("x"), tf.Float, PlaceholderShape(tf.MakeShape(-1, 10)))
	Placeholder(s.SubScope("y"), tf.Float, PlaceholderShape(tf.ScalarShape()))
	Placeholder(s.SubScope("z"), tf.Float, PlaceholderShape(tf.Shape{}))
	if _, err := s.Finalize(); err != nil {
		t.Fatal(err)
	}
}

func TestAddOperationFailure(t *testing.T) {
	// Inspired from https://github.com/tensorflow/tensorflow/issues/9931
	s := NewScope()

	resize := ResizeArea(s, Placeholder(s, tf.Float), Const(s, []int64{80, 80}))
	if err := s.Err(); err == nil {
		t.Fatal("ResizeArea expects an int32 Tensor for size, should fail when an int64 is provided")
	}
	// And any use of resize should panic with an error message more informative than SIGSEGV
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		s, ok := r.(string)
		if ok && strings.Contains(s, "see Scope.Err() for details") {
			return
		}
		t.Errorf("Expected panic string to Scope.Err(), found %T: %q", r, r)
	}()
	_ = resize.Shape()
	t.Errorf("resize.Shape() should have paniced since the underlying Operation was not created")
}

func TestShapeAttribute(t *testing.T) {
	s := NewScope()
	x := Placeholder(s.SubScope("x"), tf.Int32, PlaceholderShape(tf.MakeShape(1)))
	y := Placeholder(s.SubScope("y"), tf.Int32, PlaceholderShape(tf.Shape{}))
	z := Add(s, x, y)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}

	value, err := tf.NewTensor([]int32{7})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[tf.Output]*tf.Tensor{
		x: value,
		y: value,
	}
	fetched, err := sess.Run(feeds, []tf.Output{z}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(fetched), 1; got != want {
		t.Fatalf("Fetched %d tensors, expected %d", got, want)
	}
	if got, want := fetched[0].Value().([]int32), []int32{14}; len(got) != len(want) || len(got) != 1 || got[0] != want[0] {
		t.Fatalf("Got %v, want %v", got, want)
	}
}

func TestDataset(t *testing.T) {
	var (
		s = NewScope()

		// The use of a non-scalar here is inspired by
		// https://github.com/tensorflow/tensorflow/issues/14891
		c       = Const(s, []int32{21718, 31415})
		types   = []tf.DataType{c.DataType()}
		shapes  = []tf.Shape{c.Shape()}
		dataset = TensorDataset(s, []tf.Output{c}, shapes)

		iterator = Iterator(s, "", "", types, shapes)
		next     = IteratorGetNext(s, iterator, types, shapes)
		init     = MakeIterator(s, dataset, iterator)
	)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sess.Run(nil, nil, []*tf.Operation{init}); err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, next, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := results[0].Value().([]int32)
	if len(got) != 2 || got[0] != 21718 || got[1] != 31415 {
		t.Errorf("Got %v, want {21718, 31415}", got)
	}
	if _, err := sess.Run(nil, next, nil); err == nil {
		t.Errorf("Expected sess.Run() to fail since the iterator should have reached the end of the dataset")
	}
}
