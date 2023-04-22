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
	"fmt"
	"reflect"
	"testing"
)

func createTestGraph(t *testing.T, dt DataType) (*Graph, Output, Output) {
	g := NewGraph()
	inp, err := Placeholder(g, "p1", dt)
	if err != nil {
		t.Fatalf("Placeholder() for %v: %v", dt, err)
	}
	out, err := Neg(g, "neg1", inp)
	if err != nil {
		t.Fatalf("Neg() for %v: %v", dt, err)
	}
	return g, inp, out
}

func TestSessionRunNeg(t *testing.T) {
	var tests = []struct {
		input    interface{}
		expected interface{}
	}{
		{int64(1), int64(-1)},
		{[]float64{-1, -2, 3}, []float64{1, 2, -3}},
		{[][]float32{{1, -2}, {-3, 4}}, [][]float32{{-1, 2}, {3, -4}}},
	}

	for _, test := range tests {
		t.Run(fmt.Sprint(test.input), func(t *testing.T) {
			t1, err := NewTensor(test.input)
			if err != nil {
				t.Fatal(err)
			}
			graph, inp, out := createTestGraph(t, t1.DataType())
			s, err := NewSession(graph, &SessionOptions{})
			if err != nil {
				t.Fatal(err)
			}
			output, err := s.Run(map[Output]*Tensor{inp: t1}, []Output{out}, []*Operation{out.Op})
			if err != nil {
				t.Fatal(err)
			}
			if len(output) != 1 {
				t.Fatalf("got %d outputs, want 1", len(output))
			}
			val := output[0].Value()
			if !reflect.DeepEqual(test.expected, val) {
				t.Errorf("got %v, want %v", val, test.expected)
			}
			if err := s.Close(); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestMultipleInput(t *testing.T) {
	// The inputs to the graph get sorted. This test checks that process works
	// OK and that we still get the right output.
	graph := NewGraph()

	inputs := make([]Output, 20)
	layer2 := make([]Output, len(inputs))
	for i := range inputs {
		in, err := Placeholder(graph, fmt.Sprintf("input%d", i), Int64)
		if err != nil {
			t.Fatal(err)
		}
		inputs[i] = in

		factor, err := Const(graph, fmt.Sprintf("factor%d", i), int64(i+1))
		if err != nil {
			t.Fatal(err)
		}
		l2, err := graph.AddOperation(OpSpec{
			Type: "Mul",
			Name: fmt.Sprintf("Mul%d", i),
			Input: []Input{
				in,
				factor,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		layer2[i] = l2.Output(0)
	}

	fetch, err := graph.AddOperation(OpSpec{
		Type: "AddN",
		Input: []Input{
			OutputList(layer2),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	}()

	feeds := make(map[Output]*Tensor, len(inputs))
	for i, in := range inputs {
		tensor, err := NewTensor(int64(i + 1))
		if err != nil {
			t.Fatal(err)
		}
		feeds[in] = tensor
	}

	output, err := session.Run(
		feeds,
		[]Output{
			fetch.Output(0),
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	var exp int64
	for i := range inputs {
		exp += int64((i + 1) * (i + 1))
	}
	if v := output[0].Value().(int64); v != exp {
		t.Fatalf("expected %d got %d", exp, v)
	}
}

func TestInputOrderStable(t *testing.T) {
	graph := NewGraph()

	inputs := make([]Output, 20)
	for i := range inputs {
		in, err := Placeholder(graph, fmt.Sprintf("input%d", i), Int64)
		if err != nil {
			t.Fatal(err)
		}
		in.Index = i
		inputs[i] = in
	}

	makeArgs := func() *cRunArgs {
		feeds := make(map[Output]*Tensor, len(inputs))
		for i, in := range inputs {
			tensor, err := NewTensor(int64(i + 1))
			if err != nil {
				t.Fatal(err)
			}
			feeds[in] = tensor
		}

		return newCRunArgs(feeds, nil, nil)
	}
	args1 := makeArgs()
	args2 := makeArgs()

	if !reflect.DeepEqual(args1.feeds, args2.feeds) {
		t.Fatalf("order is not stable")
	}
}

func TestSessionRunConcat(t *testing.T) {
	// Runs the Concat operation on two matrices: m1 and m2, along the
	// first dimension (dim1).
	// This tests the use of both Output and OutputList as inputs to the
	// Concat operation.
	var (
		g       = NewGraph()
		dim1, _ = Const(g, "dim1", int32(1))
		m1, _   = Const(g, "m1", [][]int64{
			{1, 2, 3},
			{4, 5, 6},
		})
		m2, _ = Const(g, "m2", [][]int64{
			{7, 8, 9},
			{10, 11, 12},
		})
		want = [][]int64{
			{1, 2, 3, 7, 8, 9},
			{4, 5, 6, 10, 11, 12},
		}
	)
	concat, err := g.AddOperation(OpSpec{
		Type: "Concat",
		Input: []Input{
			dim1,
			OutputList{m1, m2},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewSession(g, &SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	output, err := s.Run(nil, []Output{concat.Output(0)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 1 {
		t.Fatal(len(output))
	}
	if got := output[0].Value(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Got %v, want %v", got, want)
	}
}

func TestSessionWithStringTensors(t *testing.T) {
	// Construct the graph:
	// AsString(StringToHashBucketFast("PleaseHashMe")) Will be much
	// prettier if using the ops package, but in this package graphs are
	// constructed from first principles.
	var (
		g       = NewGraph()
		feed, _ = Const(g, "input", "PleaseHashMe")
		hash, _ = g.AddOperation(OpSpec{
			Type:  "StringToHashBucketFast",
			Input: []Input{feed},
			Attrs: map[string]interface{}{
				"num_buckets": int64(1 << 32),
			},
		})
		str, _ = g.AddOperation(OpSpec{
			Type:  "AsString",
			Input: []Input{hash.Output(0)},
		})
	)
	s, err := NewSession(g, nil)
	if err != nil {
		t.Fatal(err)
	}
	output, err := s.Run(nil, []Output{str.Output(0)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 1 {
		t.Fatal(len(output))
	}
	got, ok := output[0].Value().(string)
	if !ok {
		t.Fatalf("Got %T, wanted string", output[0].Value())
	}
	if want := "1027741475"; got != want {
		t.Fatalf("Got %q, want %q", got, want)
	}
}

func TestConcurrency(t *testing.T) {
	tensor, err := NewTensor(int64(1))
	if err != nil {
		t.Fatalf("NewTensor(): %v", err)
	}

	graph, inp, out := createTestGraph(t, tensor.DataType())
	s, err := NewSession(graph, &SessionOptions{})
	if err != nil {
		t.Fatalf("NewSession(): %v", err)
	}
	for i := 0; i < 100; i++ {
		// Session may close before Run() starts, so we don't check the error.
		go s.Run(map[Output]*Tensor{inp: tensor}, []Output{out}, []*Operation{out.Op})
	}
	if err = s.Close(); err != nil {
		t.Errorf("Close() 1: %v", err)
	}
	if err = s.Close(); err != nil {
		t.Errorf("Close() 2: %v", err)
	}
}

func ExamplePartialRun() {
	var (
		// Create a graph: a + 2 + 3 + b.
		//
		// Skipping error handling for brevity of this example.
		// The 'op' package can be used to make graph construction code
		// with error handling more succinct.
		g        = NewGraph()
		a, _     = Placeholder(g, "a", Int32)
		b, _     = Placeholder(g, "b", Int32)
		two, _   = Const(g, "Two", int32(2))
		three, _ = Const(g, "Three", int32(3))

		plus2, _ = Add(g, "plus2", a, two)       // a + 2
		plus3, _ = Add(g, "plus3", plus2, three) // (a + 2) + 3
		plusB, _ = Add(g, "plusB", plus3, b)     // ((a + 2) + 3) + b

	)
	sess, err := NewSession(g, nil)
	if err != nil {
		panic(err)
	}
	defer sess.Close()

	// All the feeds, fetches and targets for subsequent PartialRun.Run
	// calls must be provided at setup.
	pr, err := sess.NewPartialRun(
		[]Output{a, b},
		[]Output{plus2, plusB},
		[]*Operation{plus3.Op},
	)
	if err != nil {
		panic(err)
	}

	// Feed 'a=1', fetch 'plus2', and compute (but do not fetch) 'plus3'.
	// Imagine this to be the forward pass of unsupervised neural network
	// training of a robot.
	val, _ := NewTensor(int32(1))
	fetches, err := pr.Run(
		map[Output]*Tensor{a: val},
		[]Output{plus2},
		nil)
	if err != nil {
		panic(err)
	}
	v1 := fetches[0].Value().(int32)

	// Now, feed 'b=4', fetch 'plusB=a+2+3+b'
	// Imagine this to be the result of actuating the robot to determine
	// the error produced by the current state of the neural network.
	val, _ = NewTensor(int32(4))
	fetches, err = pr.Run(
		map[Output]*Tensor{b: val},
		[]Output{plusB},
		nil)
	if err != nil {
		panic(err)
	}
	v2 := fetches[0].Value().(int32)

	fmt.Println(v1, v2)
	// Output: 3 10
}

func TestSessionConfig(t *testing.T) {
	// Exercise SessionOptions.
	// Arguably, a better API would be for SessionOptions.Config to be the
	// type generated by the protocol buffer compiler. But for now, the
	// tensorflow package continues to be independent of protocol buffers
	// and this test exercises the option since the implementation has a
	// nuanced conversion to C types.
	//
	// Till then, the []byte form of Config here was generated using a toy
	// tensorflow Python program:
	/*
	 import tensorflow
	 c = tensorflow.ConfigProto()
	 c.intra_op_parallelism_threads = 1
	 print c.SerializeToString()
	*/
	graph := NewGraph()
	c, err := Const(graph, "Const", int32(14))
	if err != nil {
		t.Fatal(err)
	}
	opts := SessionOptions{Config: []byte("(\x01")}
	s, err := NewSession(graph, &opts)
	if err != nil {
		t.Fatal(err)
	}
	output, err := s.Run(nil, []Output{c}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output[0].Value().(int32) != 14 {
		t.Fatalf("Got %v, want -1", output[0].Value())
	}
}

func TestListDevices(t *testing.T) {
	s, err := NewSession(NewGraph(), nil)
	if err != nil {
		t.Fatalf("NewSession(): %v", err)
	}

	devices, err := s.ListDevices()
	if err != nil {
		t.Fatalf("ListDevices(): %v", err)
	}

	if len(devices) == 0 {
		t.Fatalf("no devices detected")
	}
}

func TestDeviceString(t *testing.T) {
	d := Device{Name: "foo", Type: "bar", MemoryLimitBytes: 12345}
	got := d.String()
	want := "(Device: name \"foo\", type bar, memory limit 12345 bytes)"
	if got != want {
		t.Errorf("Got \"%s\", want \"%s\"", got, want)
	}
}

func TestDeviceStringNoMemoryLimit(t *testing.T) {
	d := Device{Name: "foo", Type: "bar", MemoryLimitBytes: -1}
	got := d.String()
	want := "(Device: name \"foo\", type bar, no memory limit)"
	if got != want {
		t.Errorf("Got \"%s\", want \"%s\"", got, want)
	}
}
