package tensorflow_test

import (
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func TestGraphPlaceholder(t *testing.T) {
	graph := tf.NewGraph()
	input1 := graph.Placeholder("input1", tf.DTInt32, []int64{3})
	input2 := graph.Placeholder("input2", tf.DTInt32, []int64{3})
	_, err := graph.Op("Add", "output", []*tf.GraphNode{input1, input2}, "", nil)
	if err != nil {
		t.Fatal("Error adding 2 tensors:", err)
	}

	_, err = graph.Op("Add", "output", []*tf.GraphNode{input2}, "", map[string]interface{}{
		"T": tf.DTInt32,
	})
	if err == nil {
		t.Error("An operation with 2 mandatory parameters was added after specifying just 1")
	}
	_, err = graph.Op("Aajajajajdd", "output", []*tf.GraphNode{input2}, "", map[string]interface{}{})
	if err == nil {
		t.Error("An undefined operation was added to the graph")
	}

	inputSlice1 := []int32{1, 2, 3}
	inputSlice2 := []int32{3, 4, 5}

	t1, err := tf.NewTensor(inputSlice1)
	if err != nil {
		t.Fatal("Error creating a new tensor:", err)
	}

	t2, err := tf.NewTensor(inputSlice2)
	if err != nil {
		t.Fatal("Error creating a new tensor:", err)
	}

	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	input := map[string]*tf.Tensor{
		"input1": t1,
		"input2": t2,
	}

	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		t.Fatal("Error running the graph:", err)
	}

	if len(out) != 1 {
		t.Fatal("Expected 1 output, got:", len(out))
	}

	for i := 0; i < len(inputSlice1); i++ {
		val, err := out[0].GetVal(i)
		if err != nil {
			t.Fatal("Error reading the output tensor:", err)
		}
		if val != inputSlice1[i]+inputSlice2[i] {
			t.Errorf("The sum of the 2 elements: %d + %d doesn't match the returned value: %d", inputSlice1[i], inputSlice2[i], val)
		}
	}
}

func TestGraphScalarConstant(t *testing.T) {
	graph := tf.NewGraph()
	testString := "this is a test..."
	testFloat := float64(123.123)

	_, err := graph.Constant("output1", testString)
	if err != nil {
		t.Fatal("Error adding a scalar constant to the graph:", err)
	}

	_, err = graph.Constant("output2", testFloat)
	if err != nil {
		t.Fatal("Error adding a scalar constant to the graph:", err)
	}

	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	out, err := s.Run(nil, []string{"output1", "output2"}, nil)
	if err != nil {
		t.Fatal("Error running the graph:", err)
	}

	if len(out) != 2 {
		t.Fatal("Expected 2 output tensors, received:", len(out))
	}

	outStr, err := out[0].Str()
	if err != nil {
		t.Error("Error reading the output:", err)
	} else {
		if string(outStr[0]) != testString {
			t.Error("The returned string: \"%s\" is not the input string: \"%s\"", testString, outStr[0])
		}
	}

	outFloat, err := out[1].Float64()
	if err != nil {
		t.Error("Error reading the output:", err)
	} else {
		if outFloat[0] != testFloat {
			t.Error("The returned float: \"%f\" is not the input 1: \"%f\"", outFloat[0], testFloat)
		}
	}
}

func TestGraphVariableConstant(t *testing.T) {
	var out []*tf.Tensor

	additions := 10
	inputSlice1 := []int32{1, 2, 3, 4}
	inputSlice2 := []int32{5, 6, 7, 8}

	graph := tf.NewGraph()
	input1, err := graph.Variable("input1", inputSlice1)
	if err != nil {
		t.Fatal("Error adding a variable to the graph:", err)
	}

	input2, err := graph.Constant("input2", inputSlice2)
	if err != nil {
		t.Fatal("Error adding a constant to the graph:", err)
	}

	add, err := graph.Op("Add", "add_tensors", []*tf.GraphNode{input1, input2}, "", map[string]interface{}{})
	if err != nil {
		t.Fatal("Error adding 2 tensors:", err)
	}

	_, err = graph.Op("Assign", "assign_inp1", []*tf.GraphNode{input1, add}, "", map[string]interface{}{})
	if err != nil {
		t.Fatal("Error assigning the result of the sum to the tensor:", err)
	}

	s, err := tf.NewSession()
	s.ExtendAndInitializeAllVariables(graph)
	if err != nil {
		t.Fatal("Error initializing the variables in the graph:", err)
	}

	for i := 0; i < additions; i++ {
		out, err = s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
		if err != nil {
			t.Fatal("Error running the graph:", err)
		}
	}
	if err != nil {
		t.Fatal("Error running the graph:", err)
	}

	if len(out) != 1 {
		t.Fatal("Expected 1 output, got:", len(out))
	}

	for i := 0; i < len(inputSlice1); i++ {
		val, err := out[0].GetVal(i)
		if err != nil {
			t.Fatal("Error reading the output tensor:", err)
		}
		if val != inputSlice1[i]+(inputSlice2[i]*int32(additions)) {
			t.Errorf("The sum of the 2 elements: %d + (%d*%d) doesn't match the returned value: %d",
				inputSlice1[i], inputSlice2[i], additions, val)
		}
	}
}
