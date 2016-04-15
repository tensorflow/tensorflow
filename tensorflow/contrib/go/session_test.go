package tensorflow_test

import (
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func TestNewSession(t *testing.T) {
	graph, err := tf.LoadGraphFromTextFile("test_data/tests_constants_outputs.pb")
	if err != nil {
		t.Fatal(err)
	}
	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}
	outputs := []string{
		"output1",
		"output2",
	}

	output, err := s.Run(nil, outputs, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(output) != len(outputs) {
		t.Fatal("Expected outputs: 2, but got:", len(output))
	}
}

func TestInputParams(t *testing.T) {
	inputSlice1 := []int64{1, 2, 3}
	inputSlice2 := []int64{3, 4, 5}

	t1, err := tf.NewTensor(inputSlice1)
	if err != nil {
		t.Fatal("Error creating a new tensor:", err)
	}

	t2, err := tf.NewTensorWithShape([][]int64{{3}}, inputSlice2)
	if err != nil {
		t.Fatal("Error creating a new tensor:", err)
	}

	graph, err := tf.LoadGraphFromTextFile("test_data/add_three_dim_graph.pb")
	if err != nil {
		t.Fatal("Error reading the graph from the origin file:", err)
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
		t.Fatal("Error running the saved graph:", err)
	}

	if len(out) != 1 {
		t.Fatal("Expected number of outputs is 1, but got:", len(out))
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

func TestInputMultDimParams(t *testing.T) {
	inputSlice1 := [][][]int64{
		{
			{1, 2},
			{3, 4},
		}, {
			{5, 6},
			{7, 8},
		},
	}
	inputSlice2 := [][][]int64{
		{
			{9, 10},
			{11, 12},
		}, {
			{13, 14},
			{15, 16},
		},
	}

	t1, err := tf.NewTensor(inputSlice1)
	if err != nil {
		t.Fatal("Error creating a new tensor:", err)
	}

	t2, err := tf.NewTensor(inputSlice2)
	if err != nil {
		t.Fatal("Error creating a new tensor:", err)
	}

	graph, err := tf.LoadGraphFromTextFile("test_data/test_graph_multi_dim.pb")
	if err != nil {
		t.Fatal("Error reading the graph from the origin file:", err)
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
		t.Fatal("Error running the saved graph:", err)
	}

	if len(out) != 1 {
		t.Fatal("Expected number of outputs is 1, but got:", len(out))
	}

	lenDimX := len(inputSlice1)
	lenDimY := len(inputSlice1[0])
	lenDimZ := len(inputSlice1[0][0])

	for x := 0; x < lenDimX; x++ {
		for y := 0; y < lenDimY; y++ {
			for z := 0; z < lenDimZ; z++ {
				val, err := out[0].GetVal(x, y, z)
				if err != nil {
					t.Fatal("Error reading the output tensor:", err)
				}
				if val != inputSlice1[x][y][z]+inputSlice2[x][y][z] {
					t.Errorf("The sum of the 2 elements: %d + %d doesn't match the returned value: %d", inputSlice1[x][y][z], inputSlice2[x][y][z], val)
				}
			}
		}
	}
}
