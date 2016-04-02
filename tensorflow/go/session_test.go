package tensorflow_test

import (
	"testing"

	"github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestNewSession(t *testing.T) {
	graph, err := tensorflow.LoadGraphFromTextFile("test_data/tests_constants_outputs.pb")
	if err != nil {
		t.Fatal(err)
	}
	s, err := tensorflow.NewSession()
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
		t.Fatal("There was", len(outputs), "expected outputs, but:", len(output), "obtained")
	}
}

func TestInputParams(t *testing.T) {
	inputSlice1 := []int64{1, 2, 3}
	inputSlice2 := []int64{3, 4, 5}

	t1, err := tensorflow.NewTensor(inputSlice1)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	t2, err := tensorflow.NewTensorWithShape([][]int64{{3}}, inputSlice2)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	graph, err := tensorflow.LoadGraphFromTextFile("test_data/add_three_dim_graph.pb")
	if err != nil {
		t.Error("Problem trying read the graph from the origin file, Error:", err)
		t.FailNow()
	}

	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	input := map[string]*tensorflow.Tensor{
		"input1": t1,
		"input2": t2,
	}
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		t.Error("Problem trying to run the saved graph, Error:", err)
		t.FailNow()
	}

	if len(out) != 1 {
		t.Errorf("The expected number of outputs is 1 but: %d returned", len(out))
		t.FailNow()
	}

	for i := 0; i < len(inputSlice1); i++ {
		val, err := out[0].GetVal(i)
		if err != nil {
			t.Error("Error trying to read the output tensor, Error:", err)
			t.FailNow()
		}
		if val != inputSlice1[i]+inputSlice2[i] {
			t.Errorf("The sum of the two elements: %d + %d doesn't match with the returned value: %d", inputSlice1[i], inputSlice2[i], val)
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

	t1, err := tensorflow.NewTensor(inputSlice1)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	t2, err := tensorflow.NewTensor(inputSlice2)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	graph, err := tensorflow.LoadGraphFromTextFile("test_data/test_graph_multi_dim.pb")
	if err != nil {
		t.Error("Problem trying read the graph from the origin file, Error:", err)
		t.FailNow()
	}

	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	input := map[string]*tensorflow.Tensor{
		"input1": t1,
		"input2": t2,
	}
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		t.Error("Problem trying to run the saved graph, Error:", err)
		t.FailNow()
	}

	if len(out) != 1 {
		t.Errorf("The expected number of outputs is 1 but: %d returned", len(out))
		t.FailNow()
	}

	for x := 0; x < len(inputSlice1); x++ {
		for y := 0; y < len(inputSlice1[0]); y++ {
			for z := 0; z < len(inputSlice1[0][0]); z++ {
				val, err := out[0].GetVal(x, y, z)
				if err != nil {
					t.Error("Error trying to read the output tensor, Error:", err)
					t.FailNow()
				}
				if val != inputSlice1[x][y][z]+inputSlice2[x][y][z] {
					t.Errorf("The sum of the two elements: %d + %d doesn't match with the returned value: %d", inputSlice1[x][y][z], inputSlice2[x][y][z], val)
				}
			}
		}
	}
}
