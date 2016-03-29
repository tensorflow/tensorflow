package tensorflow_test

import (
	"testing"

	"github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestNewSession(t *testing.T) {
	graph := &tensorflow.GraphDef{}

	graph, err := tensorflow.LoadGraphFromText("test_data/tests_constants_outputs.pb")
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
	inputSlice2 := []int64{4, 5, 6}

	t1, err := tensorflow.NewTensor([][]int64{{3}}, inputSlice1)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	t2, err := tensorflow.NewTensor([][]int64{{3}}, inputSlice2)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	graph, err := tensorflow.LoadGraphFromText("test_data/add_three_dim_graph.pb")
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
