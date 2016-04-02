package tensorflow_test

import (
	"testing"

	"github.com/tensorflow/tensorflow/tensorflow/go"
)

func TestGraphGeneration(t *testing.T) {
	graph := tensorflow.NewGraph()
	graph.AddPlaceholder("input1", tensorflow.DtInt32, []int64{3}, []string{})
	graph.AddPlaceholder("input2", tensorflow.DtInt32, []int64{3}, []string{})
	err := graph.AddOp("Add", "output", []string{"input1", "input2"}, "", map[string]interface{}{
		"T": tensorflow.DtInt32,
	})
	if err != nil {
		t.Error("Problem trying add two tensord, Error:", err)
		t.FailNow()
	}

	err = graph.AddOp("Add", "output", []string{"input1", "input2"}, "", map[string]interface{}{})
	if err == nil {
		t.Error("An operation with a mandatory attribute was added without specify this parameter")
	}

	err = graph.AddOp("Add", "output", []string{"input2"}, "", map[string]interface{}{
		"T": tensorflow.DtInt32,
	})
	if err == nil {
		t.Error("An with two mandatory parameters was added after specify just one")
	}
	err = graph.AddOp("Aajajajajdd", "output", []string{"input2"}, "", map[string]interface{}{})
	if err == nil {
		t.Error("An undefined operation was added to the graph")
	}

	inputSlice1 := []int32{1, 2, 3}
	inputSlice2 := []int32{3, 4, 5}

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
		t.Error("Problem trying to run the graph, Error:", err)
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

func TestGraphConstant(t *testing.T) {
	inputSlice1 := []int32{1, 2, 3}
	inputSlice2 := []int32{3, 4, 5}

	graph := tensorflow.NewGraph()
	graph.AddPlaceholder("input1", tensorflow.DtInt32, []int64{3}, []string{})

	_, err := graph.Constant("input2", inputSlice2)
	if err != nil {
		t.Error("Problem trying add a constant to the graph, Error:", err)
		t.FailNow()
	}

	err = graph.AddOp("Add", "output", []string{"input1", "input2"}, "", map[string]interface{}{
		"T": tensorflow.DtInt32,
	})
	if err != nil {
		t.Error("Problem trying add two tensord, Error:", err)
		t.FailNow()
	}

	err = graph.AddOp("Add", "output", []string{"input1", "input2"}, "", map[string]interface{}{})
	if err == nil {
		t.Error("An operation with a mandatory attribute was added without specify this parameter")
	}

	t1, err := tensorflow.NewTensor(inputSlice1)
	if err != nil {
		t.Error("Problem trying create a new tensor, Error:", err)
		t.FailNow()
	}

	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal(err)
	}

	input := map[string]*tensorflow.Tensor{
		"input1": t1,
	}
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		t.Error("Problem trying to run the graph, Error:", err)
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
