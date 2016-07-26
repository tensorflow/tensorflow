package tensorflow_test

import (
	"os"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func TestNewSession(t *testing.T) {
	s := loadAndExtendGraphFromFile(t, "test_data/tests_constants_outputs.pb")
	outputs := []string{
		"output1",
		"output2",
	}
	output, err := s.Run(nil, outputs, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(output) != len(outputs) {
		t.Fatal("Expected outputs: 2, got:", len(output))
	}
}

func TestInputParams(t *testing.T) {
	inputSlice1 := []int64{1, 2, 3}
	inputSlice2 := []int64{3, 4, 5}

	s := loadAndExtendGraphFromFile(t, "test_data/add_three_dim_graph.pb")

	t1, err := tf.NewTensor(inputSlice1)
	if err != nil {
		t.Fatal("Error creating Tensor:", err)
	}

	t2, err := tf.NewTensorWithShape([]int64{3}, inputSlice2)
	if err != nil {
		t.Fatal("Error creating Tensor:", err)
	}

	input := map[string]*tf.Tensor{
		"input1": t1,
		"input2": t2,
	}
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		t.Fatal("Error running loaded Graph:", err)
	}

	if len(out) != 1 {
		t.Fatal("Expected outputs: 1, got:", len(out))
	}

	for i := 0; i < len(inputSlice1); i++ {
		val, err := out[0].GetVal(int64(i))
		if err != nil {
			t.Fatal("Error reading output Tensor:", err)
		}
		if val != inputSlice1[i]+inputSlice2[i] {
			t.Errorf("Expected result: %d + %d, got: %d", inputSlice1[i], inputSlice2[i], val)
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
		t.Fatal("Error creating Tensor:", err)
	}

	t2, err := tf.NewTensor(inputSlice2)
	if err != nil {
		t.Fatal("Error creating Tensor:", err)
	}

	s := loadAndExtendGraphFromFile(t, "test_data/test_graph_multi_dim.pb")

	input := map[string]*tf.Tensor{
		"input1": t1,
		"input2": t2,
	}
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		t.Fatal("Error running loaded Graph:", err)
	}

	if len(out) != 1 {
		t.Fatal("Expected outputs: 1, got:", len(out))
	}

	lenDimX := len(inputSlice1)
	lenDimY := len(inputSlice1[0])
	lenDimZ := len(inputSlice1[0][0])

	for x := 0; x < lenDimX; x++ {
		for y := 0; y < lenDimY; y++ {
			for z := 0; z < lenDimZ; z++ {
				val, err := out[0].GetVal(int64(x), int64(y), int64(z))
				if err != nil {
					t.Fatal("Error reading output Tensor:", err)
				}
				if val != inputSlice1[x][y][z]+inputSlice2[x][y][z] {
					t.Errorf("Expected: %d + %d, got: %d", inputSlice1[x][y][z], inputSlice2[x][y][z], val)
				}
			}
		}
	}
}

func loadAndExtendGraphFromFile(t *testing.T, filePath string) (s *tf.Session) {
	s, err := tf.NewSession()
	if err != nil {
		t.Fatal("Error creating Session:", err)
	}

	reader, err := os.Open(filePath)
	if err != nil {
		t.Fatal("Error reading Graph definition file:", err)
	}
	graph, err := tf.NewGraphFromReader(reader, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.ExtendGraph(graph); err != nil {
		t.Fatal("Error extending the Graph into the Session:", err)
	}

	return s
}
