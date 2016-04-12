package tensorflow_test

import (
	"fmt"
	"log"

	"github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func ExampleOp() {
	var out []*tensorflow.Tensor

	additions := 10
	inputSlice1 := []int32{1, 2, 3, 4}
	inputSlice2 := []int32{5, 6, 7, 8}

	graph := tensorflow.NewGraph()
	input1, _ := graph.Variable("input1", inputSlice1)
	input2, _ := graph.Constant("input2", inputSlice2)

	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})

	s, _ := tensorflow.NewSession()
	s.ExtendAndInitializeAllVariables(graph)

	for i := 0; i < additions; i++ {
		out, _ = s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
	}

	for i := 0; i < len(inputSlice1); i++ {
		val, _ := out[0].GetVal(i)
		fmt.Println("The result of the operation: %d + (%d*%d) is: %d", inputSlice1[i], inputSlice2[i], additions, val)
	}
}

func ExampleLoadGraphFromTextFile() {
	// This are the input tensors to be used
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

	// Create the two tensors, the data type is recognized automatically as
	// also the tensor shape from the input slice
	t1, _ := tensorflow.NewTensor(inputSlice1)
	t2, _ := tensorflow.NewTensor(inputSlice2)

	// Load the graph from the file that we had generated from Python on
	// the previous step
	graph, _ := tensorflow.LoadGraphFromTextFile("/tmp/graph/test_graph.pb")

	// Create the session and extend the Graph
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)

	input := map[string]*tensorflow.Tensor{
		"input1": t1,
		"input2": t2,
	}
	// Execute the graph with the two input tensors, and specify the names
	// of the tensors to be returned, on this case just one
	out, _ := s.Run(input, []string{"output"}, nil)

	if len(out) != 1 {
		log.Fatalf("The expected number of outputs is 1 but: %d returned", len(out))
	}

	outputTensor := out[0]
	for x := 0; x < outputTensor.Dim(0); x++ {
		for y := 0; y < outputTensor.Dim(1); y++ {
			for z := 0; z < outputTensor.Dim(2); z++ {
				// Using GetVal we can access to the corresponding positions of
				// the tensor as if we had been accessing to the positions in a
				// multidimensional array, for instance GetVal(1, 2, 3) is
				// equivalent to array[1][2][3] on a three dimensional array
				val, _ := out[0].GetVal(x, y, z)
				fmt.Println(
					"The sum of the two elements: %d + %d is equal to: %d",
					inputSlice1[x][y][z], inputSlice2[x][y][z], val)
			}
		}
	}
}

func ExampleNewTensor() {
	tensorflow.NewTensor([][]int64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	})
}
