package tensorflow_test

import (
	"fmt"
	"os"
	"strings"

	"github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func ExampleGraph_Op() {
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
		val, _ := out[0].GetVal(int64(i))
		fmt.Println("The result of: %d + (%d*%d) is: %d", inputSlice1[i], inputSlice2[i], additions, val)
	}
}

func ExampleNewTensor_slice() {
	tensorflow.NewTensor([][]int64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	})
}

func ExampleNewTensor_scalar() {
	tensorflow.NewTensor("Hello TensorFlow")
}

func ExampleNewGraphFromText() {
	graph, err := tensorflow.NewGraphFromReader(strings.NewReader(`
		node {
			name: "output"
			op: "Const"
			attr {
				key: "dtype"
				value {
					type: DT_FLOAT
				}
			}
			attr {
				key: "value"
				value {
					tensor {
						dtype: DT_FLOAT
						tensor_shape {
						}
						float_val: 1.5 
					}
				}
			}
		}
		version: 5`), true)

	fmt.Println(graph, err)
}

func ExampleGraph_Constant() {
	graph := tensorflow.NewGraph()
	// Add a scalar string node named 'const1' to the Graph.
	graph.Constant("const1", "this is a test...")

	// Add bidimensional Constant named 'const2' to the Graph.
	graph.Constant("const2", [][]int64{
		{1, 2},
		{3, 4},
	})
}

func ExampleGraph_Placeholder() {
	graph := tensorflow.NewGraph()
	// Add Placeholder named 'input1' that must allocate a three element
	// DTInt32 tensor.
	graph.Placeholder("input1", tensorflow.DTInt32, []int64{3})
}

func ExampleGraph_Variable() {
	var out []*tensorflow.Tensor

	graph := tensorflow.NewGraph()
	// Create Variable that will be used as input and also as storage of
	// the result after every execution.
	input1, _ := graph.Variable("input1", []int32{1, 2, 3, 4})
	input2, _ := graph.Constant("input2", []int32{5, 6, 7, 8})

	// Add the two inputs.
	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	// Store the result on input1 Varable.
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})

	s, _ := tensorflow.NewSession()
	// Initialize all the Variables in memory, in this case only the
	// 'input1' Variable.
	s.ExtendAndInitializeAllVariables(graph)

	// Run ten times the 'assign_inp1"' that will run also the 'Add'
	// operation since it input depends on the result of the 'Add'
	// operation.
	// The variable 'input1' will be returned and printed on each
	// execution.
	for i := 0; i < 10; i++ {
		out, _ = s.Run(nil, []string{"input1"}, []string{"assign_inp1"})
		fmt.Println(out[0].Int32s())
	}
}

func ExampleSession_ExtendAndInitializeAllVariables() {
	graph := tensorflow.NewGraph()
	// Create Variable that will be initialized with the values []int32{1, 2, 3, 4} .
	graph.Variable("input1", []int32{1, 2, 3, 4})

	s, _ := tensorflow.NewSession()
	// Initialize all the Variables in memory, on this case only the
	// 'input1' variable.
	s.ExtendAndInitializeAllVariables(graph)
}

func ExampleSession_ExtendGraph() {
	graph := tensorflow.NewGraph()
	// Add a Placeholder named 'input1' that must allocate a three element
	// DTInt32 tensor.
	graph.Placeholder("placeholder", tensorflow.DTInt32, []int64{3})

	// Create the Session and extend the Graph on it.
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)
}

func ExampleNewGraphFromReader() {
	// Load the Graph from from a file who contains a previously generated
	// Graph as text.
	reader, _ := os.Open("/tmp/graph/test_graph.pb")
	graph, _ := tensorflow.NewGraphFromReader(reader, true)

	// Create the Session and extend the Graph on it.
	s, _ := tensorflow.NewSession()
	s.ExtendGraph(graph)
}

func ExampleSession_Run() {
	graph := tensorflow.NewGraph()
	input1, _ := graph.Variable("input1", []int32{1, 2, 3, 4})
	input2, _ := graph.Constant("input2", []int32{5, 6, 7, 8})

	add, _ := graph.Op("Add", "add_tensors", []*tensorflow.GraphNode{input1, input2}, "", map[string]interface{}{})
	graph.Op("Assign", "assign_inp1", []*tensorflow.GraphNode{input1, add}, "", map[string]interface{}{})

	s, _ := tensorflow.NewSession()
	s.ExtendAndInitializeAllVariables(graph)

	out, _ := s.Run(nil, []string{"input1"}, []string{"assign_inp1"})

	// The first of the output corresponds to the node 'input1' specified
	// on the second param.
	fmt.Println(out[0])
}

func ExampleNewTensorWithShape() {
	// Create Tensor with a single dimension of 3.
	t2, _ := tensorflow.NewTensorWithShape([]int64{3}, []int64{3, 4, 5})
	fmt.Println(t2.Int64s())
}

func ExampleTensor_GetVal() {
	t, _ := tensorflow.NewTensor([][]int64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	})

	// Print the number 8 that is in the second position of the first
	// dimension and the third of the second dimension.
	fmt.Println(t.GetVal(1, 3))
}
