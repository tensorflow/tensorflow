# tensorflow for Go

This package provides a SWIG-based Go package to run Tensorflow graphs in Go programs.

It is not an immediate goal to support Graph generation via this package.

A higher level API is presented in the 'github.com/tensorflow/tensorflow/tensorflow/go' package.

## Practical Example

This is just an example in order to show how to interact with the provided API

In order to generate a valid Graph you can use the next Python code

```python
import tensorflow as tf

input1 = tf.placeholder(tf.int64, shape=(3), name='input1')
input2 = tf.placeholder(tf.int64, shape=(3), name='input2')
output = tf.add(input1, input2, name='output')

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '/tmp/graph/', 'test_graph.pb', as_text=True)
```

The previous code will prepare two placeholders with names 'input1' and 'input2' respectively, and another tensor used as output of the addition of the two placeholders. And dumps the graph as text into a text file with path: '/tmp/graph/test_graph.pb'.

From a golang aplication you can use the next code to execute the graph:

```go
package main

import (
	"log"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// This are the input tensors to be used
	inputSlice1 := []int64{1, 2, 3}
	inputSlice2 := []int64{4, 5, 6}

	// Create the two tensors, the data type is recognised automatically,
	// the second parameter are the unrolled tensor values
	t1, err := tensorflow.NewTensor([][]int64{{3}}, inputSlice1)
	if err != nil {
		log.Fatal("Problem trying create a new tensor, Error:", err)
	}

	t2, err := tensorflow.NewTensor([][]int64{{3}}, inputSlice2)
	if err != nil {
		log.Fatal("Problem trying create a new tensor, Error:", err)
	}

	// Load the graph from the file that we had generated from Python on
	// the previous step
	graph, err := tensorflow.LoadGraphFromText("/tmp/graph/test_graph.pb")
	if err != nil {
		log.Fatal("Problem reading the graph from the text file, Error:", err)
	}

	// Create the session and extend the Graph
	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		log.Fatal("Problem extending the Graph, Error:", err)
	}

	input := map[string]*tensorflow.Tensor{
		"input1": t1,
		"input2": t2,
	}
	// Execute the graph with the two input tensors, and specify the names
	// of the tensors to be returned, on this case just one
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the saved graph, Error:", err)
	}

	if len(out) != 1 {
		log.Fatalf("The expected number of outputs is 1 but: %d returned", len(out))
	}

	for i := 0; i < len(inputSlice1); i++ {
		// Using GetVal we can access to the corresponding positions of
		// the tensor as if we had been accessing to the positions in a
		// multidimensional array, for instance GetVal(1, 2, 3) is
		// equivalent to array[1][2][3] on a three dimensional array
		val, err := out[0].GetVal(i)
		if err != nil {
			log.Fatal("Error trying to read the output tensor, Error:", err)
		}
		if val != inputSlice1[i]+inputSlice2[i] {
			log.Println("The sum of the two elements: %d + %d doesn't match with the returned value: %d", inputSlice1[i], inputSlice2[i], val)
		}

		log.Println("The value value on position:", i, "is:", val)
	}
}
```

As you can see the previous code creates two Tensors to be processed by the previously generated Graph and after the execution returns the Tensor with the result.

## Troubleshooting

```ld: library not found for -ltensorflow```

This package expects the linker to find the 'libtensorflow' shared library. 

To generate this file run:

```sh
$ go generate github.com/tensorflow/tensorflow/tensorflow/go
```

`libtensorflow.so` will end up at `${GOPATH}/src/github.com/tensorflow/bazel-bin/tensorflow/libtensorflow.so`.

