# TensorFlow Go API

This package provides a high-level Go API for TensorFlow, providing the
necessary tools to create and manipulate Tensors, Variables, Constants and
also to build, load and run Graphs.

## API documentation
* [Session](g3doc/session.md): Encapsulates the environment in which Operation
  objects are executed, and Tensor objects are evaluated.
* [Graph](g3doc/graph.md): Contains a set of Operations, which represent units
  of computation; and Tensors, which represent the units of data that flow
  between operations.
* [Tensor](g3doc/tensor.md): Typed multi-dimensional array.

## Installation

This package depends on the TensorFlow shared libraries, in order to compile
this libraries follow the [Installing fromsources](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#installing-from-sources)
guide to clone and configure the repository.

After you have cloned the repository, run the next commands at the root of the
tree:

```sh
$ bazel build //tensorflow:libtensorflow.so
$ sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/lib/
$ go get github.com/tensorflow/tensorflow/tensorflow/contrib/go
```

## Practical Examples

##### Python Graph generated and executed on Go

This example shows how to interact with the provided API.

In order to generate a valid Graph, you can use the next Python code:

```python
import tensorflow as tf

input1 = tf.placeholder(tf.int64, shape=(2, 2, 2), name='input1')
input2 = tf.placeholder(tf.int64, shape=(2, 2, 2), name='input2')
output = tf.add(input1, input2, name='output')

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '/tmp/graph/', 'test_graph.pb', as_text=True)
```

The previous code prepares two placeholders with names 'input1' and
'input2' respectively, and other tensor used as output of the addition of the
two placeholders. At the end, it dumps the graph as text into a text file with
path:
'/tmp/graph/test_graph.pb'.

From a Go application, you can use the next code to execute the graph:

```go
package main

import (
	"log"

	"github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

func main() {
	// These are the input tensors to be used
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
	t1, err := tensorflow.NewTensor(inputSlice1)
	if err != nil {
		log.Fatal("Error creating a new Tensor:", err)
	}

	t2, err := tensorflow.NewTensor(inputSlice2)
	if err != nil {
		log.Fatal("Error creating new Tensor, Error:", err)
	}

	// Load the graph from the file that we had generated from Python on
	// the previous step
	reader, err := os.Open("/tmp/graph/test_graph.pb")
	if err != nil {
		t.Fatal(err)
	}
	graph, err := tensorflow.NewGraphFromReader(reader, true)
	if err != nil {
		log.Fatal("Error reading Graph from the text file:", err)
	}

	// Create the session and extend the Graph
	s, err := tensorflow.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		log.Fatal("Error extending Graph:", err)
	}

	input := map[string]*tensorflow.Tensor{
		"input1": t1,
		"input2": t2,
	}
	// Execute the graph using two input Tensors, and specifying the names
	// of the Tensors to be returned, for this case, just one
	out, err := s.Run(input, []string{"output"}, nil)
	if err != nil {
		log.Fatal("Error running the loaded Graph:", err)
	}

	if len(out) != 1 {
		log.Fatal("Expected number of outputs: 1, but got:", len(out))
	}

	outputTensor := out[0]
	for x := 0; x < outputTensor.Dim(0); x++ {
		for y := 0; y < outputTensor.Dim(1); y++ {
			for z := 0; z < outputTensor.Dim(2); z++ {
				// Using GetVal we can access to the corresponding positions of
				// the tensor as if we would be working with a multidimensional
				// array, for instance, GetVal(1,2, 3) is equivalent to
				// array[1][2][3] on a three dimensional array
				val, err := out[0].GetVal(x, y, z)
				if err != nil {
					log.Fatal("Error reading the output Tensor:", err)
				}
				if val != inputSlice1[x][y][z]+inputSlice2[x][y][z] {
					log.Printf(
						"The value of: %d + %d is not: %d",
						inputSlice1[x][y][z], inputSlice2[x][y][z], val)
				}

				log.Println("Value on coordinates:", x, y, z, "is:", val)
			}
		}
	}
}
```

The previous code creates two Tensors to be processed by the previously
generated Graph, after the execution it returns a Tensor with the operation
result.

### Image Recognition

This is a complete code example that shows how to generate Graphs on Go and
execute them:

[Image Recognition](../../g3doc/tutorials/image_recognition/index.md#usage-with-the-go-api)

[Code](../../examples/label_image_go/main.go)

## Troubleshooting

### ld: library not found for -ltensorflow

This package expects the linker to find the 'libtensorflow' shared library. 

To generate this file run:

```sh
$ go generate github.com/tensorflow/tensorflow/tensorflow/contrib/go
```

`libtensorflow.so` will end up at `${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow/libtensorflow.so`.

