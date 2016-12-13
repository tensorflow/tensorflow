package main

/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful Go example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. The errors management
// while building the Graph has been ommited in order to make the code less
// verbose, but for production code is recomended to check the returned errors
// on each step.
//
// To use it, run in a working directory with the
// tensorflow/examples/label_image/ folder below it:
//   go run tensorflow/examples/label_image_go/main.go <image_file>
// you can use the image of "Admiral Grace Hopper"
// tensorflow/examples/label_image/data/grace_hopper.jpg as image
// file, and you can see the network correctly identifies she's wearing a
// military uniform, with a high score of 0.6.
//
// The tensorflow_inception_graph.pb file included by default is created from
// Inception.

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/contrib/go"
)

const (
	cInputWidth  = 299
	cInputHeight = 299
	cInputMean   = 128
	cInputStd    = 128

	// cInceptionGraphFile This is the path of the graph that contains the
	// model.
	cInceptionGraphFile = "tensorflow/examples/label_image_go/data/tensorflow_inception_graph.pb"
	// cLabelsFile File path that contains a label per line.
	cLabelsFile = "tensorflow/examples/label_image_go/data/imagenet_comp_graph_label_strings.txt"

	// cLabelsToShow Number of best match labels to be returned per image.
	cLabelsToShow = 5
)

// readTensorFromImageFile reads the data from the given an image file name, try to
// decode it as an image, resize it to the requested size, and then scale the
// values as desired.
func readTensorFromImageFile(filePath string) *tf.Tensor {
	var imageReader *tf.GraphNode

	graph := tf.NewGraph()

	// Load the image file path into the graph as a constant, and read its
	// content.
	fileNameNode, _ := graph.Constant("file_name", filePath)
	fileReader, _ := graph.Op("ReadFile", "file_reader", []*tf.GraphNode{fileNameNode}, "", nil)

	// Now try to figure out what kind of file it is and decode it.
	if filepath.Ext(filePath) == ".png" {
		imageReader, _ = graph.Op("DecodePng", "png_reader", []*tf.GraphNode{fileReader}, "", map[string]interface{}{
			"channels": int64(3),
		})
	} else {
		// Assume if it's not a PNG then it must be a JPEG.
		imageReader, _ = graph.Op("DecodeJpeg", "jpeg_reader", []*tf.GraphNode{fileReader}, "", map[string]interface{}{
			"channels": int64(3),
		})
	}

	// Now cast the image data to float so we can do normal math on it. In
	// the attributes we have to specify the output datatype that we want.
	floatCaster, _ := graph.Op("Cast", "float_caster", []*tf.GraphNode{imageReader}, "", map[string]interface{}{
		"DstT": tf.DTFloat,
	})

	// The convention for image ops in TensorFlow is that all images are expected
	// to be in batches, so that they're four-dimensional arrays with indices of
	// [batch, height, width, channel]. Because we only have a single image, we
	// have to add a batch dimension of 1 to the start with ExpandDims operation.
	dimIndex, _ := graph.Constant("dim_index", []int32{0})
	dimsExpander, _ := graph.Op("ExpandDims", "dims_expander", []*tf.GraphNode{floatCaster, dimIndex}, "", map[string]interface{}{
		"T":   tf.DTFloat,
		"dim": 0,
	})

	// Bilinearly resize the image to fit the required dimensions.
	sizeDims, _ := graph.Constant("size_dims", []int32{cInputWidth, cInputHeight})
	size, _ := graph.Op("ResizeBilinear", "size", []*tf.GraphNode{dimsExpander, sizeDims}, "", map[string]interface{}{
		"T": tf.DTFloat,
	})

	// Subtract the mean and divide by the scale.
	inputMean, _ := graph.Constant("input_mean", float32(cInputMean))
	subMean, _ := graph.Op("Sub", "sub_mean", []*tf.GraphNode{size, inputMean}, "", nil)
	inputStd, _ := graph.Constant("input_std", float32(cInputStd))
	_, _ = graph.Op("Div", "normalized", []*tf.GraphNode{subMean, inputStd}, "", nil)

	// Create the session and extend the Graph
	s, err := tf.NewSession()
	s.ExtendGraph(graph)

	/// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	out, err := s.Run(nil, []string{"normalized"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the graph, Error:", err)
	}

	if len(out) != 1 {
		log.Fatalf("The expected number of outputs is 1 but: %d returned", len(out))
	}

	// The outputs are sorted in the same order than they were specfied in
	// the second param of the Run method
	return out[0]
}

// getTopLabels Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
func getTopLabels(outputs *tf.Tensor, labels int32) (indexes, scores *tf.Tensor) {
	graph := tf.NewGraph()

	// We are going to use the next placeholder to allocate the tensor with
	// the scores that we will provide as input
	dims := make([]int64, outputs.NumDims())
	for i := 0; i < outputs.NumDims(); i++ {
		dims[i] = int64(outputs.Dim(i))
	}
	normalized := graph.Placeholder("normalized_placeholder", tf.DTFloat, dims)

	// Here instead of using a placeholder to send and input we are using a
	// constant that is oing to be part of the graph
	labelsConstant, _ := graph.Constant("lables", labels)

	// The TopK node returns two outputs, the scores and their original indices,
	// so we have to append :0 and :1 to specify them both.
	_, _ = graph.Op("TopKV2", "normalized", []*tf.GraphNode{normalized, labelsConstant}, "", nil)

	input := map[string]*tf.Tensor{
		"normalized_placeholder": outputs,
	}

	// Create the session and run the previously prepared graph
	s, _ := tf.NewSession()
	s.ExtendGraph(graph)
	out, _ := s.Run(input, []string{"normalized:0", "normalized:1"}, nil)

	return out[1], out[0]
}

func main() {
	if len(os.Args) < 2 {
		log.Println("Usage:", os.Args[0], "<image_path>")
	}
	imagePath := os.Args[1]

	// First we load and initialize the model.
	reader, _ := os.Open(cInceptionGraphFile)
	graph, _ := tf.NewGraphFromReader(reader, false)

	s, _ := tf.NewSession()
	s.ExtendGraph(graph)

	// Get the image from disk as an array of floats, resized and
	// normalized to the specifications the main graph expects.
	resizedTensor := readTensorFromImageFile(imagePath)

	// Prepare map of input tensors to run the model.
	input := map[string]*tf.Tensor{
		"Mul": resizedTensor,
	}

	// Actually run the image through the model.
	out, err := s.Run(input, []string{"softmax"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the saved graph, Error:", err)
	}

	// Using the output tensor from the previous run, get the best matching
	// labels.
	indexTens, scoresTens := getTopLabels(out[0], cLabelsToShow)

	// Associate the labels with the returned data and print them with the
	// corresponding scores.
	labelsStr, err := ioutil.ReadFile(cLabelsFile)
	if err != nil {
		log.Fatal("Error reading the image labels file, Error:", err)
	}

	labels := strings.Split(string(labelsStr), "\n")

	for i := 0; i < cLabelsToShow; i++ {
		index, _ := indexTens.GetVal(0, int64(i))
		score, _ := scoresTens.GetVal(0, int64(i))
		fmt.Println(labels[index.(int32)], ":", score)
	}
}
