// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow_test

import (
	"bufio"
	"fmt"
	"image"
	_ "image/jpeg"
	"io/ioutil"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func Example() {
	// An example for using the TensorFlow Go API for image recognition
	// using a pre-trained inception model (http://arxiv.org/abs/1512.00567).
	//
	// The pre-trained model takes input in the form of a 4-dimensional
	// tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
	// where:
	// - BATCH_SIZE allows for inference of multiple images in one pass through the graph
	// - IMAGE_HEIGHT is the height of the images on which the model was trained
	// - IMAGE_WIDTH is the width of the images on which the model was trained
	// - 3 is the (R, G, B) values of the pixel colors represented as a float.
	//
	// And produces as output a vector with shape [ NUM_LABELS ].
	// output[i] is the probability that the input image was recognized as
	// having the i-th label.
	//
	// A separate file contains a list of string labels corresponding to the
	// integer indices of the output.
	//
	// This example:
	// - Loads the serialized representation of the pre-trained model into a Graph
	// - Creates a Session to execute operations on the Graph
	// - Converts an image file to a Tensor to provide as input for Graph execution
	// - Exectues the graph and prints out the label with the highest probability
	const (
		// Path to a pre-trained inception model.
		// The two files are extracted from a zip archive as so:
		/*
		   curl -L https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip -o /tmp/inception5h.zip
		   unzip /tmp/inception5h.zip -d /tmp
		*/
		modelFile  = "/tmp/tensorflow_inception_graph.pb"
		labelsFile = "/tmp/imagenet_comp_graph_label_strings.txt"

		// Image file to "recognize".
		testImageFilename = "/tmp/test.jpg"
	)

	// Load the serialized GraphDef from a file.
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// Run inference on testImageFilename.
	// For multiple images, session.Run() can be called in a loop (and
	// concurrently). Furthermore, images can be batched together since the
	// model accepts batches of image data as input.
	tensor, err := makeTensorFromImageForInception(testImageFilename)
	if err != nil {
		log.Fatal(err)
	}
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	// output[0].Value() is a vector containing probabilities of
	// labels for each image in the "batch". The batch size was 1.
	// Find the most probably label index.
	probabilities := output[0].Value().([][]float32)[0]
	printBestLabel(probabilities, labelsFile)
}

func printBestLabel(probabilities []float32, labelsFile string) {
	bestIdx := 0
	for i, p := range probabilities {
		if p > probabilities[bestIdx] {
			bestIdx = i
		}
	}
	// Found a best match, now read the string from the labelsFile where
	// there is one line per label.
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
	fmt.Printf("BEST MATCH: (%2.0f%% likely) %s\n", probabilities[bestIdx]*100.0, labels[bestIdx])
}

// Given an image stored in filename, returns a Tensor which is suitable for
// providing the image data to the pre-defined model.
func makeTensorFromImageForInception(filename string) (*tf.Tensor, error) {
	const (
		// Some constants specific to the pre-trained model at:
		// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
		//
		// - The model was trained after with images scaled to 224x224 pixels.
		// - The colors, represented as R, G, B in 1-byte each were converted to
		//   float using (value - Mean)/Std.
		//
		// If using a different pre-trained model, the values will have to be adjusted.
		H, W = 224, 224
		Mean = 117
		Std  = float32(1)
	)
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	sz := img.Bounds().Size()
	if sz.X != W || sz.Y != H {
		return nil, fmt.Errorf("input image is required to be %dx%d pixels, was %dx%d", W, H, sz.X, sz.Y)
	}
	// 4-dimensional input:
	// - 1st dimension: Batch size (the model takes a batch of images as
	//                  input, here the "batch size" is 1)
	// - 2nd dimension: Rows of the image
	// - 3rd dimension: Columns of the row
	// - 4th dimension: Colors of the pixel as (B, G, R)
	// Thus, the shape is [1, 224, 224, 3]
	var ret [1][H][W][3]float32
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			px := x + img.Bounds().Min.X
			py := y + img.Bounds().Min.Y
			r, g, b, _ := img.At(px, py).RGBA()
			ret[0][y][x][0] = float32((int(b>>8) - Mean)) / Std
			ret[0][y][x][1] = float32((int(g>>8) - Mean)) / Std
			ret[0][y][x][2] = float32((int(r>>8) - Mean)) / Std
		}
	}
	return tf.NewTensor(ret)
}
