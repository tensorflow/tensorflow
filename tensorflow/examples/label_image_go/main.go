package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const (
	cLabelsToShow = 5
)

func readTensorFromImageFile(filePath string) *tf.Tensor {
	graph := tf.NewGraph()

	fileNameNode, err := graph.Constant("file_name", filePath)
	if err != nil {
		log.Fatal("Problem adding the input file name to the graph, Error:", err)
	}

	fileReader, err := graph.AddOp("ReadFile", "file_reader", []*tf.GraphNode{fileNameNode}, "", nil)
	if err != nil {
		log.Fatal("Problem adding ReadFile operation, Error:", err)
	}

	var imageReader *tf.GraphNode
	if filePath[len(filePath)-4:] == ".png" {
		imageReader, err = graph.AddOp("DecodePng", "png_reader", []*tf.GraphNode{fileReader}, "", map[string]interface{}{
			"channels": int64(3),
		})
		if err != nil {
			log.Fatal("Problem adding DecodeJpeg operation, Error:", err)
		}
	} else {
		imageReader, err = graph.AddOp("DecodeJpeg", "jpeg_reader", []*tf.GraphNode{fileReader}, "", map[string]interface{}{
			"channels": int64(3),
		})
		if err != nil {
			log.Fatal("Problem adding DecodeJpeg operation, Error:", err)
		}
	}

	floatCaster, err := graph.AddOp("Cast", "float_caster", []*tf.GraphNode{imageReader}, "", map[string]interface{}{
		"DstT": tf.DtFloat,
	})
	if err != nil {
		log.Fatal("Problem adding Cast operation, Error:", err)
	}

	dimIndex, err := graph.Constant("dim_index", []int32{0})
	if err != nil {
		log.Fatal("Problem adding a constant to the graph, Error:", err)
	}
	dimsExpander, err := graph.AddOp("ExpandDims", "dims_expander", []*tf.GraphNode{floatCaster, dimIndex}, "", map[string]interface{}{
		"T":   tf.DtFloat,
		"dim": 0,
	})
	if err != nil {
		log.Fatal("Problem adding ExpandDims operation, Error:", err)
	}

	sizeDims, err := graph.Constant("size_dims", []int32{299, 299})
	if err != nil {
		log.Fatal("Problem adding a constant to the graph, Error:", err)
	}

	size, err := graph.AddOp("ResizeBilinear", "size", []*tf.GraphNode{dimsExpander, sizeDims}, "", map[string]interface{}{
		"T": tf.DtFloat,
	})
	if err != nil {
		log.Fatal("Problem adding ResizeBilinear operation, Error:", err)
	}

	inputMean, err := graph.Constant("input_mean", float32(128))
	if err != nil {
		log.Fatal("Problem adding the input mean the graph, Error:", err)
	}
	subMean, err := graph.AddOp("Sub", "sub_mean", []*tf.GraphNode{size, inputMean}, "", nil)
	if err != nil {
		log.Fatal("Problem adding substractinv the mean, Error:", err)
	}

	inputStd, err := graph.Constant("input_std", float32(128))
	if err != nil {
		log.Fatal("Problem adding the input std the graph, Error:", err)
	}
	_, err = graph.AddOp("Div", "normalized", []*tf.GraphNode{subMean, inputStd}, "", nil)
	if err != nil {
		log.Fatal("Problem adding dividing std, Error:", err)
	}

	// Create the session and extend the Graph
	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		log.Fatal("Problem extending the Graph, Error:", err)
	}

	// Execute the graph with the two input tensors, and specify the names
	// of the tensors to be returned, on this case just one
	out, err := s.Run(nil, []string{"normalized"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the graph, Error:", err)
	}

	if len(out) != 1 {
		log.Fatalf("The expected number of outputs is 1 but: %d returned", len(out))
	}

	return out[0]
}

func getTopLabels(outputs *tf.Tensor, labels int32) (indexes, scores *tf.Tensor) {
	graph := tf.NewGraph()

	dims := make([]int64, outputs.NumDims())
	for i := 0; i < outputs.NumDims(); i++ {
		dims[i] = int64(outputs.Dim(i))
	}

	normalized := graph.AddPlaceholder("normalized_placeholder", tf.DtFloat, dims, nil)

	labelsConstant, err := graph.Constant("lables", labels)
	if err != nil {
		log.Fatal("Problem adding the constant for lables, Error:", err)
	}

	_, err = graph.AddOp("TopKV2", "normalized", []*tf.GraphNode{normalized, labelsConstant}, "", nil)
	if err != nil {
		log.Fatal("Problem adding TopKV2 for best matches, Error:", err)
	}

	// Create the session and extend the Graph
	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		log.Fatal("Problem extending the Graph, Error:", err)
	}

	input := map[string]*tf.Tensor{
		"normalized_placeholder": outputs,
	}

	out, err := s.Run(input, []string{"normalized:0", "normalized:1"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the graph, Error:", err)
	}

	return out[1], out[0]
}

func main() {
	if len(os.Args) < 2 {
		log.Println("Ussage:", os.Args[0], "<image_path>")
	}
	imagePath := os.Args[1]

	resizedTensor := readTensorFromImageFile(imagePath)

	graph, err := tf.LoadGraphFromFile("data/tensorflow_inception_graph.pb")
	if err != nil {
		log.Fatal("Problem trying read the graph from the origin file, Error:", err)
	}

	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		log.Fatal(err)
	}

	input := map[string]*tf.Tensor{
		"Mul": resizedTensor,
	}
	out, err := s.Run(input, []string{"softmax"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the saved graph, Error:", err)
	}

	indexTens, scoresTens := getTopLabels(out[0], cLabelsToShow)

	labelsStr, err := ioutil.ReadFile("data/imagenet_comp_graph_label_strings.txt")
	if err != nil {
		return
	}

	labels := strings.Split(string(labelsStr), "\n")

	for i := 0; i < cLabelsToShow; i++ {
		index, _ := indexTens.GetVal(0, i)
		score, _ := scoresTens.GetVal(0, i)
		fmt.Println(labels[index.(int32)], ":", score)
	}
}
