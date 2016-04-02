package main

import (
	"fmt"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	fileName := "/home/avidales/pit.jpg"

	graph := tf.NewGraph()

	if _, err := graph.Constant("file_name", []string{fileName}); err != nil {
		log.Fatal("Problem adding the input file name to the graph, Error:", err)
	}

	if err := graph.AddOp("ReadFile", "file_reader", []string{"file_name"}, "", nil); err != nil {
		log.Fatal("Problem adding ReadFile operation, Error:", err)
	}

	if err := graph.AddOp("DecodeJpeg", "jpeg_reader", []string{"file_reader"}, "", map[string]interface{}{
		"channels": int64(3),
	}); err != nil {
		log.Fatal("Problem adding DecodeJpeg operation, Error:", err)
	}

	if err := graph.AddOp("Cast", "float_caster", []string{"jpeg_reader"}, "", map[string]interface{}{
		"SrcT": tf.DtFloat,
		"DstT": tf.DtFloat,
	}); err != nil {
		log.Fatal("Problem adding Cast operation, Error:", err)
	}

	if _, err := graph.Constant("dim_index", []int32{0}); err != nil {
		log.Fatal("Problem adding a constant to the graph, Error:", err)
	}
	if err := graph.AddOp("ExpandDims", "dims_expander", []string{"float_caster", "dim_index"}, "", map[string]interface{}{
		"T":   tf.DtFloat,
		"dim": 0,
	}); err != nil {
		log.Fatal("Problem adding ExpandDims operation, Error:", err)
	}

	if _, err := graph.Constant("size_dims", []int32{299, 299}); err != nil {
		log.Fatal("Problem adding a constant to the graph, Error:", err)
	}

	if err := graph.AddOp("ResizeBilinear", "size", []string{"dims_expander", "size_dims"}, "", map[string]interface{}{
		"T": tf.DtFloat,
	}); err != nil {
		log.Fatal("Problem adding ResizeBilinear operation, Error:", err)
	}

	// Create the session and extend the Graph
	s, err := tf.NewSession()
	if err := s.ExtendGraph(graph); err != nil {
		log.Fatal("Problem extending the Graph, Error:", err)
	}

	// Execute the graph with the two input tensors, and specify the names
	// of the tensors to be returned, on this case just one
	out, err := s.Run(nil, []string{"size"}, nil)
	if err != nil {
		log.Fatal("Problem trying to run the saved graph, Error:", err)
	}

	if len(out) != 1 {
		log.Fatalf("The expected number of outputs is 1 but: %d returned", len(out))
	}

	outputTensor := out[0]

	fmt.Println(outputTensor)
}
