package tensorflow_test

import (
	"fmt"

	"github.com/tensorflow/tensorflow"
)

func ExampleNewSession() {
	s, err := tensorflow.NewSession()
	hello, _ := tensorflow.Constant("Hello, TensorFlow!")
	output, err := s.Run(hello)
	fmt.Println(output, err)
	// output:
	// <nil>
}
