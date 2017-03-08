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

//go:generate sh generate.sh

// Command genop generates a Go source file with functions for TensorFlow ops.
package main

import (
	"flag"
	"log"
	"os"

	"github.com/tensorflow/tensorflow/tensorflow/go/genop/internal"
)

func main() {
	filename := flag.String("outfile", "", "File to write generated source code to.")
	flag.Parse()
	if *filename == "" {
		log.Fatal("--outfile must be set")
	}
	file, err := os.OpenFile(*filename, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		log.Fatalf("Failed to open %q for writing: %v", *filename, err)
	}
	defer file.Close()
	if err = internal.GenerateFunctionsForRegisteredOps(file); err != nil {
		log.Fatal(err)
	}
}
