/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//go:generate sh generate.sh

// Command genop generates a Go source file with functions for TensorFlow ops.
package main

import (
	"bytes"
	"flag"
	"go/format"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/tensorflow/tensorflow/tensorflow/go/genop/internal"
)

func main() {
	var (
		filename = flag.String("outfile", "", "File to write generated source code to.")
		header   = flag.String("header", "", "Path to a file whose contents will be copied into the generated file. Can be empty")
		buf      bytes.Buffer
	)
	flag.Parse()
	if *filename == "" {
		log.Fatal("-outfile must be set")
	}
	if *header != "" {
		hdr, err := ioutil.ReadFile(*header)
		if err != nil {
			log.Fatalf("Unable to read %s: %v", *header, err)
		}
		buf.Write(hdr)
		buf.WriteString("\n\n")
	}
	os.MkdirAll(filepath.Dir(*filename), 0755)

	if err := internal.GenerateFunctionsForRegisteredOps(&buf); err != nil {
		log.Fatal(err)
	}
	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatalf("Failed to generate valid source? 'go fmt' failed: %v", err)
	}
	if err := ioutil.WriteFile(*filename, formatted, 0644); err != nil {
		log.Fatalf("Failed to write to %q: %v", *filename, err)
	}
}
