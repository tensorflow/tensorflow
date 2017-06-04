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

// Command generates a Go source files with functions for TensorFlow ops.
package main

import (
	"bytes"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/tensorflow/tensorflow/tensorflow/go/genop/wrap"
)

func main() {
	var (
		dir           = flag.String("dir", "", "Dir to store generated files. Required")
		headerFile    = flag.String("headerFile", "", "Path to a file whose contents will be copied into the generated files. Required")
		headerFileBuf bytes.Buffer
	)
	flag.Parse()
	if *dir == "" {
		log.Fatal("-dir must be set")
	}
	if *headerFile == "" {
		log.Fatal("-headerFile must be set")
	}
	if *headerFile != "" {
		hdr, err := ioutil.ReadFile(*headerFile)
		if err != nil {
			log.Fatalf("Unable to read %s: %v", *headerFile, err)
		}
		headerFileBuf.Write(hdr)
		headerFileBuf.WriteString("\n\n")
	}

	os.MkdirAll(filepath.Dir(*dir), 0755)

	directory := *dir

	if err := wrap.GenerateFunctionsForRegisteredOps(directory, headerFileBuf); err != nil {
		log.Fatal(err)
	}

}
