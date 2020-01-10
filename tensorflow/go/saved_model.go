/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package tensorflow

import (
	"runtime"
	"unsafe"

	"github.com/golang/protobuf/proto"

	tfpb "github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/github.com/tensorflow/tensorflow/tensorflow/go/core"
)

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

// SavedModel represents the contents of loaded SavedModel.
// TODO(jhseu): Add and document metagraphdef when we pregenerate protobufs.
type SavedModel struct {
	Session    *Session
	Graph      *Graph
	Signatures map[string]Signature
}

// LoadSavedModel creates a new SavedModel from a model previously
// exported to a directory on disk.
//
// Exported models contain a set of graphs and, optionally, variable values.
// Tags in the model identify a single graph. LoadSavedModel initializes a
// session with the identified graph and with variables initialized to from the
// checkpoints on disk.
//
// The tensorflow package currently does not have the ability to export a model
// to a directory from Go. This function thus currently targets loading models
// exported in other languages, such as using tf.saved_model.builder in Python.
// See:
// https://www.tensorflow.org/code/tensorflow/python/saved_model/
func LoadSavedModel(exportDir string, tags []string, options *SessionOptions) (*SavedModel, error) {
	status := newStatus()
	cOpt, doneOpt, err := options.c()
	defer doneOpt()
	if err != nil {
		return nil, err
	}
	cExportDir := C.CString(exportDir)
	cTags := make([]*C.char, len(tags))
	for i := range tags {
		cTags[i] = C.CString(tags[i])
	}
	graph := NewGraph()
	metaGraphDefBuf := C.TF_NewBuffer()
	defer C.TF_DeleteBuffer(metaGraphDefBuf)
	// TODO(jhseu): Add support for run_options and meta_graph_def.
	cSess := C.TF_LoadSessionFromSavedModel(cOpt, nil, cExportDir, (**C.char)(unsafe.Pointer(&cTags[0])), C.int(len(cTags)), graph.c, metaGraphDefBuf, status.c)
	for i := range cTags {
		C.free(unsafe.Pointer(cTags[i]))
	}
	C.free(unsafe.Pointer(cExportDir))

	metaGraphDefBytes := C.GoBytes(metaGraphDefBuf.data, C.int(metaGraphDefBuf.length))
	metaGraphDef := new(tfpb.MetaGraphDef)
	if err := proto.Unmarshal(metaGraphDefBytes, metaGraphDef); err != nil {
		return nil, err
	}

	signatures := generateSignatures(metaGraphDef.GetSignatureDef())

	if err := status.Err(); err != nil {
		return nil, err
	}
	s := &Session{c: cSess}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return &SavedModel{Session: s, Graph: graph, Signatures: signatures}, nil
}

func generateSignatures(pb map[string]*tfpb.SignatureDef) map[string]Signature {
	signatures := make(map[string]Signature)
	for name, signature := range pb {
		signatures[name] = signatureDefFromProto(signature)
	}
	return signatures
}
