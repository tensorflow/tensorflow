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

package op

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Scope encapsulates common properties of operations being added to a Graph.
//
// Scopes allow common properties (such as a name prefix) to be specified
// once for multiple operations being added to a graph. The With* methods
// create derivative scopes that encapsulate the same set of properties
// as the parent Scope, except for the one being changed by the specific
// With* method.
//
// Scopes are NOT safe for concurrent use by multiple goroutines.
type Scope struct {
	graph     *tf.Graph
	namemap   map[string]int
	namespace string
}

// NewScope creates a Scope initialized with an empty Graph.
func NewScope() *Scope {
	return &Scope{graph: tf.NewGraph(), namemap: make(map[string]int)}
}

// Graph returns the Graph which this Scope and its children are
func (s *Scope) Graph() *tf.Graph {
	return s.graph
}

// SubScope returns a new Scope which will cause all operations added to the
// graph to be namespaced with 'namespace'.  If namespace collides with an
// existing namespace within the scope, then a suffix will be added.
func (s *Scope) SubScope(namespace string) *Scope {
	namespace = s.uniqueName(namespace)
	if s.namespace != "" {
		namespace = s.namespace + "/" + namespace
	}
	return &Scope{
		graph:     s.graph,
		namemap:   make(map[string]int),
		namespace: namespace,
	}
}

func (s *Scope) uniqueName(name string) string {
	count := s.namemap[name]
	s.namemap[name]++
	if count == 0 {
		return name
	}
	return fmt.Sprint(name, "_", count)
}

func (s *Scope) opName(typ string) string {
	if s.namespace == "" {
		return typ
	}
	return s.namespace + "/" + typ
}
