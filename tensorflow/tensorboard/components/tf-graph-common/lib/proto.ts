/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * @fileoverview Interfaces that parallel proto definitions in
 * third_party/tensorflow/core/framework/...
 *     graph.proto
 *     step_stats.proto
 * These should stay in sync.
 */
module tf.graph.proto {
  /**
   * TensorFlow node definition as defined in the graph.proto file.
   */
  export interface NodeDef {
    /** Name of the node */
    name: string;
    /** List of nodes that are inputs for this node. */
    input: string[];
    /** The name of the device where the computation will run. */
    device: string;
    /** The name of the operation associated with this node. */
    op: string;
    /** List of attributes that describe/modify the operation. */
    attr: {key: string, value: Object}[];
  }

  /**
   * TensorFlow stats file definition as defined in the stats proto file.
   */
  export interface StepStats {
    dev_stats: {device: string, node_stats: NodeExecStats[]}[];
  }

  /**
   * TensorFlow stats for a node as defined in the step_stats proto file.
   */
  export interface NodeExecStats {
    node_name: string;
    // The next 4 properties are currently stored as string in json
    // and must be parsed.
    all_start_micros: number;
    op_start_rel_micros: number;
    op_end_rel_micros: number;
    all_end_rel_micros: number;
    memory: {
      allocator_name: string;
      total_bytes: number;  // Stored as string in json and should be parsed.
      peak_bytes: number;   // Stored as string in json and should be parsed.
    }[];
    /** Output sizes recorded for a single execution of a graph node */
    output: NodeOutput[];
    timeline_label: string;
    scheduled_micros: string;
    thread_id: string;
  }

  /**
   * Description for the output tensor(s) of an operation in the graph as
   * defined in the step_stats.proto file.
   */
  export interface NodeOutput {
    slot: number;  // Stored as string in json and should be parsed.
    tensor_description: {
      /** Data type of tensor elements */
      dtype: string;
      /** Shape of the tensor */
      shape: {
        /**
         * Dimensions of the tensor, such as [{name: 'input', size: 30},
         * {name: 'output', size: 40}] for a 30 x 40 2D tensor.  The names
         * are optional. The order of entries in 'dim' matters: It indicates
         * the layout of the values in the tensor in-memory representation.
         */
        dim: {
          /** Size of the tensor in that dimension */
          size: number,  // Stored as string in json and should be parsed.
          /** Optional name of the tensor dimension */
          name?: string
        }[];
      };
      /** Information about the size and allocator used for the data */
      allocation_description: {
        // The next 2 properties are stored as string in json and
        // should be parsed.
        /** Total number of bytes requested */
        requested_bytes: number;
        /** Total number of bytes allocated, if known */
        allocated_bytes?: number;
        /** Name of the allocator used */
        allocator_name: string;
      };
    };
  }
}
