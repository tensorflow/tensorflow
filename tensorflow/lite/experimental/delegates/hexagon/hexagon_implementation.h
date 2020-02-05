/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_IMPLEMENTATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_IMPLEMENTATION_H_

#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn_interface.h"

namespace tflite {
// Holds the methods to use to Construct/Execute NN graph using Hexagon NNLib.
struct HexagonNN {
  // Call this function before creating a graph. It allows the environment on
  // the DSP to configure some settings.
  hexagon_nn_config_fn* hexagon_nn_config;

  //   Creates a new graph and returns an identifier to refer to the new graph.
  //   After a graph is
  // initialized, nodes can be added to it.
  // The returned graph is empty and cannot be executed until all nodes have
  // been added and the graph is finalized with hexagon_nn_prepare(). Multiple
  // graphs can be created and can be kept alive in the DSP environment
  // simultaneously.
  hexagon_nn_init_fn* hexagon_nn_init;

  // Provides a simple parameter between 0 and 255 to control the power saving
  // mode.
  // A level of 255 indicates that preference should be given to minimizing
  // power consumption. A level of 0 indicates that preference should be given
  // to executing as fast as possible.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_set_powersave_level_fn* hexagon_nn_set_powersave_level;

  // Changes the debug verbosity level for messages.
  hexagon_nn_set_debug_level_fn* hexagon_nn_set_debug_level;

  // Prepares a network for execution.
  // This function is required after all the nodes have been appended and before
  // execution.
  // This call provides a hook where memory can be allocated, data
  // can be rearranged, inputs and outputs can be linked up, and things in the
  // graph can be optimized.
  // Once a network has been prepared, it can no longer
  // be appended to, but it can be executed.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_prepare_fn* hexagon_nn_prepare;

  // Adds an ordinary (non-constant) node to the graph.
  // Non-constant nodes can have zero or more inputs and zero or more outputs.
  // An input is described as a source node ID as well as an output index to
  // refer to which one of several outputs a node may have.
  // An output is described with a maximum size. The true size of an output can
  // be computed dynamically, but the caller must define the maximum amount of
  // data storage required by the output during node creation.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_append_node_fn* hexagon_nn_append_node;

  // Adds constant nodes to a graph.
  // Constant nodes produce a single output that can be connected to one graph
  // node input. Unique node_ids are required for referencing nodes when
  // connecting the graph (for example, specifying which outputs of earlier
  // nodes will be used as inputs to particular subsequent nodes). Node_ids are
  // selected by the caller, but node_id=0 and node_id>0xF0000000 are reserved.
  // Node_ids must be unique.
  // *** NOTE: On SDM835 and older targets,
  // hexagon_nn_append_const_node() will not work properly for arrays larger
  // than 32 MB. Instead, use hexagon_nn_append_empty_const_node_large_array(),
  // which expects the same arguments.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_append_const_node_fn* hexagon_nn_append_const_node;

  // Executes a network, with provided input data and returning output data.
  // Execution will fail if the network has not been prepared.
  // Input is provided to the INPUT node, and output is returned from the OUTPUT
  // node.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_execute_fn* hexagon_nn_execute;

  // Newer version of hexagon_nn_execute that utilizes hexagon_nn_tensordefs to
  // represent inputs & outputs. Executes a network with provided input tensors
  // and returns output tensors. Execution will fail if the network has not
  // been prepared.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_execute_new_fn* hexagon_nn_execute_new;

  // Tears down and frees an NN graph. This can be done at any time after
  // hexagon_nn_init(). After this function has been invoked, the nn_id id is
  // invalid.
  //
  // Returns 0 on success, otherwise failure.
  hexagon_nn_teardown_fn* hexagon_nn_teardown;

  hexagon_nn_snpprint_fn* hexagon_nn_snpprint;

  hexagon_nn_getlog_fn* hexagon_nn_getlog;

  hexagon_nn_get_perfinfo_fn* hexagon_nn_get_perfinfo;

  hexagon_nn_reset_perfinfo_fn* hexagon_nn_reset_perfinfo;

  hexagon_nn_op_id_to_name_fn* hexagon_nn_op_id_to_name;

  // Should be called once to shutdown DSP and cleanup.
  hexagon_nn_global_teardown_fn* hexagon_nn_global_teardown;

  // Should be called once to initialize DSP.
  hexagon_nn_global_init_fn* hexagon_nn_global_init;

  // Returns true if the device SoC is supported by hexagon library. False
  // Otherwise.
  hexagon_nn_is_device_supported_fn* hexagon_nn_is_device_supported;

  hexagon_nn_version_fn* hexagon_nn_version = nullptr;

  bool interface_loaded = false;
};

// Returns an instance of HexagonNN.
const HexagonNN* HexagonNNImplementation();

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_HEXAGON_HEXAGON_IMPLEMENTATION_H_
