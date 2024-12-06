# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python module for MLIR functions exported by pybind11."""

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
from pathlib import Path
import logging

def import_graphdef(
    graphdef: str,
    pass_pipeline: str,
    show_debug_info: bool,
    input_names: list = None,
    input_data_types: list = None,
    input_data_shapes: list = None,
    output_names: list = None,
) -> str:
    """
    Imports a TensorFlow GraphDef into MLIR.

    Args:
        graphdef (str): Serialized GraphDef object to import.
        pass_pipeline (str): The pass pipeline to run during import.
        show_debug_info (bool): Whether to show debug information.
        input_names (list, optional): Names of the input nodes.
        input_data_types (list, optional): Data types of the inputs.
        input_data_shapes (list, optional): Shapes of the input data.
        output_names (list, optional): Names of the output nodes.

    Returns:
        str: MLIR representation of the graph.

    Raises:
        ValueError: If inputs are invalid.
    """
    if output_names is None:
        output_names = []
    
    # Input validation
    if not isinstance(graphdef, (str, bytes)):
        raise ValueError("graphdef must be a string or bytes object")
    
    if not isinstance(pass_pipeline, str):
        raise ValueError("pass_pipeline must be a string")
    
    if input_names is not None and not isinstance(input_names, list):
        raise ValueError("input_names must be a list of strings")
    
    if show_debug_info:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(f"Importing GraphDef with inputs: {input_names}, types: {input_data_types}, shapes: {input_data_shapes}")
    
    if input_names is not None:
        return ImportGraphDef(
            str(graphdef).encode('utf-8'),
            pass_pipeline.encode('utf-8'),
            show_debug_info,
            ','.join(input_names).encode('utf-8'),
            ','.join(input_data_types).encode('utf-8'),
            ':'.join(input_data_shapes).encode('utf-8'),
            ','.join(output_names).encode('utf-8'),
        )
    return ImportGraphDef(
        str(graphdef).encode('utf-8'),
        pass_pipeline.encode('utf-8'),
        show_debug_info,
    )


def import_function(concrete_function, pass_pipeline, show_debug_info):
    """
    Imports a TensorFlow ConcreteFunction into MLIR.
    
    Args:
        concrete_function: The TensorFlow function to import.
        pass_pipeline: The pass pipeline to run during import.
        show_debug_info: Whether to show debug information.

    Returns:
        MLIR representation of the function.
    """
    ctxt = context.context()
    ctxt.ensure_initialized()
    return ImportFunction(
        ctxt._handle,
        str(concrete_function.function_def).encode('utf-8'),
        pass_pipeline.encode('utf-8'),
        show_debug_info,
    )


def experimental_convert_saved_model_to_mlir(
    saved_model_path: Path, exported_names: str, show_debug_info: bool
) -> str:
    """
    Converts a TensorFlow SavedModel to MLIR.

    Args:
        saved_model_path (Path): The path to the saved model directory.
        exported_names (str): Names of the functions to export.
        show_debug_info (bool): Whether to show debug information.

    Returns:
        str: MLIR representation of the saved model.
    """
    saved_model_path = Path(saved_model_path)  # Ensure Path object
    return ExperimentalConvertSavedModelToMlir(
        str(saved_model_path).encode('utf-8'),
        str(exported_names).encode('utf-8'),
        show_debug_info,
    )


def experimental_convert_saved_model_v1_to_mlir_lite(
    saved_model_path: Path, exported_names: str, tags: str, upgrade_legacy: bool, show_debug_info: bool
) -> str:
    """
    Converts a TensorFlow v1 SavedModel to MLIR Lite.

    Args:
        saved_model_path (Path): The path to the saved model directory.
        exported_names (str): Names of the functions to export.
        tags (str): Tags used to locate the graph in the saved model.
        upgrade_legacy (bool): Whether to upgrade legacy operations.
        show_debug_info (bool): Whether to show debug information.

    Returns:
        str: MLIR representation of the saved model.
    """
    saved_model_path = Path(saved_model_path)
    return ExperimentalConvertSavedModelV1ToMlirLite(
        str(saved_model_path).encode('utf-8'),
        str(exported_names).encode('utf-8'),
        str(tags).encode('utf-8'),
        upgrade_legacy,
        show_debug_info,
    )


def experimental_run_pass_pipeline(mlir_txt: str, pass_pipeline: str, show_debug_info: bool) -> str:
    """
    Runs a pass pipeline on the MLIR text.

    Args:
        mlir_txt (str): MLIR text to run passes on.
        pass_pipeline (str): Pass pipeline to apply.
        show_debug_info (bool): Whether to show debug information.

    Returns:
        str: The MLIR after passes have been applied.
    """
    return ExperimentalRunPassPipeline(
        mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info
    )


def experimental_write_bytecode(filename: Path, mlir_txt: str):
    """
    Writes the MLIR text to a bytecode file.

    Args:
        filename (Path): The file to write the bytecode to.
        mlir_txt (str): MLIR text to convert to bytecode.

    Returns:
        None
    """
    filename = Path(filename)
    return ExperimentalWriteBytecode(filename.encode('utf-8'), mlir_txt.encode())


def experimental_tflite_to_tosa_bytecode(
    flatbuffer: str,
    bytecode: str,
    use_external_constant: bool = False,
    ordered_input_arrays: list = None,
    ordered_output_arrays: list = None,
):
    """
    Converts a TensorFlow Lite model to TOSA bytecode.

    Args:
        flatbuffer (str): The TFLite flatbuffer to convert.
        bytecode (str): The output bytecode path.
        use_external_constant (bool): Whether to use external constants.
        ordered_input_arrays (list): Ordered input arrays (optional).
        ordered_output_arrays (list): Ordered output arrays (optional).

    Returns:
        None
    """
    if ordered_input_arrays is None:
        ordered_input_arrays = []
    if ordered_output_arrays is None:
        ordered_output_arrays = []

    return ExperimentalTFLiteToTosaBytecode(
        flatbuffer.encode('utf-8'),
        bytecode.encode('utf-8'),
        use_external_constant,
        ordered_input_arrays,
        ordered_output_arrays,
    )


def batch_import_graphdef(graphdefs: list, pass_pipeline: str, show_debug_info: bool) -> list:
    """
    Batch imports multiple GraphDefs into MLIR.

    Args:
        graphdefs (list): A list of GraphDef objects to import.
        pass_pipeline (str): The pass pipeline to apply to each GraphDef.
        show_debug_info (bool): Whether to show debug information.

    Returns:
        list: A list of MLIR representations of the graphs.
    """
    mlir_outputs = []
    for graphdef in graphdefs:
        mlir_outputs.append(import_graphdef(graphdef, pass_pipeline, show_debug_info))
    return mlir_outputs
