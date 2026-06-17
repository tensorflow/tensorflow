# Tensorflow SavedModel to StableHLO (tf-to-stablehlo-translate)

Converts TensorFlow models (SavedModel or MLIR module) to StableHLO MLIR
modules, preserving model structure and signatures. It enables seamless
integration of TensorFlow models into MLIR-based compiler frameworks for further
optimization and deployment.

## C++ APIs

```bash
tf-to-stablehlo-translate \
    --input-path=/path/to/model \
    [--exported-model-signatures=signature1,signature2] \
    [--tag-names=tag1,tag2] \
    [--input-arg-shapes-str=arg-name:shape,...] \
    [--e] \
    [--output-filename=/path/to/output.mlir]
```

* `--input-path`: The path to the input TensorFlow SavedModel or MLIR module
  with .mlir extension.
* `--exported-model-signatures`: Comma-separated list of exported model
  signatures to convert. Ignored for MLIR input.
* `--tags`: Comma-separated list of tags for loading SavedModel. Ignored for
  MLIR input.
* `--input-arg-shapes`: A string representation of input argument shapes for
  'main' entry-point, separating tensors with ':', dimension with ',', and
  using '?' for unknown sizes. For example, `input-arg-shapes=1,2::1,?`
  expresses argument shapes `[1,2]`, `[]` and `[1,?]`.
* `--e`: Elide large elements attrs while dumping the output StableHLO.
* `--output_filename`: Path to the output file where the textual StableHLO MLIR
  module will be written (default: stdout).


### Examples

* To convert [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
model to StableHLO with static input shape `4x3x224x224` for input argument with
type `tensor<?x3x224x224xf32>`.

```bash
tf-to-stablehlo-translate <saved-model-path> --input-arg-shapes=4,3,224,224
```

* To convert
[google-bert/bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased)
to StableHLO with static input shapes `1x12`, `1x12`, and `1x12` for input
arguments with types `tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>`.

```bash
tf-to-stablehlo-translate <saved-model-path> --input-arg-shapes=1,12:1,12:1,12
```

### Dependencies

* TensorFlow
* MLIR
* Abseil (absl)

## Python APIs


### `savedmodel_to_stablehlo`

Converts a TensorFlow SavedModel into StableHLO bytecode.

```Python
from tensorflow.compiler.mlir.tensorflow_to_stablehlo.python import pywrap_tensorflow_to_stablehlo as tf2shlo

stablehlo_bytes = tf2shlo.savedmodel_to_stablehlo(
        input_path="/path/to/your/savedmodel",
        exported_model_signatures=["serving_default"],
        tag_names=["serve"],
        input_arg_shapes_str="1,28,28,3::32"
)

```

#### Arguments:

* `input_path` (required): Path to your SavedModel directory.
* `exported_model_signatures` (optional): List of signature names to convert.
                                          Defaults to ["serving_default"].
* `tag_names` (optional): List of tags associated with the SavedModel. Defaults
                          to ["serve"].
* `input_arg_shapes_str` (optional): A string representation of input argument
                                     shapes for 'main' entry-point, separating
                                     tensors with ':', dimension with ',', and
                                     using '?' for unknown sizes. For example,
                                     `input-arg-shapes=1,2::1,?` expresses
                                     argument shapes `[1,2], [] and [1,?]`.

#### Error Handling

An exception will be raised with details about the error.

### `tensorflow_module_to_stablehlo`

Converts a TensorFlow MLIR module string into StableHLO bytecode.

```Python
from tensorflow.compiler.mlir.tensorflow_to_stablehlo.python import pywrap_tensorflow_to_stablehlo as tf2shlo

stablehlo_bytes = tf2shlo.tensorflow_module_to_stablehlo(
    module_op_str="your_tensorflow_mlir_module_string",
    input_arg_shapes_str="1,28,28,3::32"
)
```

#### Arguments:

* `module_op_str` (required): String containing the TensorFlow MLIR module.
* `input_arg_shapes_str` (optional): A string representation of input argument
                                     shapes for 'main' entry-point, separating
                                     tensors with ':', dimension with ',', and
                                     using '?' for unknown sizes. For example,
                                     `input-arg-shapes=1,2::1,?` expresses
                                     argument shapes `[1,2], [] and [1,?]`.

#### Error Handling

Return `py::none()` (equivalent to Python's `None`) if there's an error. An
exception will be raised with details about the error.
