## Tensorflow SavedModel to StableHLO (tf-to-stablehlo-translate)

### Description

This tool converts TensorFlow models (SavedModel or MLIR module) to StableHLO
MLIR modules, preserving model structure and signatures. It enables seamless
integration of TensorFlow models into MLIR-based compiler frameworks for further
optimization and deployment.

### Usage

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
