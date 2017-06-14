# SavedModel CLI (Command-Line Interface)

[`SavedModel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)
is a universal serialization format for Tensorflow. It provides a
language-neutral format to save machine-learned models and enables higher-level
systems and tools to produce, consume and transform TensorFlow models.

We provide SavedModel CLI(command-line interface) as a tool to inspect and
execute a [`MetaGraph`](https://www.tensorflow.org/programmers_guide/meta_graph)
in a SavedModel. You can inspect for example, what
[`SignatureDefs`](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md),
including their input and output tensors, are in the model without writing any
code. This can be useful in situations such as when you want to quickly check
your input dtype and shape match with the model. Moreover, if you want to test
out the model, it also allows you to do a sanity check by passing in sample
inputs in the format of for example, python expressions, and fetch the outputs
simply through command line.

## Get SavedModel CLI

If TensorFlow is installed on your system through pip, the `saved_model_cli`
binary can be invoked directly from command line.

To build the binary from source, run the following command:

```
$bazel build tensorflow/python/tools:saved_model_cli
```

## Commands

SavedModel CLI allows users to both show and run computations on a
[`MetaGraphDef`](https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto)
in a SavedModel. These are done through `show` and `run` commands. We will
explain the usages of both commands with detailed examples. SavedModel CLI will
also display this information with `-h` option.

### `show` command

A SavedModel contains one or more MetaGraphs, identified by their tag-sets. Each
MetaGraph contains both a TensorFlow GraphDef as well as associated metadata
necessary for running computation in a graph. In order to serve a model, you
might wonder what kind of SignatureDefs are in each model, and what are their
inputs and outputs etc. The `show` command let you examine the content of the
SavedModel in a hierarchical order.

```
usage: saved_model_cli show [-h] --dir DIR [--all]
[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
```

#### Examples

To show all available MetaGraphDef tag-sets in the SavedModel:

```
$saved_model_cli show --dir /tmp/saved_model_dir
The given SavedModel contains the following tag-sets:
serve
serve, gpu
```

To show all available SignatureDef keys in a MetaGraphDef:

```
$saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
```

For a MetaGraphDef with multiple tags in the tag-set, all tags must be passed
in, separated by ',':

```
$saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
```

To show all inputs and outputs TensorInfo for a specific SignatureDef, pass in
the SignatureDef key to `signature_def` option. This is very useful when you
want to know the tensor key value, dtype and shape of the input tensors for
executing the computation graph later.

```
$saved_model_cli show --dir \
/tmp/saved_model_dir --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
inputs['x'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['y'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y:0
Method name is: tensorflow/serving/predict
```

To show all available information in the SavedModel, use `--all` option:

```
$saved_model_cli show --dir /tmp/saved_model_dir --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
The given SavedModel SignatureDef contains the following input(s):
inputs['inputs'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x2:0
The given SavedModel SignatureDef contains the following output(s):
outputs['scores'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y3:0
Method name is: tensorflow/serving/classify

...

signature_def['serving_default']:
The given SavedModel SignatureDef contains the following input(s):
inputs['x'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['y'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y:0
Method name is: tensorflow/serving/predict
```

### `run` command

SavedModel CLI also allows you to run a graph computation by passing in inputs,
displaying, and saving the outputs.

```
usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
                           SIGNATURE_DEF_KEY [--inputs INPUTS]
                           [--input_exprs INPUT_EXPRS] [--outdir OUTDIR]
                           [--overwrite] [--tf_debug]
```

Tensor keys are used to specify which input we are passing in the values for.
There are two ways to pass inputs to the model. With '--inputs' option, you can
pass in numpy ndarray by files. With '--input_exprs' option, you can use python
expressions as inputs.

#### Input By File

To pass in inputs by files, use `--inputs` option in the format of
`<input_key>=<filename>`, or `<input_key>=<filename>[<variable_name>]`. Each
input is separated by semicolon. File specified by `filename` will be loaded
using `numpy.load`. Inputs can be loaded from only `.npy`, `.npz` or pickle
files. The `variable_name` key is optional depending on the input file type as
descripted in more details below.

When loading from a `.npy` file, which always contains a numpy ndarray, the
content will be directly assigned to the specified input tensor. If a
`variable_name` is specified, it will be ignored and a warning will be issued.

When loading from a `.npz` zip file, user can specify which variable within the
zip file to load for the input tensor key with `variable_name`. If nothing is
specified, SavedModel CLI will check that only one file is included in the zip
file and load it for the specified input tensor key.

When loading from a pickle file, if no `variable_name` is specified in the
square brackets, whatever that is inside the pickle file will be passed to the
specified input tensor key. Else SavedModel CLI will assume a dictionary is
stored in the pickle file and the value corresponding to the variable_name will
be used.

#### Input By Python Expression

To pass in inputs by python expressions, use `--input_exprs` option. `numpy`
module is available as `np`. For example, `input_key=np.ones((32, 32, 3))` or
`input_key=[[1], [2], [3]]`. This can be useful for when you don't have data
files lying around, but still want to sanity check the model with some simple
inputs that match the dtype and shape of the model signature.

#### Save Output

By default, SavedModel CLI will print outputs to console. If a directory is
passed to `--outdir` option, the outputs will be saved as npy files named after
output tensor keys under the given directory. Use `--overwrite` to overwrite
existing output files.

#### TensorFlow Debugger (tfdbg) Integration

If `--tf_debug` option is set, SavedModel CLI will use TensorFlow Debugger
(tfdbg) to watch the intermediate Tensors and runtime GraphDefs while running
the SavedModel.

#### Examples

If we have a simple model that adds `x1` and `x2` to get output `y`, where all
tensors are of shape `(-1, 1)`, and we have two `npz` files. File
`/tmp/my_data1.npy` contains a numpy ndarray `[[1], [2], [3]]`, file
`/tmp/my_data2.npy` contains another numpy ndarray `[[0.5], [0.5], [0.5]]`. Now
let's run these two `npy` files through the model to get `y`:

```
$saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npy;x2=/tmp/my_data2.npy \
--outdir /tmp/out
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

Similarly, we can run input tensors from `npz` file and pickle file, as well as
overwrite the previous output file:

```
$saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y \
--inputs x1=/tmp/my_data1.npz[x];x2=/tmp/my_data2.pkl --outdir /tmp/out \
--overwrite
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

You can also use python expression instead of input file. Here we replace input
`x2` with a python expression:

```
$saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npz[x] \
--input_exprs 'x2=np.ones((3,1))'
Result for output key y:
[[ 2]
 [ 3]
 [ 4]]
```

To run model with TensorFlow Debugger on:

```
$saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def serving_default --inputs x=/tmp/data.npz[x] --tf_debug
```
