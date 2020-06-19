# Adding metadata to TensorFlow Lite model

Note: TensorFlow Lite Metadata is in experimental (beta) phase.

TensorFlow Lite metadata provides a standard for model descriptions. The
metadata is an important source of knowledge about what the model does and its
input / output information. The metadata consists of both

*   human readable parts which convey the best practice when using the model,
    and
*   machine readable parts that can be leveraged by code generators, such as
    [the TensorFlow Lite Android code generator](../guide/codegen.md) and
    [the Android Studio ML Binding feature](https://developer.android.com/studio/preview/features#tensor-flow-lite-models).

## Setup the metadata tools

Before adding metadata to your model, you will need to a Python programming
environment setup for running TensorFlow. There is a detailed guide on how to
set this up [here](https://www.tensorflow.org/install).

After setup the Python programming environment, you will need to install
additional tooling:

```sh
pip install tflite-support
```

TensorFlow Lite metadata tooling supports both Python 2 and Python 3.

## Adding metadata

There are three parts to the
[model metadata](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs):

1.  **Model information** - Overall description of the model as well as items
    such as licence terms. See
    [ModelMetadata](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs#L515).
2.  **Input information** - Description of the inputs and pre-processing
    required such as normalization. See
    [SubGraphMetadata.input_tensor_metadata](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs#L500).
3.  **Output information** - Description of the output and post-processing
    required such as mapping to labels. See
    [SubGraphMetadata.output_tensor_metadata](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs#L509).

Since TensorFlow Lite only supports single subgraph at this point, the
[TensorFlow Lite code generator](../guide/codegen.md) and
[the Android Studio ML Binding feature](https://developer.android.com/studio/preview/features#tensor-flow-lite-models)
will use `ModelMetadata.name` and `ModelMetadata.description`, instead of
`SubGraphMetadata.name` and `SubGraphMetadata.description`, when displaying
metadata and generating code.

### Supported Input / Output types

TensorFlow Lite metadata for input and output are not designed with specific
model types in mind but rather input and output types. It does not matter what
the model functionally does, as long as the input and output types consists of
the following or a combination of the following, it is supported by TensorFlow
Lite metadata:

*   Feature - Numbers which are unsigned integers or float32.
*   Image - Metadata currently supports RGB and greyscale images.
*   Bounding box - Rectangular shape bounding boxes. The schema supports
    [a variety of numbering schemes](https://github.com/tensorflow/tensorflow/blob/268853ee81edab09e07f455cc918f7ef9a421485/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs#L165).

### Pack the associated files

TensorFlow Lite models may come with different associated files. For example,
natural language models usually have vocab files that map word pieces to word
IDs; classification models may have label files that indicate object categories.
Without the associated files (if there are), a model will not function well.

The associated files can now be bundled with the model through the metadata
Python library. The new TensorFlow Lite model becomes a zip file that contains
both the model and the associated files. It can be unpacked with common zip
tools. This new model format keeps using the same file extension, `.tflite`. It
is compatible with existing TFLite framework and Interpreter. See
[Pack mtadata and associated files into the model](#pack-metadata-and-associated-files-into-the-model)
for more details.

The associate file information can be recored in the metadata. Depending on the
file type and where the file is attached to (i.e. `ModelMetadata`,
`SubGraphMetadata`, and `TensorMetadata`),
[the TensorFlow Lite Android code generator](../guide/codegen.md) may apply
corresponding pre/post processing automatically to the object. See
[the \<Codegen usage\> section of each associate file type](https://github.com/tensorflow/tensorflow/blob/268853ee81edab09e07f455cc918f7ef9a421485/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs#L37-L77)
in the schema for more details.

### Examples

Note: The export directory specified has to exist before you run the script; it
does not get created as part of the process.

You can find examples on how the metadata should be populated for different
types of models here:

#### Image classification

Download the script
[here](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py)
, which populates metadata to
[mobilenet_v1_0.75_160_quantized.tflite](https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.75_160_quantized/1/default/1).
Run the script like this:

```sh
python ./metadata_writer_for_image_classifier.py \
    --model_file=./model_without_metadata/mobilenet_v1_0.75_160_quantized.tflite \
    --label_file=./model_without_metadata/labels.txt \
    --export_directory=model_with_metadata
```

To populate metadata for other image classification models, add the model specs
like
[this](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py#L63-L74)
into the script. The rest of this guide will highlight some of the key sections
in the image classification example to illustrate the key elements.

### Deep dive into the image classification example

#### Model information

Metadata starts by creating a new model info:

```python
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

""" ... """
"""Creates the metadata for an image classifier."""

# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "MobileNetV1 image classifier"
model_meta.description = ("Identify the most prominent object in the "
                          "image from a set of 1,001 categories such as "
                          "trees, animals, food, vehicles, person etc.")
model_meta.version = "v1"
model_meta.author = "TensorFlow"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")
```

#### Input / output information

This section shows you how to describe your model's input and output signature.
This metadata may be used by automatic code generators to create pre- and post-
processing code. To create input or output information about a tensor:

```python
# Creates input info.
input_meta = _metadata_fb.TensorMetadataT()

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
```

#### Image input

Image is a common input type for machine learning. TensorFlow Lite metadata
supports information such as colorspace and pre-processing information such as
normalization. The dimension of the image does not require manual specification
since it is already provided by the shape of the input tensor and can be
automatically inferred.

```python
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(160, 160))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5]
input_normalization.options.std = [127.5]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats
```

#### Label output

Label can be mapped to an output tensor via an associated file using
`TENSOR_AXIS_LABELS`.

```python
# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 1001 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename("your_path_to_label_file")
label_file.description = "Labels for objects that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]
```

#### Create the metadata Flatbuffers

The following code combines the model information with the input and output
information:

```python
# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()
```

#### Pack metadata and associated files into the model

Once the metadata Flatbuffers is created, the metadata and the label file are
written into the TFLite file via the `populate` method:

```python
populator = _metadata.MetadataPopulator.with_model_file(model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files(["your_path_to_label_file"])
populator.populate()
```

You can pack as many associated files as you want into the model through
`load_associated_files`. However, it is required to pack at least those files
documented in the metadata. In this example, packing the lable file is
mandatory.

### Visualize the metadata

You can use [Netron](https://github.com/lutzroeder/netron) to visualize your
metadata, or you can read the metadata from a TensorFlow Lite model into a json
format using the `MetadataDisplayer`:

```python
displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
export_json_file = os.path.join(FLAGS.export_directory,
                    os.path.splitext(model_basename)[0] + ".json")
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file
with open(export_json_file, "w") as f:
  f.write(json_file)
```
