## Training a model

The following document will walk you through the process of training your own
250 KB embedded vision model using scripts that are easy to run. You can use
either the [Visual Wake Words dataset](https://arxiv.org/abs/1906.05721) for
person detection, or choose one of the [80
categories from the MSCOCO dataset](http://cocodataset.org/#explore).

This model will take several days to train on a powerful machine with GPUs. We
recommend using a [Google Cloud Deep
Learning VM](https://cloud.google.com/deep-learning-vm/).

### Training framework choice

Keras is the recommended interface for building models in TensorFlow, but when
the person detector model was being created it didn't yet support all the
features we needed. For that reason, we'll be showing you how to train a model
using tf.slim, an older interface. It is still widely used but deprecated, so
future versions of TensorFlow may not support this approach. We hope to publish
Keras instructions in the future.

The model definitions for Slim are part of the
[TensorFlow models repository](https://github.com/tensorflow/models), so to get
started you'll need to download it from GitHub using a command like this:

```
! cd ~
! git clone https://github.com/tensorflow/models.git
```

The following guide is going to assume that you've done this from your home
directory, so the model repository code is at ~/models, and that all commands
are run from the home directory too unless otherwise noted. You can place the
repository somewhere else, but you'll need to update all references to it.

To use Slim, you'll need to make sure its modules can be found by Python, and
install one dependency. Here's how to do this in an iPython notebook:

```
! pip install contextlib2
import os
new_python_path = (os.environ.get("PYTHONPATH") or '') + ":models/research/slim"
%env PYTHONPATH=$new_python_path
```

Updating `PYTHONPATH` through an `EXPORT` statement like this only works for the
current Jupyter session, so if you're using bash directly, you should add it to
a persistent startup script, running something like this:

```
echo 'export PYTHONPATH=$PYTHONPATH:models/research/slim' >> ~/.bashrc
source ~/.bashrc
```

If you see import errors running the slim scripts, you should make sure the
`PYTHONPATH` is set up correctly, and that contextlib2 has been installed. You
can find more general information on tf.slim in the
[repository's
README](https://github.com/tensorflow/models/tree/master/research/slim).

### Building the dataset

In order to train a person detector model, we need a large collection of images
that are labeled depending on whether or not they have people in them. The
ImageNet one-thousand class data that's widely used for training image
classifiers doesn't include labels for people, but luckily the
[COCO dataset](http://cocodataset.org/#home) does. You can also download this
data without manually registering too, and Slim provides a convenient script to
grab it automatically:

```
! chmod +x models/research/slim/datasets/download_mscoco.sh
! bash models/research/slim/datasets/download_mscoco.sh coco
```

This is a large download, about 40GB, so it will take a while and you'll need
to make sure you have at least 100GB free on your drive to allow space for
unpacking and further processing. The argument to the script is the path that
the data will be downloaded to. If you change this, you'll also need to update
the commands below that use it.

The dataset is designed to be used for training models for localization, so the
images aren't labeled with the "contains a person", "doesn't contain a person"
categories that we want to train for. Instead each image comes with a list of
bounding boxes for all of the objects it contains. "Person" is one of these
object categories, so to get to the classification labels we want, we have to
look for images with bounding boxes for people. To make sure that they aren't
too tiny to be recognizable we also need to exclude very small bounding boxes.
Slim contains a script to convert the bounding box into labels:

```
! python models/research/slim/datasets/build_visualwakewords_data.py
--logtostderr \
--train_image_dir=coco/raw-data/train2014 \
--val_image_dir=coco/raw-data/val2014 \
--train_annotations_file=coco/raw-data/annotations/instances_train2014.json \
--val_annotations_file=coco/raw-data/annotations/instances_val2014.json \
--output_dir=coco/processed \
--small_object_area_threshold=0.005 \
--foreground_class_of_interest='person'
```

Don't be surprised if this takes up to twenty minutes to complete. When it's
done, you'll have a set of TFRecords in `coco/processed` holding the labeled
image information. This data was created by Aakanksha Chowdhery and is known as
the [Visual Wake Words dataset](https://arxiv.org/abs/1906.05721). It's designed
to be useful for benchmarking and testing embedded computer vision, since it
represents a very common task that we need to accomplish with tight resource
constraints. We're hoping to see it drive even better models for this and
similar tasks.

### Training the model

One of the nice things about using tf.slim to handle the training is that the
parameters you commonly need to modify are available as command line arguments,
so we can just call the standard `train_image_classifier.py` script to train
our model. You can use this command to build the model we use in the example:

```
! python models/research/slim/train_image_classifier.py \
    --train_dir=vww_96_grayscale \
    --dataset_name=visualwakewords \
    --dataset_split_name=train \
    --dataset_dir=coco/processed \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --train_image_size=96 \
    --input_grayscale=True \
    --save_summaries_secs=300 \
    --learning_rate=0.045 \
    --label_smoothing=0.1 \
    --learning_rate_decay_factor=0.98 \
    --num_epochs_per_decay=2.5 \
    --moving_average_decay=0.9999 \
    --batch_size=96 \
    --max_number_of_steps=1000000
```

This will take a couple of days on a single-GPU v100 instance to complete all
one-million steps, but you should be able to get a fairly accurate model after
a few hours if you want to experiment early.

-   The checkpoints and summaries will the saved in the folder given in the
    `--train_dir` argument, so that's where you'll have to look for the results.
-   The `--dataset_dir` parameter should match the one where you saved the
    TFRecords from the Visual Wake Words build script.
-   The architecture we'll be using is defined by the `--model_name` argument.
    The 'mobilenet_v1' prefix tells the script to use the first version of
    MobileNet. We did experiment with later versions, but these used more RAM
    for their intermediate activation buffers, so for now we kept with the
    original. The '025' is the depth multiplier to use, which mostly affects the
    number of weight parameters, this low setting ensures the model fits within
    250KB of Flash.
-   `--preprocessing_name` controls how input images are modified before they're
    fed into the model. The 'mobilenet_v1' version shrinks the width and height
    of the images to the size given in `--train_image_size` (in our case 96
    pixels since we want to reduce the compute requirements). It also scales the
    pixel values from 0 to 255 integers into -1.0 to +1.0 floating point numbers
    (though we'll be quantizing those after training).
-   The
    [HM01B0](https://himax.com.tw/products/cmos-image-sensor/image-sensors/hm01b0/)
    camera we're using on the SparkFun Edge board is monochrome, so to get the
    best results we have to train our model on black and white images too, so we
    pass in the `--input_grayscale` flag to enable that preprocessing.
-   The `--learning_rate`, `--label_smoothing`, `--learning_rate_decay_factor`,
    `--num_epochs_per_decay`, `--moving_average_decay` and `--batch_size` are
    all parameters that control how weights are updated during the training
    process. Training deep networks is still a bit of a dark art, so these exact
    values we found through experimentation for this particular model. You can
    try tweaking them to speed up training or gain a small boost in accuracy,
    but we can't give much guidance for how to make those changes, and it's easy
    to get combinations where the training accuracy never converges.
-   The `--max_number_of_steps` defines how long the training should continue.
    There's no good way to figure out this threshold in advance, you have to
    experiment to tell when the accuracy of the model is no longer improving to
    tell when to cut it off. In our case we default to a million steps, since
    with this particular model we know that's a good point to stop.

Once you start the script, you should see output that looks something like this:

```
INFO:tensorflow:global step 4670: loss = 0.7112 (0.251 sec/step)
I0928 00:16:21.774756 140518023943616 learning.py:507] global step 4670: loss =
0.7112 (0.251 sec/step)
INFO:tensorflow:global step 4680: loss = 0.6596 (0.227 sec/step)
I0928 00:16:24.365901 140518023943616 learning.py:507] global step 4680: loss =
0.6596 (0.227 sec/step)
```

Don't worry about the line duplication, this is just a side-effect of the way
TensorFlow log printing interacts with Python. Each line has two key bits of
information about the training process. The global step is a count of how far
through the training we are. Since we've set the limit as a million steps, in
this case we're nearly five percent complete. The steps per second estimate is
also useful, since you can use it to estimate a rough duration for the whole
training process. In this case, we're completing about four steps a second, so
a million steps will take about 70 hours, or three days. The other crucial
piece of information is the loss. This is a measure of how close the
partially-trained model's predictions are to the correct values, and lower
values are better. This will show a lot of variation but should on average
decrease during training if the model is learning. Because it's so noisy, the
amounts will bounce around a lot over short time periods, but if things are
working well you should see a noticeable drop if you wait an hour or so and
check back. This kind of variation is a lot easier to see in a graph, which is
one of the main reasons to try TensorBoard.

### TensorBoard

TensorBoard is a web application that lets you view data visualizations from
TensorFlow training sessions, and it's included by default in most cloud
instances. If you're using Google Cloud's AI Platform, you can start up a new
TensorBoard session by open the command palette from the left tabs on the
notebook interface, and scrolling down to select "Create a new tensorboard".
You'll be prompted for the location of the summary logs, enter the path you
used for `--train_dir` in the training script, in our example
'vww_96_grayscale'. One common error to watch out for is adding a slash to the
end of the path, which will cause tensorboard to fail to find the directory. If
you're starting tensorboard from the command line in a different environment
you'll have to pass in this path as the `--logdir` argument to the tensorboard
command line tool, and point your browser to http://localhost:6006 (or the
address of the machine you're running it on).

It may take a little while for the graphs to have anything useful in them, since
the script only saves summaries every five minutes. The most important graph is
called 'clone_loss', and this shows the progression of the same loss value
that's displayed on the logging output. It fluctuates a lot, but the
overall trend is downwards over time. If you don't see this sort of progression
after a few hours of training, it's a good sign that your model isn't
converging to a good solution, and you may need to debug what's going wrong
either with your dataset or the training parameters.

Tensorboard defaults to the 'Scalars' tab when it opens, but the other section
that can be useful during training is 'Images'. This shows a
random selection of the pictures the model is currently being trained on,
including any distortions and other preprocessing. This information isn't as
essential as the loss graphs, but it can be useful to ensure the dataset is what
you expect, and it is interesting to see the examples updating as training
progresses.

### Evaluating the model

The loss function correlates with how well your model is training, but it isn't
a direct, understandable metric. What we really care about is how many people
our model detects correctly, but to get calculate this we need to run a
separate script. You don't need to wait until the model is fully trained, you
can check the accuracy of any checkpoints in the `--train_dir` folder.

```
! python models/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=vww_96_grayscale/model.ckpt-698580 \
    --dataset_dir=coco/processed/ \
    --dataset_name=visualwakewords \
    --dataset_split_name=val \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --input_grayscale=True \
    --train_image_size=96
```

You'll need to make sure that `--checkpoint_path` is pointing to a valid set of
checkpoint data. Checkpoints are stored in three separate files, so the value
should be their common prefix. For example if you have a checkpoint file called
'model.ckpt-5179.data-00000-of-00001', the prefix would be 'model.ckpt-5179'.
The script should produce output that looks something like this:

```
INFO:tensorflow:Evaluation [406/406]
I0929 22:52:59.936022 140225887045056 evaluation.py:167] Evaluation [406/406]
eval/Accuracy[0.717438412]eval/Recall_5[1]
```

The important number here is the accuracy. It shows the proportion of the
images that were classified correctly, which is 72% in this case, after
converting to a percentage. If you follow the example script, you should expect
a fully-trained model to achieve an accuracy of around 84% after one million
steps, and show a loss of around 0.4.

### Exporting the model to TensorFlow Lite

When the model has trained to an accuracy you're happy with, you'll need to
convert the results from the TensorFlow training environment into a form you
can run on an embedded device. As we've seen in previous chapters, this can be
a complex process, and tf.slim adds a few of its own wrinkles too.

#### Exporting to a GraphDef protobuf file

Slim generates the architecture from the model_name every time one of its
scripts is run, so for a model to be used outside of Slim it needs to be saved
in a common format. We're going to use the GraphDef protobuf serialization
format, since that's understood by both Slim and the rest of TensorFlow.

```
! python models/research/slim/export_inference_graph.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --model_name=mobilenet_v1_025 \
    --image_size=96 \
    --input_grayscale=True \
    --output_file=vww_96_grayscale_graph.pb
```

If this succeeds, you should have a new 'vww_96_grayscale_graph.pb' file in
your home folder. This contains the layout of the operations in the model, but
doesn't yet have any of the weight data.

#### Freezing the weights

The process of storing the trained weights together with the operation graph is
known as freezing. This converts all of the variables in the graph to
constants, after loading their values from a checkpoint file. The command below
uses a checkpoint from the millionth training step, but you can supply any
valid checkpoint path. The graph freezing script is stored inside the main
tensorflow repository, so we have to download this from GitHub before running
this command.

```
! git clone https://github.com/tensorflow/tensorflow
! python tensorflow/tensorflow/python/tools/freeze_graph.py \
--input_graph=vww_96_grayscale_graph.pb \
--input_checkpoint=vww_96_grayscale/model.ckpt-1000000 \
--input_binary=true --output_graph=vww_96_grayscale_frozen.pb \
--output_node_names=MobilenetV1/Predictions/Reshape_1
```

After this, you should see a file called 'vww_96_grayscale_frozen.pb'.

#### Quantizing and converting to TensorFlow Lite

Quantization is a tricky and involved process, and it's still very much an
active area of research, so taking the float graph that we've trained so far
and converting it down to eight bit takes quite a bit of code. You can find
more of an explanation of what quantization is and how it works in the chapter
on latency optimization, but here we'll show you how to use it with the model
we've trained. The majority of the code is preparing example images to feed
into the trained network, so that the ranges of the activation layers in
typical use can be measured. We rely on the TFLiteConverter class to handle the
quantization and conversion into the TensorFlow Lite flatbuffer file that we
need for the inference engine.

```
import tensorflow as tf
import io
import PIL
import numpy as np

def representative_dataset_gen():

  record_iterator =
tf.python_io.tf_record_iterator(path='coco/processed/val.record-00000-of-00010')

  count = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    image_stream =
io.BytesIO(example.features.feature['image/encoded'].bytes_list.value[0])
    image = PIL.Image.open(image_stream)
    image = image.resize((96, 96))
    image = image.convert('L')
    array = np.array(image)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    array = ((array / 127.5) - 1.0).astype(np.float32)
    yield([array])
    count += 1
    if count > 300:
        break

converter =
tf.lite.TFLiteConverter.from_frozen_graph('vww_96_grayscale_frozen.pb',
['input'], ['MobilenetV1/Predictions/Reshape_1'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
open("vww_96_grayscale_quantized.tflite", "wb").write(tflite_quant_model)
```

#### Converting into a C source file

The converter writes out a file, but most embedded devices don't have a file
system. To access the serialized data from our program, we have to compile it
into the executable and store it in Flash. The easiest way to do that is to
convert the file into a C data array.

```
# Install xxd if it is not available
!apt-get -qq install xxd
# Save the file as a C source file
!xxd -i vww_96_grayscale_quantized.tflite > person_detect_model_data.cc
```

You can now replace the existing person_detect_model_data.cc file with the
version you've trained, and be able to run your own model on embedded devices.

### Training for other categories

There are over 60 different object types in the MS-COCO dataset, so an easy way
to customize your model would be to choose one of those instead of 'person'
when you build the training dataset. Here's an example that looks for cars:

```
! python models/research/slim/datasets/build_visualwakewords_data.py
--logtostderr \
--train_image_dir=coco/raw-data/train2014 \
--val_image_dir=coco/raw-data/val2014 \
--train_annotations_file=coco/raw-data/annotations/instances_train2014.json \
--val_annotations_file=coco/raw-data/annotations/instances_val2014.json \
--output_dir=coco/processed_cars \
--small_object_area_threshold=0.005 \
--foreground_class_of_interest='car'
```

You should be able to follow the same steps you did for the person detector,
but substitute the new 'coco/processed_cars' path wherever 'coco/processed'
used to be.

If the kind of object you're interested in isn't present in MS-COCO, you may be
able to use transfer learning to help you train on a custom dataset you've
gathered, even if it's much smaller. We don't have an example of this
yet, but we hope to share one soon.

### Understanding the architecture

[MobileNets](https://arxiv.org/abs/1704.04861) are a family of architectures
designed to provide good accuracy for as few weight parameters and arithmetic
operations as possible. There are now multiple versions, but in our case we're
using the original v1 since it required the smallest amount of RAM at runtime.
The core concept behind the architecture is depthwise separable convolution.
This is a variant of classical two-dimensional convolutions that works in a
much more efficient way, without sacrificing very much accuracy. Regular
convolution calculates an output value based on applying a filter of a
particular size across all channels of the input. This means the number of
calculations involved in each output is width of the filter multiplied by
height, multiplied by the number of input channels. Depthwise convolution
breaks this large calculation into separate parts. First each input channel is
filtered by one or more rectangular filters to produce intermediate values.
These values are then combined using pointwise convolutions. This dramatically
reduces the number of calculations needed, and in practice produces similar
results to regular convolution.

MobileNet v1 is a stack of 14 of these depthwise separable convolution layers
with an average pool, then a fully-connected layer followed by a softmax at the
end. We've specified a 'width multiplier' of 0.25, which has the effect of
reducing the number of computations down to around 60 million per inference, by
shrinking the number of channels in each activation layer by 75% compared to
the standard model. In essence it's very similar to a normal convolutional
neural network in operation, with each layer learning patterns in the input.
Earlier layers act more like edge recognition filters, spotting low-level
structure in the image, and later layers synthesize that information into more
abstract patterns that help with the final object classification.
