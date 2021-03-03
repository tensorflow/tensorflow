# Gesture Recognition Magic Wand Training Scripts

## Introduction

The scripts in this directory can be used to train a TensorFlow model that
classifies gestures based on accelerometer data. The code uses Python 3.7 and
TensorFlow 2.0. The resulting model is less than 20KB in size.

The following document contains instructions on using the scripts to train a
model, and capturing your own training data.

This project was inspired by the [Gesture Recognition Magic Wand](https://github.com/jewang/gesture-demo)
project by Jennifer Wang.

## Training

### Dataset

Three magic gestures were chosen, and data were collected from 7
different people. Some random long movement sequences were collected and divided
into shorter pieces, which made up "negative" data along with some other
automatically generated random data.

The dataset can be downloaded from the following URL:

[download.tensorflow.org/models/tflite/magic_wand/data.tar.gz](http://download.tensorflow.org/models/tflite/magic_wand/data.tar.gz)

### Training in Colab

The following [Google Colaboratory](https://colab.research.google.com)
notebook demonstrates how to train the model. It's the easiest way to get
started:

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/magic_wand/train/train_magic_wand_model.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/magic_wand/train/train_magic_wand_model.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

If you'd prefer to run the scripts locally, use the following instructions.

### Running the scripts

Use the following command to install the required dependencies:

```shell
pip install -r requirements.txt
```

There are two ways to train the model:

- Random data split, which mixes different people's data together and randomly
  splits them into training, validation, and test sets
- Person data split, which splits the data by person

#### Random data split

Using a random split results in higher training accuracy than a person split,
but inferior performance on new data.

```shell
$ python data_prepare.py

$ python data_split.py

$ python train.py --model CNN --person false
```

#### Person data split

Using a person data split results in lower training accuracy but better
performance on new data.

```shell
$ python data_prepare.py

$ python data_split_person.py

$ python train.py --model CNN --person true
```

#### Model type

In the `--model` argument, you can provide `CNN` or `LSTM`. The CNN model has a
smaller size and lower latency.

## Collecting new data

To obtain new training data using the
[SparkFun Edge development board](https://sparkfun.com/products/15170), you can
modify one of the examples in the [SparkFun Edge BSP](https://github.com/sparkfun/SparkFun_Edge_BSP)
and deploy it using the Ambiq SDK.

### Install the Ambiq SDK and SparkFun Edge BSP

Follow SparkFun's
[Using SparkFun Edge Board with Ambiq Apollo3 SDK](https://learn.sparkfun.com/tutorials/using-sparkfun-edge-board-with-ambiq-apollo3-sdk/all)
guide to set up the Ambiq SDK and SparkFun Edge BSP.

#### Modify the example code

First, `cd` into
`AmbiqSuite-Rel2.2.0/boards/SparkFun_Edge_BSP/examples/example1_edge_test`.

##### Modify `src/tf_adc/tf_adc.c`

Add `true` in line 62 as the second parameter of function
`am_hal_adc_samples_read`.

##### Modify `src/main.c`

Add the line below in `int main(void)`, just before the line `while(1)`:

```cc
am_util_stdio_printf("-,-,-\r\n");
```

Change the following lines in `while(1){...}`

```cc
am_util_stdio_printf("Acc [mg] %04.2f x, %04.2f y, %04.2f z, Temp [deg C] %04.2f, MIC0 [counts / 2^14] %d\r\n", acceleration_mg[0], acceleration_mg[1], acceleration_mg[2], temperature_degC, (audioSample) );
```

to:

```cc
am_util_stdio_printf("%04.2f,%04.2f,%04.2f\r\n", acceleration_mg[0], acceleration_mg[1], acceleration_mg[2]);
```

#### Flash the binary

Follow the instructions in
[SparkFun's guide](https://learn.sparkfun.com/tutorials/using-sparkfun-edge-board-with-ambiq-apollo3-sdk/all#example-applications)
to flash the binary to the device.

#### Collect accelerometer data

First, in a new terminal window, run the following command to begin logging
output to `output.txt`:

```shell
$ script output.txt
```

Next, in the same window, use `screen` to connect to the device:

```shell
$ screen ${DEVICENAME} 115200
```

Output information collected from accelerometer sensor will be shown on the
screen and saved in `output.txt`, in the format of "x,y,z" per line.

Press the `RST` button to start capturing a new gesture, then press Button 14
when it ends. New data will begin with a line "-,-,-".

To exit `screen`, hit +Ctrl\\+A+, immediately followed by the +K+ key,
then hit the +Y+ key. Then run

```shell
$ exit
```

to stop logging data. Data will be saved in `output.txt`. For compatibility
with the training scripts, change the file name to include person's name and
the gesture name, in the following format:

```
output_{gesture_name}_{person_name}.txt
```

#### Edit and run the scripts

Edit the following files to include your new gesture names (replacing
"wing", "ring", and "slope")

- `data_load.py`
- `data_prepare.py`
- `data_split.py`

Edit the following files to include your new person names (replacing "hyw",
"shiyun", "tangsy", "dengyl", "jiangyh", "xunkai", "lsj", "pengxl", "liucx",
and "zhangxy"):

- `data_prepare.py`
- `data_split_person.py`

Finally, run the commands described earlier to train a new model.
