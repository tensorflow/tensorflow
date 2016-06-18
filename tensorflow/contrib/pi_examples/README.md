# TensorFlow Raspberry Pi Examples

This folder contains examples of how to build applications for the Raspberry Pi using TensorFlow.

## Building the Examples

 - Follow the Raspberry Pi section of the instructions at [tensorflow/contrib/makefile](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile) to compile a static library containing the core TensorFlow code.

 - Install libjpeg, so we can load image files:

```
sudo apt-get install -y libjpeg-dev
```

 - To download the example model you'll need, run these commands:
 
```bash
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015_stripped.zip \
-o /tmp/inception_dec_2015_stripped.zip
unzip /tmp/inception_dec_2015_stripped.zip \
-d tensorflow/contrib/pi_examples/label_image/data/
```

 - From the root of the TensorFlow source tree, run `make -f tensorflow/contrib/pi_examples/label_image/Makefile` to build a basic example.

## Usage

Run `tensorflow/contrib/pi_examples/label_image/gen/bin/label_image` to try out image labeling with the default Grace Hopper image. You should several lines of output, with "Military Uniform" shown as the top result, something like this:

```bash
I tensorflow/contrib/pi_examples/label_image/label_image.cc:384] Running model succeeded!
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] military uniform (866): 0.624293
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] suit (794): 0.0473981
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] academic gown (896): 0.0280926
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] bolo tie (940): 0.0156956
I tensorflow/contrib/pi_examples/label_image/label_image.cc:284] bearskin (849): 0.0143348
```

Once you've verified that is working, you can supply your own images with `--image=your_image.jpg`, or even with graphs you've trained yourself with the TensorFlow for Poets tutorial using `--graph=your_graph.pb --input=Mul:0 --output=softmax:0`.