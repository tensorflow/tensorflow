# Benchmarker for LPIRC Workshop at CVPR 2018

This folder contains building code for track one of the [Low Power ImageNet Recognition Challenge workshop at CVPR 2018.](https://rebootingcomputing.ieee.org/home/sitemap/14-lpirc/80-low-power-image-recognition-challenge-lpirc-2018)

## Pre-requesits

Follow the steps [here](https://www.tensorflow.org/mobile/tflite/demo_android) to install Tensorflow, Bazel, and the Android NDK and SDK.

## To test the benchmarker:

The testing utilities helps the developers (you) to make sure that your submissions in TfLite format will be processed as expected in the competition's benchmarking system.

Note: for now the tests only provides correctness checks, i.e. classifier predicts the correct category on the test image, but no on-device latency measurements. To test the latency measurement functionality, the tests will print the latency running on a desktop computer, which is not indicative of the on-device run-time.
We are releasing an benchmarker Apk that would allow developers to measure latency on their own devices.

### Obtain the sample models

The test data (models and images) should be downloaded automatically for you by Bazel. In case they are not, you can manually install them as below.

Note: all commands should be called from your tensorflow installation folder (under this folder you should find `tensorflow/contrib/lite`).


* Download the [testdata package](https://storage.googleapis.com/download.tensorflow.org/data/ovic.zip):

```sh
curl -L https://storage.googleapis.com/download.tensorflow.org/data/ovic.zip -o /tmp/ovic.zip
```

* Unzip the package into the testdata folder:

```sh
unzip -j /tmp/ovic.zip -d tensorflow/contrib/lite/java/ovic/src/testdata/
```

### Run tests

You can run test with Bazel as below. This helps to ensure that the installation is correct.

```sh
bazel test --cxxopt=--std=c++11 //tensorflow/contrib/lite/java:OvicClassifierTest --test_output=all
```

### Test your submissions

Once you have a submission that follows the instructions from the [competition site](https://rebootingcomputing.ieee.org/home/sitemap/14-lpirc/80-low-power-image-recognition-challenge-lpirc-2018), you can verify it as below.

* Move your submission to the testdata folder:

Let say the submission file is located at `/tmp/my_model.lite`, then

```sh
cp /tmp/my_model.lite tensorflow/contrib/lite/java/ovic/src/testdata/
```

* Resize the test image to the resolutions that are expected by your submission:

The test images can be found at `tensorflow/contrib/lite/java/ovic/src/testdata/test_image_*.jpg`. You may reuse these images if your image resolutions are 128x128 or 224x224.

* Add your model and test image to the BUILD rule:

```JSON
java_test(
  name = "OvicClassifierTest",
  size = "medium",
  srcs = ["ovic/src/test/java/org/tensorflow/ovic/OvicClassifierTest.java"],
  data = [
      "ovic/src/testdata/float_model.lite",
      "ovic/src/testdata/labels.txt",
      "ovic/src/testdata/low_res_model.lite",
      "ovic/src/testdata/quantized_model.lite",
      "ovic/src/testdata/test_image_128.jpg",
      "ovic/src/testdata/test_image_224.jpg",
      "ovic/src/testdata/my_model.lite",        # <--- Your submission.
      "ovic/src/testdata/my_test_image.jpg",    # <--- Your test image.
  ],
      ...
```

* Modify `OvicClassifierTest.java` to test your model.

Change `TEST_IMAGE_PATH` to `testdata/my_test_image.jpg`. If your model runs inference in floating point, change `FLOAT_MODEL_PATH` to `testdata/my_model.lite`. If your model runs [quantized inference](https://www.tensorflow.org/performance/quantization), change `QUANTIZED_MODEL_PATH` to `testdata/my_model.lite`.

Now you can run the bazel tests to catch any runtime issues with the submission.
