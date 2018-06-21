# Benchmarker for LPIRC Workshop at CVPR 2018

This folder contains building code for track one of the [Low Power ImageNet Recognition Challenge workshop at CVPR 2018.](https://rebootingcomputing.ieee.org/home/sitemap/14-lpirc/80-low-power-image-recognition-challenge-lpirc-2018)

## Pre-requisite

Follow the steps [here](https://www.tensorflow.org/mobile/tflite/demo_android) to install Tensorflow, Bazel, and the Android NDK and SDK.

## Test the benchmarker:

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
bazel test --cxxopt=--std=c++11 //tensorflow/contrib/lite/java/ovic:OvicClassifierTest --cxxopt=-Wno-all --test_output=all
```

### Test your submissions

Once you have a submission that follows the instructions from the [competition site](https://rebootingcomputing.ieee.org/home/sitemap/14-lpirc/80-low-power-image-recognition-challenge-lpirc-2018), you can verify it in two ways:

#### Validate using randomly generated images

You can call the validator binary below to verify that your model fits the format requirements. This often helps you to catch size mismatches (e.g. output should be [1, 1001] instead of [1,1,1,1001]). Let say the submission file is located at `/path/to/my_model.lite`, then call:

```sh
bazel build --cxxopt=--std=c++11 //tensorflow/contrib/lite/java/ovic:ovic_validator --cxxopt=-Wno-all
bazel-bin/tensorflow/contrib/lite/java/ovic/ovic_validator /path/to/my_model.lite
```

Successful validation should print the following message to terminal:

```
Successfully validated /path/to/my_model.lite.

```

#### Test that the model produces sensible outcomes

You can go a step further to verify that the model produces results as expected. This helps you catch bugs during TOCO conversion (e.g. using the wrong mean and std values).

* Move your submission to the testdata folder:

```sh
cp /path/to/my_model.lite tensorflow/contrib/lite/java/ovic/src/testdata/
```

* Resize the test image to the resolutions that are expected by your submission:

The test images can be found at `tensorflow/contrib/lite/java/ovic/src/testdata/test_image_*.jpg`. You may reuse these images if your image resolutions are 128x128 or 224x224.

* Add your model and test image to the BUILD rule at `tensorflow/contrib/lite/java/ovic/src/testdata/BUILD`:

```JSON
filegroup(
    name = "ovic_testdata",
    srcs = [
        "@tflite_ovic_testdata//:float_model.lite",
        "@tflite_ovic_testdata//:low_res_model.lite",
        "@tflite_ovic_testdata//:quantized_model.lite",
        "@tflite_ovic_testdata//:test_image_128.jpg",
        "@tflite_ovic_testdata//:test_image_224.jpg"
        "my_model.lite",        # <--- Your submission.
        "my_test_image.jpg",    # <--- Your test image.
    ],
    ...
```

* Modify `OvicClassifierTest.java` to test your model.

Change `TEST_IMAGE_PATH` to `my_test_image.jpg`. Change either `FLOAT_MODEL_PATH` or `QUANTIZED_MODEL_PATH` to `my_model.lite` depending on whether your model runs inference in float or [8-bit](https://www.tensorflow.org/performance/quantization).

Now you can run the bazel tests to catch any runtime issues with the submission.

Note: Please make sure that your submission passes the test. If a submission fails to pass the test it will not be processed by the submission server.

## Measure on-device latency

We provide two ways to measure the on-device latency of your submission. The first is through our competition server, which is reliable and repeatable, but is limited to a few trials per day. The second is through the benchmarker Apk, which requires a device and may not be as accurate as the server, but has a fast turn-around and no access limitations. We recommend that the participants use the benchmarker apk for early development, and reserve the competition server for evaluating promising submissions.

### Running the benchmarker app

Make sure that you have followed instructions in [Test your submissions](#test-your-submissions) to add your model to the testdata folder and to the corresponding build rules.

Modify `tensorflow/contrib/lite/java/ovic/demo/app/OvicBenchmarkerActivity.java`:

* Add your model to the benchmarker apk by changing `MODEL_PATH` and `TEST_IMAGE_PATH` below to your submission and test image.

```
  private static final String TEST_IMAGE_PATH = "my_test_image.jpg";
  private static final String MODEL_PATH = "my_model.lite";
```

* Adjust the benchmark parameters when needed:

You can chnage the length of each experiment, and the processor affinity below. `BIG_CORE_MASK` is an integer whose binary encoding represents the set of used cores. This number is phone-specific. For example, Pixel 2 has 8 cores: the 4 little cores are represented by the 4 less significant bits, and the 4 big cores by the 4 more significant bits. Therefore a mask value of 16, or in binary `00010000`, represents using only the first big core. The mask 32, or in binary `00100000` uses the second big core and should deliver identical results as the mask 16 because the big cores are interchangeable.

```
  /** Wall time for each benchmarking experiment. */
  private static final double WALL_TIME = 3000;
  /** Maximum number of iterations in each benchmarking experiment. */
  private static final int MAX_ITERATIONS = 100;
  /** Mask for binding to a single big core. Pixel 1 (4), Pixel 2 (16). */
  private static final int BIG_CORE_MASK = 16;
```

Note: You'll need ROOT access to the phone to change processor affinity.

* Build and install the app.

```
bazel build -c opt --cxxopt=--std=c++11 --cxxopt=-Wno-all //tensorflow/contrib/lite/java/ovic/demo/app:ovic_benchmarker_binary
adb install -r bazel-bin/tensorflow/contrib/lite/java/ovic/demo/app/ovic_benchmarker_binary.apk
```

Start the app and click the `Start` button in dark green. The button should turn bright green, signaling that the experiment is running. The benchmarking results will be displayed after about the `WALL_TIME` you specified above. For example:

```
my_model.lite: Average latency=158.6ms after 20 runs.
```

### Sample latencies

Note: the benchmarking results can be quite different depending on the background processes running on the phone. A few things that help stabilize the app's readings are placing the phone on a cooling plate, restarting the phone, and shutting down internet access.

| Model                | Pixel 1 latency (ms)  | Pixel 2 latency (ms) |
| -------------------- |:---------------------:| --------------------:|
|  float_model.lite    | 120                   | 155                  |
| quantized_model.lite | 85                    | 74                   |
|  low_res_model.lite  | 4.2                   | 4.0                  |

Since Pixel 2 has excellent support for 8-bit quantized models, we strongly recommend you to check out the [quantization training tutorial](https://www.tensorflow.org/performance/quantization).

