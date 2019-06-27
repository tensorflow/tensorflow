# TensorFlow Lite Android image classification example

This document walks through the code of a simple Android mobile application that
demonstrates [image classification](overview.md) using the device camera.

The application code is located in the
[Tensorflow examples](https://github.com/tensorflow/examples) repository, along
with instructions for building and deploying the app.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Example
application</a>

## Explore the code

We're now going to walk through the most important parts of the sample code.

### Get camera input

This mobile application gets the camera input using the functions defined in the
file
[`CameraActivity.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/CameraActivity.java).
This file depends on
[`AndroidManifest.xml`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/AndroidManifest.xml)
to set the camera orientation.

`CameraActivity` also contains code to capture user preferences from the UI and
make them available to other classes via convenience methods.

```java
model = Model.valueOf(modelSpinner.getSelectedItem().toString().toUpperCase());
device = Device.valueOf(deviceSpinner.getSelectedItem().toString());
numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
```

### Classifier

The file
[`Classifier.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/Classifier.java)
contains most of the complex logic for processing the camera input and running
inference.

Two subclasses of the file exist, in
[`ClassifierFloatMobileNet.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/ClassifierFloatMobileNet.java)
and
[`ClassifierQuantizedMobileNet.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/ClassifierQuantizedMobileNet.java),
to demonstrate the use of both floating point and
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
models.

The `Classifier` class implements a static method, `create`, which is used to
instantiate the appropriate subclass based on the supplied model type (quantized
vs floating point).

#### Load model and create interpreter

To perform inference, we need to load a model file and instantiate an
`Interpreter`. This happens in the constructor of the `Classifier` class, along
with loading the list of class labels. Information about the device type and
number of threads is used to configure the `Interpreter` via the
`Interpreter.Options` instance passed into its constructor. Note how that in the
case of a GPU being available, a
[`Delegate`](https://www.tensorflow.org/lite/performance/gpu) is created using
`GpuDelegateHelper`.

```java
protected Classifier(Activity activity, Device device, int numThreads) throws IOException {
  tfliteModel = loadModelFile(activity);
  switch (device) {
    case NNAPI:
      tfliteOptions.setUseNNAPI(true);
      break;
    case GPU:
      gpuDelegate = GpuDelegateHelper.createGpuDelegate();
      tfliteOptions.addDelegate(gpuDelegate);
      break;
    case CPU:
      break;
  }
  tfliteOptions.setNumThreads(numThreads);
  tflite = new Interpreter(tfliteModel, tfliteOptions);
  labels = loadLabelList(activity);
...
```

For Android devices, we recommend pre-loading and memory mapping the model file
to offer faster load times and reduce the dirty pages in memory. The method
`loadModelFile` does this, returning a `MappedByteBuffer` containing the model.

```java
private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
  AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
  FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
  FileChannel fileChannel = inputStream.getChannel();
  long startOffset = fileDescriptor.getStartOffset();
  long declaredLength = fileDescriptor.getDeclaredLength();
  return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
}
```

Note: If your model file is compressed then you will have to load the model as a
`File`, as it cannot be directly mapped and used from memory.

The `MappedByteBuffer` is passed into the `Interpreter` constructor, along with
an `Interpreter.Options` object. This object can be used to configure the
interpreter, for example by setting the number of threads (`.setNumThreads(1)`)
or enabling [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)
(`.setUseNNAPI(true)`).

#### Pre-process bitmap image

Next in the `Classifier` constructor, we take the input camera bitmap image and
convert it to a `ByteBuffer` format for efficient processing. We pre-allocate
the memory for the `ByteBuffer` object based on the image dimensions because
Bytebuffer objects can't infer the object shape.

The `ByteBuffer` represents the image as a 1D array with three bytes per channel
(red, green, and blue). We call `order(ByteOrder.nativeOrder())` to ensure bits
are stored in the device's native order.

```java
imgData =
  ByteBuffer.allocateDirect(
    DIM_BATCH_SIZE
      * getImageSizeX()
      * getImageSizeY()
      * DIM_PIXEL_SIZE
      * getNumBytesPerChannel());
imgData.order(ByteOrder.nativeOrder());
```

The code in `convertBitmapToByteBuffer` pre-processes the incoming bitmap images
from the camera to this `ByteBuffer`. It calls the method `addPixelValue` to add
each set of pixel values to the `ByteBuffer` sequentially.

```java
imgData.rewind();
bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
// Convert the image to floating point.
int pixel = 0;
for (int i = 0; i < getImageSizeX(); ++i) {
  for (int j = 0; j < getImageSizeY(); ++j) {
    final int val = intValues[pixel++];
    addPixelValue(val);
  }
}
```

In `ClassifierQuantizedMobileNet`, `addPixelValue` is overridden to put a single
byte for each channel. The bitmap contains an encoded color for each pixel in
ARGB format, so we need to mask the least significant 8 bits to get blue, and
next 8 bits to get green and next 8 bits to get blue. Since we have an opaque
image, alpha can be ignored.

```java
@Override
protected void addPixelValue(int pixelValue) {
  imgData.put((byte) ((pixelValue >> 16) & 0xFF));
  imgData.put((byte) ((pixelValue >> 8) & 0xFF));
  imgData.put((byte) (pixelValue & 0xFF));
}
```

For `ClassifierFloatMobileNet`, we must provide a floating point number for each
channel where the value is between `0` and `1`. To do this, we mask out each
color channel as before, but then divide each resulting value by `255.f`.

```java
@Override
protected void addPixelValue(int pixelValue) {
  imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.f);
  imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.f);
  imgData.putFloat((pixelValue & 0xFF) / 255.f);
}
```

#### Run inference

The method that runs inference, `runInference`, is implemented by each subclass
of `Classifier`. In `ClassifierQuantizedMobileNet`, the method looks as follows:

```java
protected void runInference() {
  tflite.run(imgData, labelProbArray);
}
```

The output of the inference is stored in a byte array `labelProbArray`, which is
allocated in the subclass's constructor. It consists of a single outer element,
containing one innner element for each label in the classification model.

To run inference, we call `run()` on the interpreter instance, passing the input
and output buffers as arguments.

#### Recognize image

Rather than call `runInference` directly, the method `recognizeImage` is used.
It accepts a bitmap, runs inference, and returns a sorted `List` of
`Recognition` instances, each corresponding to a label. The method will return a
number of results bounded by `MAX_RESULTS`, which is 3 by default.

`Recognition` is a simple class that contains information about a specific
recognition result, including its `title` and `confidence`.

A `PriorityQueue` is used for sorting. Each `Classifier` subclass has a
`getNormalizedProbability` method, which is expected to return a probability
between 0 and 1 of a given class being represented by the image.

```java
PriorityQueue<Recognition> pq =
  new PriorityQueue<Recognition>(
    3,
    new Comparator<Recognition>() {
      @Override
      public int compare(Recognition lhs, Recognition rhs) {
        // Intentionally reversed to put high confidence at the head of the queue.
        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
      }
    });
for (int i = 0; i < labels.size(); ++i) {
  pq.add(
    new Recognition(
      "" + i,
      labels.size() > i ? labels.get(i) : "unknown",
      getNormalizedProbability(i),
      null));
}
```

### Display results

The classifier is invoked and inference results are displayed by the
`processImage()` function in
[`ClassifierActivity.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/ClassifierActivity.java).

`ClassifierActivity` is a subclass of `CameraActivity` that contains method
implementations that render the camera image, run classification, and display
the results. The method `processImage()` runs classification on a background
thread as fast as possible, rendering information on the UI thread to avoid
blocking inference and creating latency.

```java
protected void processImage() {
  rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
  final Canvas canvas = new Canvas(croppedBitmap);
  canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

  runInBackground(
      new Runnable() {
        @Override
        public void run() {
          if (classifier != null) {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.v("Detect: %s", results);
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showResultsInBottomSheet(results);
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showCameraResolution(canvas.getWidth() + "x" + canvas.getHeight());
                    showRotationInfo(String.valueOf(sensorOrientation));
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
          readyForNextImage();
        }
      });
}
```

Another important role of `ClassifierActivity` is to determine user preferences
(by interrogating `CameraActivity`), and instantiate the appropriately
configured `Classifier` subclass. This happens when the video feed begins (via
`onPreviewSizeChosen()`) and when options are changed in the UI (via
`onInferenceConfigurationChanged()`).

```java
private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU) {
      if (!GpuDelegateHelper.isGpuDelegateAvailable()) {
        LOGGER.d("Not creating classifier: GPU support unavailable.");
        runOnUiThread(
            () -> {
              Toast.makeText(this, "GPU acceleration unavailable.", Toast.LENGTH_LONG).show();
            });
        return;
      } else if (model == Model.QUANTIZED && device == Device.GPU) {
        LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
        runOnUiThread(
            () -> {
              Toast.makeText(
                      this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                  .show();
            });
        return;
      }
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }
  }
```
