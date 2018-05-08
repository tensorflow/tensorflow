/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package ovic.demo.app;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import org.tensorflow.ovic.OvicSingleImageResult;

/** Class that benchmark image classifier models. */
public class OvicBenchmarkerActivity extends Activity {
  /** Tag for the {@link Log}. */
  private static final String TAG = "OvicBenchmarkerActivity";

  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "labels.txt";

  private static final String TEST_IMAGE_PATH = "test_image_224.jpg";
  private static final String MODEL_PATH = "float_model.lite";
  /**
   * Each bottom press will launch a benchmarking experiment. The experiment stops when either the
   * total native latency reaches WALL_TIME or the number of iterations reaches MAX_ITERATIONS,
   * whichever comes first.
   */
  /** Wall time for each benchmarking experiment. */
  private static final double WALL_TIME = 3000;
  /** Maximum number of iterations in each benchmarking experiment. */
  private static final int MAX_ITERATIONS = 100;

  /* The model to be benchmarked. */
  private MappedByteBuffer model = null;
  private InputStream labelInputStream = null;
  private OvicBenchmarker benchmarker;
  /** Inference result of each iteration. */
  OvicSingleImageResult iterResult = null;

  private TextView textView = null;
  // private Button startButton = null;
  private static final DecimalFormat df2 = new DecimalFormat(".##");

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // TextView used to display the progress, for information purposes only.
    textView = (TextView) findViewById(R.id.textView);
  }

  private Bitmap loadTestBitmap() throws IOException {
    InputStream imageStream = getAssets().open(TEST_IMAGE_PATH);
    return BitmapFactory.decodeStream(imageStream);
  }

  public void initializeTest() throws IOException {
    Log.i(TAG, "Initializing benchmarker.");
    benchmarker = new OvicBenchmarker(WALL_TIME);
    AssetManager am = getAssets();
    AssetFileDescriptor fileDescriptor = am.openFd(MODEL_PATH);
    FileInputStream modelInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = modelInputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    model = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    labelInputStream = am.open(LABEL_PATH);
  }

  public Boolean doTestIteration() throws IOException, InterruptedException {
    if (benchmarker == null) {
      throw new RuntimeException("Benchmarker has not been initialized.");
    }
    if (benchmarker.shouldStop()) {
      return false;
    }
    if (!benchmarker.readyToTest()) {
      Log.i(TAG, "getting ready to test.");
      benchmarker.getReadyToTest(labelInputStream, model);
      if (!benchmarker.readyToTest()) {
        throw new RuntimeException("Failed to get the benchmarker ready.");
      }
    }
    Log.i(TAG, "Going to do test iter.");
    // Start testing.
    Bitmap testImageBitmap = loadTestBitmap();
    iterResult = benchmarker.doTestIteration(testImageBitmap);
    testImageBitmap.recycle();
    if (iterResult == null) {
      throw new RuntimeException("Inference failed to produce a result.");
    }
    Log.i(TAG, iterResult.toString());
    return true;
  }

  public void startPressed(View view) throws IOException {
    Log.i(TAG, "Start pressed");
    try {
      initializeTest();
    } catch (IOException e) {
      Log.e(TAG, "Can't initialize benchmarker.", e);
      throw e;
    }
    Log.i(TAG, "Successfully initialized benchmarker.");
    int testIter = 0;
    Boolean iterSuccess = false;
    double totalLatency = 0.0f;
    while (testIter < MAX_ITERATIONS) {
      try {
        iterSuccess = doTestIteration();
      } catch (IOException e) {
        Log.e(TAG, "Error during iteration " + testIter);
        throw e;
      } catch (InterruptedException e) {
        Log.e(TAG, "Interrupted at iteration " + testIter);
      }
      if (!iterSuccess) {
        break;
      }
      testIter++;
      totalLatency += (double) iterResult.latency;
    }
    ;
    Log.i(TAG, "Benchmarking finished");

    if (textView != null) {
      if (testIter > 0) {
        textView
            .setText(
                MODEL_PATH
                    + ": Average latency="
                    + df2.format(totalLatency / testIter)
                    + "ms after "
                    + testIter
                    + " runs.");
      } else {
        textView.setText("Benchmarker failed to run on more than one images.");
      }
    }
  }
}
