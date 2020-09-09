/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.benchmark.firebase;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * {@code Activity} class for Firebase Game Loop test.
 *
 * <p>This Activity receives and handles an {@code Intent} for Firebase Game Loop test. Refer to
 * https://firebase.google.com/docs/test-lab/android/game-loop.
 */
public class BenchmarkModelActivity extends Activity {

  private static final String TAG = "tflite_BenchmarkModelActivity";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();
    if (!intent.getAction().equals("com.google.intent.action.TEST_LOOP")) {
      Log.e(TAG, "Received non Firebase Game Loop test intent " + intent.getAction());
      finish();
    }
    int scenario = intent.getIntExtra("scenario", 0);
    Log.i(TAG, "Running TensorFlow Lite benchmark with scenario: " + scenario);

    ParcelFileDescriptor parcelFileDescriptor = null;
    Uri reportFile = intent.getData();
    if (reportFile != null) {
      Log.i(TAG, "Logging the result to " + reportFile.getEncodedPath());
      try {
        parcelFileDescriptor =
            getContentResolver().openAssetFileDescriptor(reportFile, "w").getParcelFileDescriptor();
      } catch (FileNotFoundException | NullPointerException e) {
        Log.e(TAG, "Error while opening Firebase Test Lab report file", e);
      }
    }

    int reportFd = parcelFileDescriptor != null ? parcelFileDescriptor.getFd() : -1;
    BenchmarkModel.run(this, scenario, reportFd);

    if (parcelFileDescriptor != null) {
      try {
        parcelFileDescriptor.close();
      } catch (IOException e) {
        Log.e(TAG, "Failed to close Firebase Test Lab result file", e);
      }
    }

    finish();
  }
}
