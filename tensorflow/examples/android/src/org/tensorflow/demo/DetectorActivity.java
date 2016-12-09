/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.demo.R;

/**
 * An activity that uses a TensorFlowMultiboxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final int NUM_LOCATIONS = 784;
  private static final int INPUT_SIZE = 224;
  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128;
  private static final String INPUT_NAME = "ResizeBilinear";
  private static final String OUTPUT_NAMES = "output_locations/Reshape,output_scores/Reshape";

  private static final String MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String LOCATION_FILE = "file:///android_asset/multibox_location_priors.pb";

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE = 0.1f;

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  private static final boolean MAINTAIN_ASPECT = false;

  private static final float TEXT_SIZE_DIP = 18;

  private Integer sensorOrientation;

  private TensorFlowMultiBoxDetector detector;

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private boolean computing = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private Bitmap cropCopyBitmap;

  private MultiBoxTracker tracker;

  private byte[] luminance;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);

    tracker = new MultiBoxTracker(getResources().getDisplayMetrics());

    detector = new TensorFlowMultiBoxDetector();
    try {
      detector.initializeTensorFlow(
          getAssets(),
          MODEL_FILE,
          LOCATION_FILE,
          NUM_LOCATIONS,
          INPUT_SIZE,
          IMAGE_MEAN,
          IMAGE_STD,
          INPUT_NAME,
          OUTPUT_NAMES);
    } catch (final IOException e) {
      LOGGER.e(e, "Exception!");
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbBytes = new int[previewWidth * previewHeight];
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            INPUT_SIZE, INPUT_SIZE,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
    yuvBytes = new byte[3][];

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            final Bitmap copy = cropCopyBitmap;

            tracker.draw(canvas);

            if (!isDebug()) {
              return;
            }

            tracker.drawDebug(canvas);

            if (copy != null) {
              final Matrix matrix = new Matrix();
              final float scaleFactor = 2;
              matrix.postScale(scaleFactor, scaleFactor);
              matrix.postTranslate(
                  canvas.getWidth() - copy.getWidth() * scaleFactor,
                  canvas.getHeight() - copy.getHeight() * scaleFactor);
              canvas.drawBitmap(copy, matrix, new Paint());

              final Vector<String> lines = new Vector<String>();
              lines.add("Frame: " + previewWidth + "x" + previewHeight);
              lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
              lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
              lines.add("Rotation: " + sensorOrientation);
              lines.add("Inference time: " + lastProcessingTimeMs + "ms");

              int lineNum = 0;
              for (final String line : lines) {
                borderedText.drawText(
                    canvas,
                    10,
                    canvas.getHeight() - 10 - borderedText.getTextSize() * lineNum,
                    line);
                ++lineNum;
              }
            }
          }
        });
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;

    ++timestamp;
    final long currTimestamp = timestamp;

    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      tracker.onFrame(
          previewWidth,
          previewHeight,
          planes[0].getRowStride(),
          sensorOrientation,
          yuvBytes[0],
          timestamp);

      requestRender();

      // No mutex needed as this method is not reentrant.
      if (computing) {
        image.close();
        return;
      }
      computing = true;

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
          yuvBytes[0],
          yuvBytes[1],
          yuvBytes[2],
          rgbBytes,
          previewWidth,
          previewHeight,
          yRowStride,
          uvRowStride,
          uvPixelStride,
          false);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    if (luminance == null) {
      luminance = new byte[yuvBytes[0].length];
    }
    System.arraycopy(yuvBytes[0], 0, luminance, 0, luminance.length);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, luminance, currTimestamp);

            requestRender();
            computing = false;
          }
        });

    Trace.endSection();
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected int getDesiredPreviewFrameSize() {
    return INPUT_SIZE;
  }
}
