package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;

import junit.framework.Assert;

import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

import java.nio.ByteBuffer;
import java.util.List;

/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with Tensorflow.
 */
public class TensorflowImageListener implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  private static final int NUM_CLASSES = 1001;
  private static final int INPUT_SIZE = 224;
  private static final int IMAGE_MEAN = 117;

  // TODO(andrewharp): Get orientation programatically.
  private final int screenRotation = 90;

  private final TensorflowClassifier tensorflow = new TensorflowClassifier();

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[] yuvBytes = null;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private RecognitionScoreView scoreView;

  public void initialize(final AssetManager assetManager, final RecognitionScoreView scoreView) {
    tensorflow.initializeTensorflow(
        assetManager, MODEL_FILE, LABEL_FILE, NUM_CLASSES, INPUT_SIZE, IMAGE_MEAN);
    this.scoreView = scoreView;
  }

  private void drawResizedBitmap(final Bitmap src, final Bitmap dst) {
    Assert.assertEquals(dst.getWidth(), dst.getHeight());
    final float minDim = Math.min(src.getWidth(), src.getHeight());

    final Matrix matrix = new Matrix();

    // We only want the center square out of the original rectangle.
    final float translateX = -Math.max(0, (src.getWidth() - minDim) / 2);
    final float translateY = -Math.max(0, (src.getHeight() - minDim) / 2);
    matrix.preTranslate(translateX, translateY);

    final float scaleFactor = dst.getHeight() / minDim;
    matrix.postScale(scaleFactor, scaleFactor);

    // Rotate around the center if necessary.
    if (screenRotation != 0) {
      matrix.postTranslate(-dst.getWidth() / 2.0f, -dst.getHeight() / 2.0f);
      matrix.postRotate(screenRotation);
      matrix.postTranslate(dst.getWidth() / 2.0f, dst.getHeight() / 2.0f);
    }

    final Canvas canvas = new Canvas(dst);
    canvas.drawBitmap(src, matrix, null);
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;
    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      // Initialize the storage bitmaps once when the resolution is known.
      if (previewWidth != image.getWidth() || previewHeight != image.getHeight()) {
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        previewWidth = image.getWidth();
        previewHeight = image.getHeight();
        rgbBytes = new int[previewWidth * previewHeight];
        yuvBytes = new byte[ImageUtils.getYUVByteSize(previewWidth, previewHeight)];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);
      }

      final Plane[] planes = image.getPlanes();
      int position = 0;

      // Copy the bytes from the Image into a buffer for easier conversion to RGB.
      // TODO(andrewharp): It may not be correct to do it this way.
      final int[] planeOrder = {0, 2};
      for (int i = 0; i < planeOrder.length; ++i) {
        final Plane plane = planes[planeOrder[i]];
        final ByteBuffer buffer = plane.getBuffer();

        buffer.rewind();
        final int readAmount = buffer.remaining();

        buffer.get(yuvBytes, position, readAmount);
        position += readAmount;
      }

      image.close();

      ImageUtils.convertYUV420SPToARGB8888(yuvBytes, rgbBytes, previewWidth, previewHeight, false);
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    drawResizedBitmap(rgbFrameBitmap, croppedBitmap);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    final List<Classifier.Recognition> results = tensorflow.recognizeImage(croppedBitmap);

    LOGGER.v("%d results", results.size());
    for (final Classifier.Recognition result : results) {
      LOGGER.v("Result: " + result.getTitle());
    }
    scoreView.setResults(results);
  }
}
