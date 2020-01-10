package org.tensorflow.demo.env;

import android.graphics.Bitmap;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;

/**
 * Utility class for manipulating images.
 **/
public class ImageUtils {
  @SuppressWarnings("unused")
  private static final Logger LOGGER = new Logger();

  /**
   * Utility method to compute the allocated size in bytes of a YUV420SP image
   * of the given dimensions.
   */
  public static int getYUVByteSize(final int width, final int height) {
    // The luminance plane requires 1 byte per pixel.
    final int ySize = width * height;

    // The UV plane works on 2x2 blocks, so dimensions with odd size must be rounded up.
    // Each 2x2 block takes 2 bytes to encode, one each for U and V.
    final int uvSize = ((width + 1) / 2) * ((height + 1) / 2) * 2;

    return ySize + uvSize;
  }

  /**
   * Saves a Bitmap object to disk for analysis.
   *
   * @param bitmap The bitmap to save.
   */
  public static void saveBitmap(final Bitmap bitmap) {
    final String root =
        Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "tensorflow";
    LOGGER.i("Saving %dx%d bitmap to %s.", bitmap.getWidth(), bitmap.getHeight(), root);
    final File myDir = new File(root);

    if (!myDir.mkdirs()) {
      LOGGER.i("Make dir failed");
    }

    final String fname = "preview.png";
    final File file = new File(myDir, fname);
    if (file.exists()) {
      file.delete();
    }
    try {
      final FileOutputStream out = new FileOutputStream(file);
      bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
      out.flush();
      out.close();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
    }
  }

  /**
   * Converts YUV420 semi-planar data to ARGB 8888 data using the supplied width
   * and height. The input and output must already be allocated and non-null.
   * For efficiency, no error checking is performed.
   *
   * @param input The array of YUV 4:2:0 input data.
   * @param output A pre-allocated array for the ARGB 8:8:8:8 output data.
   * @param width The width of the input image.
   * @param height The height of the input image.
   * @param halfSize If true, downsample to 50% in each dimension, otherwise not.
   */
  public static native void convertYUV420SPToARGB8888(
      byte[] input, int[] output, int width, int height, boolean halfSize);

  /**
   * Converts YUV420 semi-planar data to RGB 565 data using the supplied width
   * and height. The input and output must already be allocated and non-null.
   * For efficiency, no error checking is performed.
   *
   * @param input The array of YUV 4:2:0 input data.
   * @param output A pre-allocated array for the RGB 5:6:5 output data.
   * @param width The width of the input image.
   * @param height The height of the input image.
   */
  public static native void convertYUV420SPToRGB565(
      byte[] input, byte[] output, int width, int height);

  /**
   * Converts 32-bit ARGB8888 image data to YUV420SP data.  This is useful, for
   * instance, in creating data to feed the classes that rely on raw camera
   * preview frames.
   *
   * @param input An array of input pixels in ARGB8888 format.
   * @param output A pre-allocated array for the YUV420SP output data.
   * @param width The width of the input image.
   * @param height The height of the input image.
   */
  public static native void convertARGB8888ToYUV420SP(
      int[] input, byte[] output, int width, int height);

  /**
   * Converts 16-bit RGB565 image data to YUV420SP data.  This is useful, for
   * instance, in creating data to feed the classes that rely on raw camera
   * preview frames.
   *
   * @param input An array of input pixels in RGB565 format.
   * @param output A pre-allocated array for the YUV420SP output data.
   * @param width The width of the input image.
   * @param height The height of the input image.
   */
  public static native void convertRGB565ToYUV420SP(
      byte[] input, byte[] output, int width, int height);
}
