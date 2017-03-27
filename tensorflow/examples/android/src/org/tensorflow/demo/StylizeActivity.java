/*
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.Toast;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Vector;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.R;

/**
 * Sample activity that stylizes the camera preview according to "A Learned Representation For
 * Artistic Style" (https://arxiv.org/abs/1610.07629)
 */
public class StylizeActivity extends CameraActivity implements OnImageAvailableListener {
  static {
    System.loadLibrary("tensorflow_demo");
  }

  private static final Logger LOGGER = new Logger();

  private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";
  private static final String INPUT_NODE = "input";
  private static final String STYLE_NODE = "style_num";
  private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";
  private static final int NUM_STYLES = 26;

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  // Whether to actively manipulate non-selected sliders so that sum of activations always appears
  // to be 1.0. The actual style input tensor will be normalized to sum to 1.0 regardless.
  private static final boolean NORMALIZE_SLIDERS = true;

  private static final float TEXT_SIZE_DIP = 12;

  private static final boolean DEBUG_MODEL = false;

  private static final int[] SIZES = {32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};

  // Start at a medium size, but let the user step up through smaller sizes so they don't get
  // immediately stuck processing a large image.
  private int desiredSizeIndex = -1;
  private int desiredSize = 256;
  private int initializedSize = 0;

  private Integer sensorOrientation;

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private final float[] styleVals = new float[NUM_STYLES];
  private int[] intValues;
  private float[] floatValues;

  private int frameNum = 0;

  private Bitmap cropCopyBitmap;
  private Bitmap textureCopyBitmap;

  private boolean computing = false;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;

  private TensorFlowInferenceInterface inferenceInterface;

  private int lastOtherStyle = 1;

  private boolean allZero = false;

  private ImageGridAdapter adapter;
  private GridView grid;

  private final OnTouchListener gridTouchAdapter =
      new OnTouchListener() {
        ImageSlider slider = null;

        @Override
        public boolean onTouch(final View v, final MotionEvent event) {
          switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
              for (int i = 0; i < NUM_STYLES; ++i) {
                final ImageSlider child = adapter.items[i];
                final Rect rect = new Rect();
                child.getHitRect(rect);
                if (rect.contains((int) event.getX(), (int) event.getY())) {
                  slider = child;
                  slider.setHilighted(true);
                }
              }
              break;

            case MotionEvent.ACTION_MOVE:
              if (slider != null) {
                final Rect rect = new Rect();
                slider.getHitRect(rect);

                final float newSliderVal =
                    (float)
                        Math.min(
                            1.0,
                            Math.max(
                                0.0, 1.0 - (event.getY() - slider.getTop()) / slider.getHeight()));

                setStyle(slider, newSliderVal);
              }
              break;

            case MotionEvent.ACTION_UP:
              if (slider != null) {
                slider.setHilighted(false);
                slider = null;
              }
              break;

            default: // fall out

          }
          return true;
        }
      };

  @Override
  public void onCreate(final Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_stylize;
  }

  @Override
  protected int getDesiredPreviewFrameSize() {
    return SIZES[SIZES.length - 1];
  }

  public static Bitmap getBitmapFromAsset(final Context context, final String filePath) {
    final AssetManager assetManager = context.getAssets();

    Bitmap bitmap = null;
    try {
      final InputStream inputStream = assetManager.open(filePath);
      bitmap = BitmapFactory.decodeStream(inputStream);
    } catch (final IOException e) {
      LOGGER.e("Error opening bitmap!", e);
    }

    return bitmap;
  }

  private class ImageSlider extends ImageView {
    private float value = 0.0f;
    private boolean hilighted = false;

    private final Paint boxPaint;
    private final Paint linePaint;

    public ImageSlider(final Context context) {
      super(context);
      value = 0.0f;

      boxPaint = new Paint();
      boxPaint.setColor(Color.BLACK);
      boxPaint.setAlpha(128);

      linePaint = new Paint();
      linePaint.setColor(Color.WHITE);
      linePaint.setStrokeWidth(10.0f);
      linePaint.setStyle(Style.STROKE);
    }

    @Override
    public void onDraw(final Canvas canvas) {
      super.onDraw(canvas);
      final float y = (1.0f - value) * canvas.getHeight();

      // If all sliders are zero, don't bother shading anything.
      if (!allZero) {
        canvas.drawRect(0, 0, canvas.getWidth(), y, boxPaint);
      }

      if (value > 0.0f) {
        canvas.drawLine(0, y, canvas.getWidth(), y, linePaint);
      }

      if (hilighted) {
        canvas.drawRect(0, 0, getWidth(), getHeight(), linePaint);
      }
    }

    @Override
    protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
      super.onMeasure(widthMeasureSpec, heightMeasureSpec);
      setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
    }

    public void setValue(final float value) {
      this.value = value;
      postInvalidate();
    }

    public void setHilighted(final boolean highlighted) {
      this.hilighted = highlighted;
      this.postInvalidate();
    }
  }

  private class ImageGridAdapter extends BaseAdapter {
    final ImageSlider[] items = new ImageSlider[NUM_STYLES];
    final ArrayList<Button> buttons = new ArrayList<>();

    {
      final Button sizeButton =
          new Button(StylizeActivity.this) {
            @Override
            protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
              super.onMeasure(widthMeasureSpec, heightMeasureSpec);
              setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
            }
          };
      sizeButton.setText("" + desiredSize);
      sizeButton.setOnClickListener(
          new OnClickListener() {
            @Override
            public void onClick(final View v) {
              desiredSizeIndex = (desiredSizeIndex + 1) % SIZES.length;
              desiredSize = SIZES[desiredSizeIndex];
              sizeButton.setText("" + desiredSize);
              sizeButton.postInvalidate();
            }
          });

      final Button saveButton =
          new Button(StylizeActivity.this) {
            @Override
            protected void onMeasure(final int widthMeasureSpec, final int heightMeasureSpec) {
              super.onMeasure(widthMeasureSpec, heightMeasureSpec);
              setMeasuredDimension(getMeasuredWidth(), getMeasuredWidth());
            }
          };
      saveButton.setText("Save");
      saveButton.setOnClickListener(
          new OnClickListener() {
            @Override
            public void onClick(final View v) {
              if (textureCopyBitmap != null) {
                // TODO(andrewharp): Save as jpeg with guaranteed unique filename.
                ImageUtils.saveBitmap(textureCopyBitmap, "stylized" + frameNum + ".png");
                Toast.makeText(
                        StylizeActivity.this,
                        "Saved image to: /sdcard/tensorflow/" + "stylized" + frameNum + ".png",
                        Toast.LENGTH_LONG)
                    .show();
              }
            }
          });

      buttons.add(sizeButton);
      buttons.add(saveButton);

      for (int i = 0; i < NUM_STYLES; ++i) {
        LOGGER.v("Creating item %d", i);

        if (items[i] == null) {
          final ImageSlider slider = new ImageSlider(StylizeActivity.this);
          final Bitmap bm =
              getBitmapFromAsset(StylizeActivity.this, "thumbnails/style" + i + ".jpg");
          slider.setImageBitmap(bm);

          items[i] = slider;
        }
      }
    }

    @Override
    public int getCount() {
      return buttons.size() + NUM_STYLES;
    }

    @Override
    public Object getItem(final int position) {
      if (position < buttons.size()) {
        return buttons.get(position);
      } else {
        return items[position - buttons.size()];
      }
    }

    @Override
    public long getItemId(final int position) {
      return getItem(position).hashCode();
    }

    @Override
    public View getView(final int position, final View convertView, final ViewGroup parent) {
      if (convertView != null) {
        return convertView;
      }
      return (View) getItem(position);
    }
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d", rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            renderDebug(canvas);
          }
        });

    adapter = new ImageGridAdapter();
    grid = (GridView) findViewById(R.id.grid_layout);
    grid.setAdapter(adapter);
    grid.setOnTouchListener(gridTouchAdapter);

    setStyle(adapter.items[0], 1.0f);
  }

  private void setStyle(final ImageSlider slider, final float value) {
    slider.setValue(value);

    if (NORMALIZE_SLIDERS) {
      // Slider vals correspond directly to the input tensor vals, and normalization is visually
      // maintained by remanipulating non-selected sliders.
      float otherSum = 0.0f;

      for (int i = 0; i < NUM_STYLES; ++i) {
        if (adapter.items[i] != slider) {
          otherSum += adapter.items[i].value;
        }
      }

      if (otherSum > 0.0) {
        float highestOtherVal = 0;
        final float factor = otherSum > 0.0f ? (1.0f - value) / otherSum : 0.0f;
        for (int i = 0; i < NUM_STYLES; ++i) {
          final ImageSlider child = adapter.items[i];
          if (child == slider) {
            continue;
          }
          final float newVal = child.value * factor;
          child.setValue(newVal > 0.01f ? newVal : 0.0f);

          if (child.value > highestOtherVal) {
            lastOtherStyle = i;
            highestOtherVal = child.value;
          }
        }
      } else {
        // Everything else is 0, so just pick a suitable slider to push up when the
        // selected one goes down.
        if (adapter.items[lastOtherStyle] == slider) {
          lastOtherStyle = (lastOtherStyle + 1) % NUM_STYLES;
        }
        adapter.items[lastOtherStyle].setValue(1.0f - value);
      }
    }

    final boolean lastAllZero = allZero;
    float sum = 0.0f;
    for (int i = 0; i < NUM_STYLES; ++i) {
      sum += adapter.items[i].value;
    }
    allZero = sum == 0.0f;

    // Now update the values used for the input tensor. If nothing is set, mix in everything
    // equally. Otherwise everything is normalized to sum to 1.0.
    for (int i = 0; i < NUM_STYLES; ++i) {
      styleVals[i] = allZero ? 1.0f / NUM_STYLES : adapter.items[i].value / sum;

      if (lastAllZero != allZero) {
        adapter.items[i].postInvalidate();
      }
    }
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;

    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (computing) {
        image.close();
        return;
      }

      if (desiredSize != initializedSize) {
        LOGGER.i(
            "Initializing at size preview size %dx%d, stylize size %d",
            previewWidth, previewHeight, desiredSize);
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(desiredSize, desiredSize, Config.ARGB_8888);

        frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                desiredSize, desiredSize,
                sensorOrientation, true);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        yuvBytes = new byte[3][];

        intValues = new int[desiredSize * desiredSize];
        floatValues = new float[desiredSize * desiredSize * 3];
        initializedSize = desiredSize;
      }

      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

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

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

            final long startTime = SystemClock.uptimeMillis();
            stylizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            textureCopyBitmap = Bitmap.createBitmap(croppedBitmap);

            requestRender();
            computing = false;
          }
        });

    Trace.endSection();
  }

  private void stylizeImage(final Bitmap bitmap) {
    ++frameNum;
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    if (DEBUG_MODEL) {
      // Create a white square that steps through a black background 1 pixel per frame.
      final int centerX = (frameNum + bitmap.getWidth() / 2) % bitmap.getWidth();
      final int centerY = bitmap.getHeight() / 2;
      final int squareSize = 10;
      for (int i = 0; i < intValues.length; ++i) {
        final int x = i % bitmap.getWidth();
        final int y = i / bitmap.getHeight();
        final float val =
            Math.abs(x - centerX) < squareSize && Math.abs(y - centerY) < squareSize ? 1.0f : 0.0f;
        floatValues[i * 3] = val;
        floatValues[i * 3 + 1] = val;
        floatValues[i * 3 + 2] = val;
      }
    } else {
      for (int i = 0; i < intValues.length; ++i) {
        final int val = intValues[i];
        floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
        floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
        floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
      }
    }

    // Copy the input data into TensorFlow.
    inferenceInterface.feed(
        INPUT_NODE, floatValues, 1, bitmap.getWidth(), bitmap.getHeight(), 3);
    inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);

    inferenceInterface.run(new String[] {OUTPUT_NODE}, isDebug());
    inferenceInterface.fetch(OUTPUT_NODE, floatValues);

    for (int i = 0; i < intValues.length; ++i) {
      intValues[i] =
          0xFF000000
              | (((int) (floatValues[i * 3] * 255)) << 16)
              | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
              | ((int) (floatValues[i * 3 + 2] * 255));
    }

    bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
  }

  private void renderDebug(final Canvas canvas) {
    // TODO(andrewharp): move result display to its own View instead of using debug overlay.
    final Bitmap texture = textureCopyBitmap;
    if (texture != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor =
          DEBUG_MODEL
              ? 4.0f
              : Math.min(
                  (float) canvas.getWidth() / texture.getWidth(),
                  (float) canvas.getHeight() / texture.getHeight());
      matrix.postScale(scaleFactor, scaleFactor);
      canvas.drawBitmap(texture, matrix, new Paint());
    }

    if (!isDebug()) {
      return;
    }

    final Bitmap copy = cropCopyBitmap;
    if (copy == null) {
      return;
    }

    canvas.drawColor(0x55000000);

    final Matrix matrix = new Matrix();
    final float scaleFactor = 2;
    matrix.postScale(scaleFactor, scaleFactor);
    matrix.postTranslate(
        canvas.getWidth() - copy.getWidth() * scaleFactor,
        canvas.getHeight() - copy.getHeight() * scaleFactor);
    canvas.drawBitmap(copy, matrix, new Paint());

    final Vector<String> lines = new Vector<>();

    final String[] statLines = inferenceInterface.getStatString().split("\n");
    Collections.addAll(lines, statLines);

    lines.add("");

    lines.add("Frame: " + previewWidth + "x" + previewHeight);
    lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
    lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
    lines.add("Rotation: " + sensorOrientation);
    lines.add("Inference time: " + lastProcessingTimeMs + "ms");
    lines.add("Desired size: " + desiredSize);
    lines.add("Initialized size: " + initializedSize);

    borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
  }
}
