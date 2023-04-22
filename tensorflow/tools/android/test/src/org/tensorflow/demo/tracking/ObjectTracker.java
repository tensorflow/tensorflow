/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.demo.tracking;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.Typeface;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import javax.microedition.khronos.opengles.GL10;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.env.Size;

/**
 * True object detector/tracker class that tracks objects across consecutive preview frames.
 * It provides a simplified Java interface to the analogous native object defined by
 * jni/client_vision/tracking/object_tracker.*.
 *
 * Currently, the ObjectTracker is a singleton due to native code restrictions, and so must
 * be allocated by ObjectTracker.getInstance(). In addition, release() should be called
 * as soon as the ObjectTracker is no longer needed, and before a new one is created.
 *
 * nextFrame() should be called as new frames become available, preferably as often as possible.
 *
 * After allocation, new TrackedObjects may be instantiated via trackObject(). TrackedObjects
 * are associated with the ObjectTracker that created them, and are only valid while that
 * ObjectTracker still exists.
 */
public class ObjectTracker {
  private static final Logger LOGGER = new Logger();

  private static boolean libraryFound = false;

  static {
    try {
      System.loadLibrary("tensorflow_demo");
      libraryFound = true;
    } catch (UnsatisfiedLinkError e) {
      LOGGER.e("libtensorflow_demo.so not found, tracking unavailable");
    }
  }

  private static final boolean DRAW_TEXT = false;

  /**
   * How many history points to keep track of and draw in the red history line.
   */
  private static final int MAX_DEBUG_HISTORY_SIZE = 30;

  /**
   * How many frames of optical flow deltas to record.
   * TODO(andrewharp): Push this down to the native level so it can be polled
   * efficiently into a an array for upload, instead of keeping a duplicate
   * copy in Java.
   */
  private static final int MAX_FRAME_HISTORY_SIZE = 200;

  private static final int DOWNSAMPLE_FACTOR = 2;

  private final byte[] downsampledFrame;

  protected static ObjectTracker instance;

  private final Map<String, TrackedObject> trackedObjects;

  private long lastTimestamp;

  private FrameChange lastKeypoints;

  private final Vector<PointF> debugHistory;

  private final LinkedList<TimestampedDeltas> timestampedDeltas;

  protected final int frameWidth;
  protected final int frameHeight;
  private final int rowStride;
  protected final boolean alwaysTrack;

  private static class TimestampedDeltas {
    final long timestamp;
    final byte[] deltas;

    public TimestampedDeltas(final long timestamp, final byte[] deltas) {
      this.timestamp = timestamp;
      this.deltas = deltas;
    }
  }

  /**
   * A simple class that records keypoint information, which includes
   * local location, score and type. This will be used in calculating
   * FrameChange.
   */
  public static class Keypoint {
    public final float x;
    public final float y;
    public final float score;
    public final int type;

    public Keypoint(final float x, final float y) {
      this.x = x;
      this.y = y;
      this.score = 0;
      this.type = -1;
    }

    public Keypoint(final float x, final float y, final float score, final int type) {
      this.x = x;
      this.y = y;
      this.score = score;
      this.type = type;
    }

    Keypoint delta(final Keypoint other) {
      return new Keypoint(this.x - other.x, this.y - other.y);
    }
  }

  /**
   * A simple class that could calculate Keypoint delta.
   * This class will be used in calculating frame translation delta
   * for optical flow.
   */
  public static class PointChange {
    public final Keypoint keypointA;
    public final Keypoint keypointB;
    Keypoint pointDelta;
    private final boolean wasFound;

    public PointChange(final float x1, final float y1,
                       final float x2, final float y2,
                       final float score, final int type,
                       final boolean wasFound) {
      this.wasFound = wasFound;

      keypointA = new Keypoint(x1, y1, score, type);
      keypointB = new Keypoint(x2, y2);
    }

    public Keypoint getDelta() {
      if (pointDelta == null) {
        pointDelta = keypointB.delta(keypointA);
      }
      return pointDelta;
    }
  }

  /** A class that records a timestamped frame translation delta for optical flow. */
  public static class FrameChange {
    public static final int KEYPOINT_STEP = 7;

    public final Vector<PointChange> pointDeltas;

    private final float minScore;
    private final float maxScore;

    public FrameChange(final float[] framePoints) {
      float minScore = 100.0f;
      float maxScore = -100.0f;

      pointDeltas = new Vector<PointChange>(framePoints.length / KEYPOINT_STEP);

      for (int i = 0; i < framePoints.length; i += KEYPOINT_STEP) {
        final float x1 = framePoints[i + 0] * DOWNSAMPLE_FACTOR;
        final float y1 = framePoints[i + 1] * DOWNSAMPLE_FACTOR;

        final boolean wasFound = framePoints[i + 2] > 0.0f;

        final float x2 = framePoints[i + 3] * DOWNSAMPLE_FACTOR;
        final float y2 = framePoints[i + 4] * DOWNSAMPLE_FACTOR;
        final float score = framePoints[i + 5];
        final int type = (int) framePoints[i + 6];

        minScore = Math.min(minScore, score);
        maxScore = Math.max(maxScore, score);

        pointDeltas.add(new PointChange(x1, y1, x2, y2, score, type, wasFound));
      }

      this.minScore = minScore;
      this.maxScore = maxScore;
    }
  }

  public static synchronized ObjectTracker getInstance(
      final int frameWidth, final int frameHeight, final int rowStride, final boolean alwaysTrack) {
    if (!libraryFound) {
      LOGGER.e(
          "Native object tracking support not found. "
              + "See tensorflow/tools/android/test/README.md for details.");
      return null;
    }

    if (instance == null) {
      instance = new ObjectTracker(frameWidth, frameHeight, rowStride, alwaysTrack);
      instance.init();
    } else {
      throw new RuntimeException(
          "Tried to create a new objectracker before releasing the old one!");
    }
    return instance;
  }

  public static synchronized void clearInstance() {
    if (instance != null) {
      instance.release();
    }
  }

  protected ObjectTracker(
      final int frameWidth, final int frameHeight, final int rowStride, final boolean alwaysTrack) {
    this.frameWidth = frameWidth;
    this.frameHeight = frameHeight;
    this.rowStride = rowStride;
    this.alwaysTrack = alwaysTrack;
    this.timestampedDeltas = new LinkedList<TimestampedDeltas>();

    trackedObjects = new HashMap<String, TrackedObject>();

    debugHistory = new Vector<PointF>(MAX_DEBUG_HISTORY_SIZE);

    downsampledFrame =
        new byte
            [(frameWidth + DOWNSAMPLE_FACTOR - 1)
                / DOWNSAMPLE_FACTOR
                * (frameHeight + DOWNSAMPLE_FACTOR - 1)
                / DOWNSAMPLE_FACTOR];
  }

  protected void init() {
    // The native tracker never sees the full frame, so pre-scale dimensions
    // by the downsample factor.
    initNative(frameWidth / DOWNSAMPLE_FACTOR, frameHeight / DOWNSAMPLE_FACTOR, alwaysTrack);
  }

  private final float[] matrixValues = new float[9];

  private long downsampledTimestamp;

  @SuppressWarnings("unused")
  public synchronized void drawOverlay(final GL10 gl,
      final Size cameraViewSize, final Matrix matrix) {
    final Matrix tempMatrix = new Matrix(matrix);
    tempMatrix.preScale(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR);
    tempMatrix.getValues(matrixValues);
    drawNative(cameraViewSize.width, cameraViewSize.height, matrixValues);
  }

  public synchronized void nextFrame(
      final byte[] frameData, final byte[] uvData,
      final long timestamp, final float[] transformationMatrix,
      final boolean updateDebugInfo) {
    if (downsampledTimestamp != timestamp) {
      ObjectTracker.downsampleImageNative(
          frameWidth, frameHeight, rowStride, frameData, DOWNSAMPLE_FACTOR, downsampledFrame);
      downsampledTimestamp = timestamp;
    }

    // Do Lucas Kanade using the fullframe initializer.
    nextFrameNative(downsampledFrame, uvData, timestamp, transformationMatrix);

    timestampedDeltas.add(new TimestampedDeltas(timestamp, getKeypointsPacked(DOWNSAMPLE_FACTOR)));
    while (timestampedDeltas.size() > MAX_FRAME_HISTORY_SIZE) {
      timestampedDeltas.removeFirst();
    }

    for (final TrackedObject trackedObject : trackedObjects.values()) {
      trackedObject.updateTrackedPosition();
    }

    if (updateDebugInfo) {
      updateDebugHistory();
    }

    lastTimestamp = timestamp;
  }

  public synchronized void release() {
    releaseMemoryNative();
    synchronized (ObjectTracker.class) {
      instance = null;
    }
  }

  private void drawHistoryDebug(final Canvas canvas) {
    drawHistoryPoint(
        canvas, frameWidth * DOWNSAMPLE_FACTOR / 2, frameHeight * DOWNSAMPLE_FACTOR / 2);
  }

  private void drawHistoryPoint(final Canvas canvas, final float startX, final float startY) {
    final Paint p = new Paint();
    p.setAntiAlias(false);
    p.setTypeface(Typeface.SERIF);

    p.setColor(Color.RED);
    p.setStrokeWidth(2.0f);

    // Draw the center circle.
    p.setColor(Color.GREEN);
    canvas.drawCircle(startX, startY, 3.0f, p);

    p.setColor(Color.RED);

    // Iterate through in backwards order.
    synchronized (debugHistory) {
      final int numPoints = debugHistory.size();
      float lastX = startX;
      float lastY = startY;
      for (int keypointNum = 0; keypointNum < numPoints; ++keypointNum) {
        final PointF delta = debugHistory.get(numPoints - keypointNum - 1);
        final float newX = lastX + delta.x;
        final float newY = lastY + delta.y;
        canvas.drawLine(lastX, lastY, newX, newY, p);
        lastX = newX;
        lastY = newY;
      }
    }
  }

  private static int floatToChar(final float value) {
    return Math.max(0, Math.min((int) (value * 255.999f), 255));
  }

  private void drawKeypointsDebug(final Canvas canvas) {
    final Paint p = new Paint();
    if (lastKeypoints == null) {
      return;
    }
    final int keypointSize = 3;

    final float minScore = lastKeypoints.minScore;
    final float maxScore = lastKeypoints.maxScore;

    for (final PointChange keypoint : lastKeypoints.pointDeltas) {
      if (keypoint.wasFound) {
        final int r =
            floatToChar((keypoint.keypointA.score - minScore) / (maxScore - minScore));
        final int b =
            floatToChar(1.0f - (keypoint.keypointA.score - minScore) / (maxScore - minScore));

        final int color = 0xFF000000 | (r << 16) | b;
        p.setColor(color);

        final float[] screenPoints = {keypoint.keypointA.x, keypoint.keypointA.y,
                                      keypoint.keypointB.x, keypoint.keypointB.y};
        canvas.drawRect(screenPoints[2] - keypointSize,
                        screenPoints[3] - keypointSize,
                        screenPoints[2] + keypointSize,
                        screenPoints[3] + keypointSize, p);
        p.setColor(Color.CYAN);
        canvas.drawLine(screenPoints[2], screenPoints[3],
                        screenPoints[0], screenPoints[1], p);

        if (DRAW_TEXT) {
          p.setColor(Color.WHITE);
          canvas.drawText(keypoint.keypointA.type + ": " + keypoint.keypointA.score,
              keypoint.keypointA.x, keypoint.keypointA.y, p);
        }
      } else {
        p.setColor(Color.YELLOW);
        final float[] screenPoint = {keypoint.keypointA.x, keypoint.keypointA.y};
        canvas.drawCircle(screenPoint[0], screenPoint[1], 5.0f, p);
      }
    }
  }

  private synchronized PointF getAccumulatedDelta(final long timestamp, final float positionX,
      final float positionY, final float radius) {
    final RectF currPosition = getCurrentPosition(timestamp,
        new RectF(positionX - radius, positionY - radius, positionX + radius, positionY + radius));
    return new PointF(currPosition.centerX() - positionX, currPosition.centerY() - positionY);
  }

  private synchronized RectF getCurrentPosition(final long timestamp, final RectF
      oldPosition) {
    final RectF downscaledFrameRect = downscaleRect(oldPosition);

    final float[] delta = new float[4];
    getCurrentPositionNative(timestamp, downscaledFrameRect.left, downscaledFrameRect.top,
        downscaledFrameRect.right, downscaledFrameRect.bottom, delta);

    final RectF newPosition = new RectF(delta[0], delta[1], delta[2], delta[3]);

    return upscaleRect(newPosition);
  }

  private void updateDebugHistory() {
    lastKeypoints = new FrameChange(getKeypointsNative(false));

    if (lastTimestamp == 0) {
      return;
    }

    final PointF delta =
        getAccumulatedDelta(
            lastTimestamp, frameWidth / DOWNSAMPLE_FACTOR, frameHeight / DOWNSAMPLE_FACTOR, 100);

    synchronized (debugHistory) {
      debugHistory.add(delta);

      while (debugHistory.size() > MAX_DEBUG_HISTORY_SIZE) {
        debugHistory.remove(0);
      }
    }
  }

  public synchronized void drawDebug(final Canvas canvas, final Matrix frameToCanvas) {
    canvas.save();
    canvas.setMatrix(frameToCanvas);

    drawHistoryDebug(canvas);
    drawKeypointsDebug(canvas);

    canvas.restore();
  }

  public Vector<String> getDebugText() {
    final Vector<String> lines = new Vector<String>();

    if (lastKeypoints != null) {
      lines.add("Num keypoints " + lastKeypoints.pointDeltas.size());
      lines.add("Min score: " + lastKeypoints.minScore);
      lines.add("Max score: " + lastKeypoints.maxScore);
    }

    return lines;
  }

  public synchronized List<byte[]> pollAccumulatedFlowData(final long endFrameTime) {
    final List<byte[]> frameDeltas = new ArrayList<byte[]>();
    while (timestampedDeltas.size() > 0) {
      final TimestampedDeltas currentDeltas = timestampedDeltas.peek();
      if (currentDeltas.timestamp <= endFrameTime) {
        frameDeltas.add(currentDeltas.deltas);
        timestampedDeltas.removeFirst();
      } else {
        break;
      }
    }

    return frameDeltas;
  }

  private RectF downscaleRect(final RectF fullFrameRect) {
    return new RectF(
        fullFrameRect.left / DOWNSAMPLE_FACTOR,
        fullFrameRect.top / DOWNSAMPLE_FACTOR,
        fullFrameRect.right / DOWNSAMPLE_FACTOR,
        fullFrameRect.bottom / DOWNSAMPLE_FACTOR);
  }

  private RectF upscaleRect(final RectF downsampledFrameRect) {
    return new RectF(
        downsampledFrameRect.left * DOWNSAMPLE_FACTOR,
        downsampledFrameRect.top * DOWNSAMPLE_FACTOR,
        downsampledFrameRect.right * DOWNSAMPLE_FACTOR,
        downsampledFrameRect.bottom * DOWNSAMPLE_FACTOR);
  }

  /**
   * A TrackedObject represents a native TrackedObject, and provides access to the
   * relevant native tracking information available after every frame update. They may
   * be safely passed around and accessed externally, but will become invalid after
   * stopTracking() is called or the related creating ObjectTracker is deactivated.
   *
   * @author andrewharp@google.com (Andrew Harp)
   */
  public class TrackedObject {
    private final String id;

    private long lastExternalPositionTime;

    private RectF lastTrackedPosition;
    private boolean visibleInLastFrame;

    private boolean isDead;

    TrackedObject(final RectF position, final long timestamp, final byte[] data) {
      isDead = false;

      id = Integer.toString(this.hashCode());

      lastExternalPositionTime = timestamp;

      synchronized (ObjectTracker.this) {
        registerInitialAppearance(position, data);
        setPreviousPosition(position, timestamp);
        trackedObjects.put(id, this);
      }
    }

    public void stopTracking() {
      checkValidObject();

      synchronized (ObjectTracker.this) {
        isDead = true;
        forgetNative(id);
        trackedObjects.remove(id);
      }
    }

    public float getCurrentCorrelation() {
      checkValidObject();
      return ObjectTracker.this.getCurrentCorrelation(id);
    }

    void registerInitialAppearance(final RectF position, final byte[] data) {
      final RectF externalPosition = downscaleRect(position);
      registerNewObjectWithAppearanceNative(id,
            externalPosition.left, externalPosition.top,
            externalPosition.right, externalPosition.bottom,
            data);
    }

    synchronized void setPreviousPosition(final RectF position, final long timestamp) {
      checkValidObject();
      synchronized (ObjectTracker.this) {
        if (lastExternalPositionTime > timestamp) {
          LOGGER.w("Tried to use older position time!");
          return;
        }
        final RectF externalPosition = downscaleRect(position);
        lastExternalPositionTime = timestamp;

        setPreviousPositionNative(id,
            externalPosition.left, externalPosition.top,
            externalPosition.right, externalPosition.bottom,
            lastExternalPositionTime);

        updateTrackedPosition();
      }
    }

    void setCurrentPosition(final RectF position) {
      checkValidObject();
      final RectF downsampledPosition = downscaleRect(position);
      synchronized (ObjectTracker.this) {
        setCurrentPositionNative(id,
            downsampledPosition.left, downsampledPosition.top,
            downsampledPosition.right, downsampledPosition.bottom);
      }
    }

    private synchronized void updateTrackedPosition() {
      checkValidObject();

      final float[] delta = new float[4];
      getTrackedPositionNative(id, delta);
      lastTrackedPosition = new RectF(delta[0], delta[1], delta[2], delta[3]);

      visibleInLastFrame = isObjectVisible(id);
    }

    public synchronized RectF getTrackedPositionInPreviewFrame() {
      checkValidObject();

      if (lastTrackedPosition == null) {
        return null;
      }
      return upscaleRect(lastTrackedPosition);
    }

    synchronized long getLastExternalPositionTime() {
      return lastExternalPositionTime;
    }

    public synchronized boolean visibleInLastPreviewFrame() {
      return visibleInLastFrame;
    }

    private void checkValidObject() {
      if (isDead) {
        throw new RuntimeException("TrackedObject already removed from tracking!");
      } else if (ObjectTracker.this != instance) {
        throw new RuntimeException("TrackedObject created with another ObjectTracker!");
      }
    }
  }

  public synchronized TrackedObject trackObject(
      final RectF position, final long timestamp, final byte[] frameData) {
    if (downsampledTimestamp != timestamp) {
      ObjectTracker.downsampleImageNative(
          frameWidth, frameHeight, rowStride, frameData, DOWNSAMPLE_FACTOR, downsampledFrame);
      downsampledTimestamp = timestamp;
    }
    return new TrackedObject(position, timestamp, downsampledFrame);
  }

  public synchronized TrackedObject trackObject(final RectF position, final byte[] frameData) {
    return new TrackedObject(position, lastTimestamp, frameData);
  }

  /** ********************* NATIVE CODE ************************************ */

  /** This will contain an opaque pointer to the native ObjectTracker */
  private long nativeObjectTracker;

  private native void initNative(int imageWidth, int imageHeight, boolean alwaysTrack);

  protected native void registerNewObjectWithAppearanceNative(
      String objectId, float x1, float y1, float x2, float y2, byte[] data);

  protected native void setPreviousPositionNative(
      String objectId, float x1, float y1, float x2, float y2, long timestamp);

  protected native void setCurrentPositionNative(
      String objectId, float x1, float y1, float x2, float y2);

  protected native void forgetNative(String key);

  protected native String getModelIdNative(String key);

  protected native boolean haveObject(String key);
  protected native boolean isObjectVisible(String key);
  protected native float getCurrentCorrelation(String key);

  protected native float getMatchScore(String key);

  protected native void getTrackedPositionNative(String key, float[] points);

  protected native void nextFrameNative(
      byte[] frameData, byte[] uvData, long timestamp, float[] frameAlignMatrix);

  protected native void releaseMemoryNative();

  protected native void getCurrentPositionNative(long timestamp,
      final float positionX1, final float positionY1,
      final float positionX2, final float positionY2,
      final float[] delta);

  protected native byte[] getKeypointsPacked(float scaleFactor);

  protected native float[] getKeypointsNative(boolean onlyReturnCorrespondingKeypoints);

  protected native void drawNative(int viewWidth, int viewHeight, float[] frameToCanvas);

  protected static native void downsampleImageNative(
      int width, int height, int rowStride, byte[] input, int factor, byte[] output);
}
