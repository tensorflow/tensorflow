/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.common;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import org.checkerframework.checker.nullness.qual.NonNull;

/** File I/O utilities. */
public class FileUtil {
  private FileUtil() {}

  /**
   * Loads labels from the label file into a list of strings.
   *
   * <p>A legal label file is the plain text file whose contents are split into lines, and each line
   * is an individual value. The file should be in assets of the context.
   *
   * @param context The context holds assets.
   * @param filePath The path of the label file, relative with assets directory.
   * @return a list of labels.
   * @throws IOException if error occurs to open or read the file.
   */
  @NonNull
  public static List<String> loadLabels(@NonNull Context context, @NonNull String filePath)
      throws IOException {
    return loadLabels(context, filePath, Charset.defaultCharset());
  }

  /**
   * Loads labels from the label file into a list of strings.
   *
   * <p>A legal label file is the plain text file whose contents are split into lines, and each line
   * is an individual value. The empty lines will be ignored. The file should be in assets of the
   * context.
   *
   * @param context The context holds assets.
   * @param filePath The path of the label file, relative with assets directory.
   * @param cs {@code Charset} to use when decoding content of label file.
   * @return a list of labels.
   * @throws IOException if error occurs to open or read the file.
   */
  @NonNull
  public static List<String> loadLabels(
      @NonNull Context context, @NonNull String filePath, Charset cs) throws IOException {
    SupportPreconditions.checkNotNull(context, "Context cannot be null.");
    SupportPreconditions.checkNotNull(filePath, "File path cannot be null.");
    try (InputStream inputStream = context.getAssets().open(filePath)) {
      return loadLabels(inputStream, cs);
    }
  }

  /**
   * Loads labels from an input stream of an opened label file. See details for label files in
   * {@link FileUtil#loadLabels(Context, String)}.
   *
   * @param inputStream the input stream of an opened label file.
   * @return a list of labels.
   * @throws IOException if error occurs to open or read the file.
   */
  @NonNull
  public static List<String> loadLabels(@NonNull InputStream inputStream) throws IOException {
    return loadLabels(inputStream, Charset.defaultCharset());
  }

  /**
   * Loads labels from an input stream of an opened label file. See details for label files in
   * {@link FileUtil#loadLabels(Context, String)}.
   *
   * @param inputStream the input stream of an opened label file.
   * @param cs {@code Charset} to use when decoding content of label file.
   * @return a list of labels.
   * @throws IOException if error occurs to open or read the file.
   */
  @NonNull
  public static List<String> loadLabels(@NonNull InputStream inputStream, Charset cs)
      throws IOException {
    List<String> labels = new ArrayList<>();
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, cs))) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.trim().length() > 0) {
          labels.add(line);
        }
      }
      return labels;
    }
  }

  /**
   * Loads a vocabulary file (a single-column text file) into a list of strings.
   *
   * <p>A vocabulary file is a single-column plain text file whose contents are split into lines,
   * and each line is an individual value. The file should be in assets of the context.
   *
   * @param context The context holds assets.
   * @param filePath The path of the vocabulary file, relative with assets directory.
   * @return a list of vocabulary words.
   * @throws IOException if error occurs to open or read the file.
   */
  @NonNull
  public static List<String> loadSingleColumnTextFile(
      @NonNull Context context, @NonNull String filePath, Charset cs) throws IOException {
    return loadLabels(context, filePath, cs);
  }

  /**
   * Loads vocabulary from an input stream of an opened vocabulary file (which is a single-column
   * text file). See details for vocabulary files in {@link FileUtil#loadVocabularyFile(Context,
   * String)}.
   *
   * @param inputStream the input stream of an opened vocabulary file.
   * @return a list of vocabulary words.
   * @throws IOException if error occurs to open or read the file.
   */
  @NonNull
  public static List<String> loadSingleColumnTextFile(@NonNull InputStream inputStream, Charset cs)
      throws IOException {
    return loadLabels(inputStream, cs);
  }

  /**
   * Loads a file from the asset folder through memory mapping.
   *
   * @param context Application context to access assets.
   * @param filePath Asset path of the file.
   * @return the loaded memory mapped file.
   * @throws IOException if an I/O error occurs when loading the tflite model.
   */
  @NonNull
  public static MappedByteBuffer loadMappedFile(@NonNull Context context, @NonNull String filePath)
      throws IOException {
    SupportPreconditions.checkNotNull(context, "Context should not be null.");
    SupportPreconditions.checkNotNull(filePath, "File path cannot be null.");
    try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(filePath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /**
   * Loads a binary file from the asset folder.
   *
   * @param context Application context to access assets.
   * @param filePath Asset path of the file.
   * @return the byte array for the binary file.
   * @throws IOException if an I/O error occurs when loading file.
   */
  @NonNull
  public static byte[] loadByteFromFile(@NonNull Context context, @NonNull String filePath)
      throws IOException {
    ByteBuffer buffer = loadMappedFile(context, filePath);
    byte[] byteArray = new byte[buffer.remaining()];
    buffer.get(byteArray);
    return byteArray;
  }
}
