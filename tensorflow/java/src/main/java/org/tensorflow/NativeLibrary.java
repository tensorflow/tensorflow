/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Helper class for loading the TensorFlow Java native library.
 *
 * <p>The Java TensorFlow bindings require a native (JNI) library. This library
 * (libtensorflow_jni.so on Linux, libtensorflow_jni.dylib on OS X, tensorflow_jni.dll on Windows)
 * can be made available to the JVM using the java.library.path System property (e.g., using
 * -Djava.library.path command-line argument). However, doing so requires an additional step of
 * configuration.
 *
 * <p>Alternatively, the native libraries can be packaed in a .jar, making them easily usable from
 * build systems like Maven. However, in such cases, the native library has to be extracted from the
 * .jar archive.
 *
 * <p>NativeLibrary.load() takes care of this. First looking for the library in java.library.path
 * and failing that, it tries to find the OS and architecture specific version of the library in the
 * set of ClassLoader resources (under org/tensorflow/native/OS-ARCH). The resources paths used for
 * lookup must be consistent with any packaging (such as on Maven Central) of the TensorFlow Java
 * native libraries.
 */
final class NativeLibrary {
  private static final boolean DEBUG =
      System.getProperty("org.tensorflow.NativeLibrary.DEBUG") != null;
  private static final String JNI_LIBNAME = "tensorflow_jni";

  public static void load() {
    if (isLoaded() || tryLoadLibrary()) {
      // Either:
      // (1) The native library has already been statically loaded, OR
      // (2) The required native code has been statically linked (through a custom launcher), OR
      // (3) The native code is part of another library (such as an application-level library)
      // that has already been loaded. For example, tensorflow/examples/android and
      // tensorflow/contrib/android include the required native code in differently named libraries.
      //
      // Doesn't matter how, but it seems the native code is loaded, so nothing else to do.
      return;
    }
    // Native code is not present, perhaps it has been packaged into the .jar file containing this.
    // Extract the JNI library itself
    final String jniLibName = System.mapLibraryName(JNI_LIBNAME);
    final String jniResourceName = makeResourceName(jniLibName);
    log("jniResourceName: " + jniResourceName);
    final InputStream jniResource =
        NativeLibrary.class.getClassLoader().getResourceAsStream(jniResourceName);
    // Extract the JNI's dependency
    final String frameworkLibName =
        maybeAdjustForMacOS(System.mapLibraryName("tensorflow_framework"));
    final String frameworkResourceName = makeResourceName(frameworkLibName);
    log("frameworkResourceName: " + frameworkResourceName);
    final InputStream frameworkResource =
        NativeLibrary.class.getClassLoader().getResourceAsStream(frameworkResourceName);
    // Do not complain if the framework resource wasn't found. This may just mean that we're
    // building with --config=monolithic (in which case it's not needed and not included).
    if (jniResource == null) {
      throw new UnsatisfiedLinkError(
          String.format(
              "Cannot find TensorFlow native library for OS: %s, architecture: %s. See "
                  + "https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java/README.md"
                  + " for possible solutions (such as building the library from source). Additional"
                  + " information on attempts to find the native library can be obtained by adding"
                  + " org.tensorflow.NativeLibrary.DEBUG=1 to the system properties of the JVM.",
              os(), architecture()));
    }
    try {
      // Create a temporary directory for the extracted resource and its dependencies.
      final File tempPath = createTemporaryDirectory();
      // Deletions are in the reverse order of requests, so we need to request that the directory be
      // deleted first, so that it is empty when the request is fulfilled.
      tempPath.deleteOnExit();
      final String tempDirectory = tempPath.getCanonicalPath();
      if (frameworkResource != null) {
        extractResource(frameworkResource, frameworkLibName, tempDirectory);
      } else {
        log(
            frameworkResourceName
                + " not found. This is fine assuming "
                + jniResourceName
                + " is not built to depend on it.");
      }
      System.load(extractResource(jniResource, jniLibName, tempDirectory));
    } catch (IOException e) {
      throw new UnsatisfiedLinkError(
          String.format(
              "Unable to extract native library into a temporary file (%s)", e.toString()));
    }
  }

  private static boolean tryLoadLibrary() {
    try {
      System.loadLibrary(JNI_LIBNAME);
      return true;
    } catch (UnsatisfiedLinkError e) {
      log("tryLoadLibraryFailed: " + e.getMessage());
      return false;
    }
  }

  private static boolean isLoaded() {
    try {
      TensorFlow.version();
      log("isLoaded: true");
      return true;
    } catch (UnsatisfiedLinkError e) {
      return false;
    }
  }

  private static String maybeAdjustForMacOS(String libFilename) {
    if (!System.getProperty("os.name").contains("OS X")) {
      return libFilename;
    }
    // This is macOS, and the TensorFlow release process might have setup dependencies on
    // libtensorflow_framework.so instead of libtensorflow_framework.dylib. Adjust for that.
    final ClassLoader cl = NativeLibrary.class.getClassLoader();
    if (cl.getResource(makeResourceName(libFilename)) != null) {
      return libFilename;
    }
    // liftensorflow_framework.dylib not found, try libtensorflow_framework.so
    final String suffix = ".dylib";
    if (!libFilename.endsWith(suffix)) {
      return libFilename;
    }
    return libFilename.substring(0, libFilename.length() - suffix.length()) + ".so";
  }

  private static String extractResource(
      InputStream resource, String resourceName, String extractToDirectory) throws IOException {
    final File dst = new File(extractToDirectory, resourceName);
    dst.deleteOnExit();
    final String dstPath = dst.toString();
    log("extracting native library to: " + dstPath);
    final long nbytes = copy(resource, dst);
    log(String.format("copied %d bytes to %s", nbytes, dstPath));
    return dstPath;
  }

  private static String os() {
    final String p = System.getProperty("os.name").toLowerCase();
    if (p.contains("linux")) {
      return "linux";
    } else if (p.contains("os x") || p.contains("darwin")) {
      return "darwin";
    } else if (p.contains("windows")) {
      return "windows";
    } else {
      return p.replaceAll("\\s", "");
    }
  }

  private static String architecture() {
    final String arch = System.getProperty("os.arch").toLowerCase();
    return (arch.equals("amd64")) ? "x86_64" : arch;
  }

  private static void log(String msg) {
    if (DEBUG) {
      System.err.println("org.tensorflow.NativeLibrary: " + msg);
    }
  }

  private static String makeResourceName(String baseName) {
    return "org/tensorflow/native/" + String.format("%s-%s/", os(), architecture()) + baseName;
  }

  private static long copy(InputStream src, File dstFile) throws IOException {
    FileOutputStream dst = new FileOutputStream(dstFile);
    try {
      byte[] buffer = new byte[1 << 20]; // 1MB
      long ret = 0;
      int n = 0;
      while ((n = src.read(buffer)) >= 0) {
        dst.write(buffer, 0, n);
        ret += n;
      }
      return ret;
    } finally {
      dst.close();
      src.close();
    }
  }

  // Shamelessly adapted from Guava to avoid using java.nio, for Android API
  // compatibility.
  private static File createTemporaryDirectory() {
    File baseDirectory = new File(System.getProperty("java.io.tmpdir"));
    String directoryName = "tensorflow_native_libraries-" + System.currentTimeMillis() + "-";
    for (int attempt = 0; attempt < 1000; attempt++) {
      File temporaryDirectory = new File(baseDirectory, directoryName + attempt);
      if (temporaryDirectory.mkdir()) {
        return temporaryDirectory;
      }
    }
    throw new IllegalStateException(
        "Could not create a temporary directory (tried to make "
            + directoryName
            + "*) to extract TensorFlow native libraries.");
  }

  private NativeLibrary() {}
}
