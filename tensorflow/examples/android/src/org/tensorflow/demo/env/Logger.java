/* Copyright 2015 Google Inc. All Rights Reserved.

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

package org.tensorflow.demo.env;

import android.util.Log;

import java.util.HashSet;
import java.util.Set;

/**
 * Wrapper for the platform log function, allows convenient message prefixing and log disabling.
 */
public final class Logger {
  private static final String DEFAULT_TAG = "tensorflow";
  private static final int DEFAULT_MIN_LOG_LEVEL = Log.DEBUG;

  // Classes to be ignored when examining the stack trace
  private static final Set<String> IGNORED_CLASS_NAMES;

  static {
    IGNORED_CLASS_NAMES = new HashSet<String>(3);
    IGNORED_CLASS_NAMES.add("dalvik.system.VMStack");
    IGNORED_CLASS_NAMES.add("java.lang.Thread");
    IGNORED_CLASS_NAMES.add(Logger.class.getCanonicalName());
  }

  private final String tag;
  private final String messagePrefix;
  private int minLogLevel = DEFAULT_MIN_LOG_LEVEL;

  /**
   * Creates a Logger using the class name as the message prefix.
   *
   * @param clazz the simple name of this class is used as the message prefix.
   */
  public Logger(final Class<?> clazz) {
    this(clazz.getSimpleName());
  }

  /**
   * Creates a Logger using the specified message prefix.
   *
   * @param messagePrefix is prepended to the text of every message.
   */
  public Logger(final String messagePrefix) {
    this(DEFAULT_TAG, messagePrefix);
  }

  /**
   * Creates a Logger with a custom tag and a custom message prefix. If the message prefix
   * is set to <pre>null</pre>, the caller's class name is used as the prefix.
   *
   * @param tag identifies the source of a log message.
   * @param messagePrefix prepended to every message if non-null. If null, the name of the caller is
   *                      being used
   */
  public Logger(final String tag, final String messagePrefix) {
    this.tag = tag;
    final String prefix = messagePrefix == null ? getCallerSimpleName() : messagePrefix;
    this.messagePrefix = (prefix.length() > 0) ? prefix + ": " : prefix;
  }

  /**
   * Creates a Logger using the caller's class name as the message prefix.
   */
  public Logger() {
    this(DEFAULT_TAG, null);
  }

  /**
   * Creates a Logger using the caller's class name as the message prefix.
   */
  public Logger(final int minLogLevel) {
    this(DEFAULT_TAG, null);
    this.minLogLevel = minLogLevel;
  }

  public void setMinLogLevel(final int minLogLevel) {
    this.minLogLevel = minLogLevel;
  }

  public boolean isLoggable(final int logLevel) {
    return logLevel >= minLogLevel || Log.isLoggable(tag, logLevel);
  }

  /**
   * Return caller's simple name.
   *
   * Android getStackTrace() returns an array that looks like this:
   *     stackTrace[0]: dalvik.system.VMStack
   *     stackTrace[1]: java.lang.Thread
   *     stackTrace[2]: com.google.android.apps.unveil.env.UnveilLogger
   *     stackTrace[3]: com.google.android.apps.unveil.BaseApplication
   *
   * This function returns the simple version of the first non-filtered name.
   *
   * @return caller's simple name
   */
  private static String getCallerSimpleName() {
    // Get the current callstack so we can pull the class of the caller off of it.
    final StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();

    for (final StackTraceElement elem : stackTrace) {
      final String className = elem.getClassName();
      if (!IGNORED_CLASS_NAMES.contains(className)) {
        // We're only interested in the simple name of the class, not the complete package.
        final String[] classParts = className.split("\\.");
        return classParts[classParts.length - 1];
      }
    }

    return Logger.class.getSimpleName();
  }

  private String toMessage(final String format, final Object... args) {
    return messagePrefix + (args.length > 0 ? String.format(format, args) : format);
  }

  public void v(final String format, final Object... args) {
    if (isLoggable(Log.VERBOSE)) {
      Log.v(tag, toMessage(format, args));
    }
  }

  public void v(final Throwable t, final String format, final Object... args) {
    if (isLoggable(Log.VERBOSE)) {
      Log.v(tag, toMessage(format, args), t);
    }
  }

  public void d(final String format, final Object... args) {
    if (isLoggable(Log.DEBUG)) {
      Log.d(tag, toMessage(format, args));
    }
  }

  public void d(final Throwable t, final String format, final Object... args) {
    if (isLoggable(Log.DEBUG)) {
      Log.d(tag, toMessage(format, args), t);
    }
  }

  public void i(final String format, final Object... args) {
    if (isLoggable(Log.INFO)) {
      Log.i(tag, toMessage(format, args));
    }
  }

  public void i(final Throwable t, final String format, final Object... args) {
    if (isLoggable(Log.INFO)) {
      Log.i(tag, toMessage(format, args), t);
    }
  }

  public void w(final String format, final Object... args) {
    if (isLoggable(Log.WARN)) {
      Log.w(tag, toMessage(format, args));
    }
  }

  public void w(final Throwable t, final String format, final Object... args) {
    if (isLoggable(Log.WARN)) {
      Log.w(tag, toMessage(format, args), t);
    }
  }

  public void e(final String format, final Object... args) {
    if (isLoggable(Log.ERROR)) {
      Log.e(tag, toMessage(format, args));
    }
  }

  public void e(final Throwable t, final String format, final Object... args) {
    if (isLoggable(Log.ERROR)) {
      Log.e(tag, toMessage(format, args), t);
    }
  }
}
