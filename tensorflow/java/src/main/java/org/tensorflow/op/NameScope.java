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

package org.tensorflow.op;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * A class to manage scoped (hierarchical) names for operators.
 *
 * <p>{@code NameScope} manages hierarchical names where each component in the hierarchy is
 * separated by a forward slash {@code '/'}. For instance, {@code nn/Const_72} or {@code
 * nn/gradient/assign/init}. Each scope is a subtree in this hierarchy.
 *
 * <p>Use {@code NameScope} to group related operations within a hierarchy, which for example lets
 * tensorboard coalesce nodes for better graph visualizations.
 *
 * <p>This class is package private, user code creates {@link Scope} which internally delegates
 * calls to an underlying {@code NameScope}.
 *
 * <p>This class is <b>not</b> thread-safe.
 */
final class NameScope {

  NameScope withSubScope(String scopeName) {
    checkPattern(NAME_REGEX, scopeName);
    // Override with opName if it exists.
    String actualName = (opName != null) ? opName : scopeName;
    String newPrefix = fullyQualify(makeUnique(actualName));
    return new NameScope(newPrefix, null, null);
  }

  NameScope withName(String name) {
    checkPattern(NAME_REGEX, name);
    // All context except for the opName is shared with the new scope.
    return new NameScope(opPrefix, name, ids);
  }

  String makeOpName(String name) {
    checkPattern(NAME_REGEX, name);
    // Override with opName if it exists.
    String actualName = (opName != null) ? opName : name;
    return fullyQualify(makeUnique(actualName));
  }

  /**
   * Create a new, root-level namescope.
   *
   * <p>A root-level namescope generates operator names with no components, like {@code Const_72}
   * and {@code result}.
   */
  NameScope() {
    this(null, null, null);
  }

  private NameScope(String opPrefix, String opName, Map<String, Integer> ids) {
    this.opPrefix = opPrefix;
    this.opName = opName;
    if (ids != null) {
      this.ids = ids;
    } else {
      this.ids = new HashMap<String, Integer>();
    }
  }

  // Generate a unique name, different from existing ids.
  //
  // ids is a map from id to integer, representing a counter of the
  // number of previous requests to generate a unique name for the
  // given id.
  //
  // For instance, the first use of makeUnique("a") adds "a" -> 1
  // to ids and returns "a".
  //
  // The second use of makeUnique("a") updates ids to "a" -> 2
  // and returns "a_1", and so on.
  private String makeUnique(String id) {
    if (!ids.containsKey(id)) {
      ids.put(id, 1);
      return id;
    } else {
      int cur = ids.get(id);
      ids.put(id, cur + 1);
      return String.format("%s_%d", id, cur);
    }
  }

  private String fullyQualify(String name) {
    if (opPrefix != null) {
      return String.format("%s/%s", opPrefix, name);
    } else {
      return name;
    }
  }

  // If opPrefix is non-null, it is a prefix applied to all names
  // created by this instance.
  private final String opPrefix;

  // If opName is non-null, it is used to derive the unique name
  // for operators rather than the provided default name.
  private final String opName;

  // NameScope generates unique names by appending a numeric suffix if
  // needed. This is a map containing names already created by this
  // instance mapped to the next available numeric suffix for it.
  private final Map<String, Integer> ids;

  private static void checkPattern(Pattern pattern, String name) {
    if (name == null) {
      throw new IllegalArgumentException("Names cannot be null");
    }
    if (!pattern.matcher(name).matches()) {
      throw new IllegalArgumentException(
          String.format(
              "invalid name: '%s' does not match the regular expression %s",
              name, NAME_REGEX.pattern()));
    }
  }

  // The constraints for operator and scope names originate from restrictions on node names
  // noted in the proto definition core/framework/node_def.proto for NodeDef and actually
  // implemented in core/framework/node_def_util.cc [Note that the proto comment does not include
  // dash (-) in names, while the actual implementation permits it. This regex follows the actual
  // implementation.]
  //
  // This pattern is used to ensure fully qualified names always start with a LETTER_DIGIT_DOT,
  // followed by zero or more LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE. SLASH is not permitted in
  // actual user-supplied names to NameScope - it is used as a reserved character to separate
  // subcomponents within fully qualified names.
  private static final Pattern NAME_REGEX = Pattern.compile("[A-Za-z0-9.][A-Za-z0-9_.\\-]*");
}
