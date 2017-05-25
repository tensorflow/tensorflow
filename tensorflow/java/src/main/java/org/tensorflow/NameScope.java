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
 * <p>This class is thread-safe.
 */
final class NameScope {

  NameScope withSubScope(String scopeName) {
    if (baseName == null) {
      checkPattern(OP_NAME_REGEX, scopeName);
    } else {
      checkPattern(ROOT_SCOPE_NAME_REGEX, scopeName);
    }

    if (baseOpName != null) {
      // Use the base name instead to derive the subscope.
      scopeName = baseOpName;
    }

    String newBaseName = fullyQualify(makeUnique(scopeName));
    return NameScope.builder().baseName(newBaseName).build();
  }

  NameScope withOpName(String name) {
    checkPattern(OP_NAME_REGEX, name);

    // All context except for the baseOpName is shared with the new scope.
    return NameScope.builder().ids(ids).baseName(baseName).baseOpName(name).build();
  }

  String makeOpName(String opName) {
    checkPattern(OP_NAME_REGEX, opName);

    if (baseOpName != null) {
      opName = baseOpName;
    }
    opName = makeUnique(opName);
    return fullyQualify(opName);
  }

  /**
   * Create a new, root-level namescope.
   *
   * <p>A root-level namescope generates operator names with no components, like {@code Const_72}
   * and {@code result}.
   *
   * @return a NameScope that generates top-level names.
   */
  static NameScope create() {
    return NameScope.builder().build();
  }

  private NameScope(Builder builder) {
    baseName = builder.baseName;
    baseOpName = builder.baseOpName;
    if (builder.ids != null) {
      ids = builder.ids;
    } else {
      ids = new HashMap<String, Integer>();
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
    synchronized (ids) {
      if (!ids.containsKey(id)) {
        ids.put(id, 1);
        return id;
      } else {
        int cur = ids.get(id);
        ids.put(id, cur + 1);
        return String.format("%s_%d", id, cur);
      }
    }
  }

  private String fullyQualify(String name) {
    if (baseName != null) {
      return String.format("%s/%s", baseName, name);
    } else {
      return name;
    }
  }

  // If baseName is non-null, it is a prefix applied to all names
  // created by this instance.
  private final String baseName;

  // If baseOpName is non-null, it is used to derive the unique name
  // for operators rather than the provided default name.
  private final String baseOpName;

  // NameScope generates unique names by appending a numeric suffix if
  // needed. This is a map containing names already created by this
  // instance mapped to the next available numeric suffix for it.
  private final Map<String, Integer> ids;

  private static void checkPattern(Pattern pattern, String name) {
    if (name == null) {
      throw new IllegalArgumentException("Names cannot be null");
    }
    if (!pattern.matcher(name).matches()) {
      throw new IllegalArgumentException(String.format("Invalid name '%s'", name));
    }
  }

  private static final Pattern OP_NAME_REGEX = Pattern.compile("[A-Za-z0-9.][A-Za-z0-9_.\\-]*");
  private static final Pattern ROOT_SCOPE_NAME_REGEX = Pattern.compile("[A-Za-z0-9_.\\-]+");

  private static Builder builder() {
    return new Builder();
  }

  private static final class Builder {
    private Builder() {}

    private Builder baseName(String name) {
      baseName = name;
      return this;
    }

    private Builder baseOpName(String name) {
      baseOpName = name;
      return this;
    }

    private Builder ids(Map<String, Integer> map) {
      ids = map;
      return this;
    }

    private NameScope build() {
      return new NameScope(this);
    }

    private String baseName = null;
    private String baseOpName = null;
    private Map<String, Integer> ids = null;
  }
}
