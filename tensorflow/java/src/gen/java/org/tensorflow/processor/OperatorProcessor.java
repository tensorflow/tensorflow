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

package org.tensorflow.processor;

import com.google.common.base.CaseFormat;
import com.google.common.base.Strings;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.FieldSpec;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import com.squareup.javapoet.TypeVariableName;
import com.squareup.javapoet.WildcardTypeName;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Filer;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.TypeParameterElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.Elements;
import javax.tools.Diagnostic.Kind;

/**
 * A compile-time Processor that aggregates classes annotated with {@link
 * org.tensorflow.op.annotation.Operator} and generates the {@code Ops} convenience API. Please
 * refer to the {@link org.tensorflow.op.annotation.Operator} annotation for details about the API
 * generated for each annotated class.
 *
 * <p>Note that this processor can only be invoked once, in a single compilation run that includes
 * all the {@code Operator} annotated source classes. The reason is that the {@code Ops} API is an
 * "aggregating" API, and annotation processing does not permit modifying an already generated
 * class.
 *
 * @see org.tensorflow.op.annotation.Operator
 */
public final class OperatorProcessor extends AbstractProcessor {

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latest();
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    messager = processingEnv.getMessager();
    filer = processingEnv.getFiler();
    elements = processingEnv.getElementUtils();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    // Nothing needs to be done at the end of all rounds.
    if (roundEnv.processingOver()) {
      return false;
    }

    // Nothing to look at in this round.
    if (annotations.size() == 0) {
      return false;
    }

    // We expect to be registered for exactly one annotation.
    if (annotations.size() != 1) {
      throw new IllegalStateException(
          "Unexpected - multiple annotations registered: " + annotations);
    }
    TypeElement annotation = annotations.iterator().next();
    Set<? extends Element> annotated = roundEnv.getElementsAnnotatedWith(annotation);

    // If there are no annotated elements, claim the annotation but do nothing.
    if (annotated.size() == 0) {
      return false;
    }

    // This processor has to aggregate all op classes in one round, as it generates a single Ops
    // API class which cannot be modified once generated. If we find an annotation after we've
    // generated our code, flag the location of each such class.
    if (hasRun) {
      for (Element e : annotated) {
        error(
            e,
            "The Operator processor has already processed @Operator annotated sources\n"
                + "and written out an Ops API. It cannot process additional @Operator sources.\n"
                + "One reason this can happen is if other annotation processors generate\n"
                + "new @Operator source files.");
      }
      return false;
    }

    // Collect all classes tagged with our annotation.
    Multimap<String, MethodSpec> groupedMethods = HashMultimap.create();
    if (!collectOpsMethods(roundEnv, groupedMethods, annotation)) {
      return false;
    }

    // Nothing to do when there are no tagged classes.
    if (groupedMethods.isEmpty()) {
      return false;
    }

    // Validate operator classes and generate Op API.
    writeApi(groupedMethods);

    hasRun = true;
    return false;
  }

  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return Collections.singleton("org.tensorflow.op.annotation.Operator");
  }

  private static final Pattern JAVADOC_TAG_PATTERN =
      Pattern.compile("@(?:param|return|throws|exception|see)\\s+.*");
  private static final TypeName T_OP = ClassName.get("org.tensorflow.op", "Op");
  private static final TypeName T_OPS = ClassName.get("org.tensorflow.op", "Ops");
  private static final TypeName T_OPERATOR =
      ClassName.get("org.tensorflow.op.annotation", "Operator");
  private static final TypeName T_SCOPE = ClassName.get("org.tensorflow.op", "Scope");
  private static final TypeName T_EXEC_ENV =
      ClassName.get("org.tensorflow", "ExecutionEnvironment");
  private static final TypeName T_EAGER_SESSION = ClassName.get("org.tensorflow", "EagerSession");
  private static final TypeName T_STRING = ClassName.get(String.class);
  // Operand<?>
  private static final TypeName T_OPERAND =
      ParameterizedTypeName.get(
          ClassName.get("org.tensorflow", "Operand"), WildcardTypeName.subtypeOf(Object.class));
  // Iterable<Operand<?>>
  private static final TypeName T_ITERABLE_OPERAND =
      ParameterizedTypeName.get(ClassName.get(Iterable.class), T_OPERAND);

  private Filer filer;
  private Messager messager;
  private Elements elements;
  private boolean hasRun = false;

  private void error(Element e, String message, Object... args) {
    if (args != null && args.length > 0) {
      message = String.format(message, args);
    }
    messager.printMessage(Kind.ERROR, message, e);
  }

  private void write(TypeSpec spec) {
    try {
      JavaFile.builder("org.tensorflow.op", spec).skipJavaLangImports(true).build().writeTo(filer);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  private void writeApi(Multimap<String, MethodSpec> groupedMethods) {
    Map<String, ClassName> groups = new HashMap<>();

    // Generate a API class for each group collected other than the default one (= empty string)
    for (Map.Entry<String, Collection<MethodSpec>> entry : groupedMethods.asMap().entrySet()) {
      if (!entry.getKey().isEmpty()) {
        TypeSpec groupClass = buildGroupClass(entry.getKey(), entry.getValue());
        write(groupClass);
        groups.put(entry.getKey(), ClassName.get("org.tensorflow.op", groupClass.name));
      }
    }
    // Generate the top API class, adding any methods added to the default group
    TypeSpec topClass = buildTopClass(groups, groupedMethods.get(""));
    write(topClass);
  }

  private boolean collectOpsMethods(
      RoundEnvironment roundEnv,
      Multimap<String, MethodSpec> groupedMethods,
      TypeElement annotation) {
    boolean result = true;
    for (Element e : roundEnv.getElementsAnnotatedWith(annotation)) {
      // @Operator can only apply to types, so e must be a TypeElement.
      if (!(e instanceof TypeElement)) {
        error(
            e,
            "@Operator can only be applied to classes, but this is a %s",
            e.getKind().toString());
        result = false;
        continue;
      }
      TypeElement opClass = (TypeElement) e;
      // Skip deprecated operations for now, as we do not guarantee API stability yet
      if (opClass.getAnnotation(Deprecated.class) == null) {
        collectOpMethods(groupedMethods, opClass, annotation);
      }
    }
    return result;
  }

  private void collectOpMethods(
      Multimap<String, MethodSpec> groupedMethods, TypeElement opClass, TypeElement annotation) {
    AnnotationMirror am = getAnnotationMirror(opClass, annotation);
    String groupName = getAnnotationElementValueAsString("group", am);
    String methodName = getAnnotationElementValueAsString("name", am);
    ClassName opClassName = ClassName.get(opClass);
    if (Strings.isNullOrEmpty(methodName)) {
      methodName = CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_CAMEL, opClassName.simpleName());
    }
    // Build a method for each @Operator found in the class path. There should be one method per
    // operation factory called
    // "create", which takes in parameter a scope and, optionally, a list of arguments
    for (ExecutableElement opMethod : ElementFilter.methodsIn(opClass.getEnclosedElements())) {
      if (opMethod.getModifiers().contains(Modifier.STATIC)
          && opMethod.getSimpleName().contentEquals("create")) {
        MethodSpec method = buildOpMethod(methodName, opClassName, opMethod);
        groupedMethods.put(groupName, method);
      }
    }
  }

  private MethodSpec buildOpMethod(
      String methodName, ClassName opClassName, ExecutableElement factoryMethod) {
    MethodSpec.Builder builder =
        MethodSpec.methodBuilder(methodName)
            .addModifiers(Modifier.PUBLIC)
            .returns(TypeName.get(factoryMethod.getReturnType()))
            .varargs(factoryMethod.isVarArgs())
            .addJavadoc("$L", buildOpMethodJavadoc(opClassName, factoryMethod));

    for (TypeParameterElement tp : factoryMethod.getTypeParameters()) {
      TypeVariableName tvn = TypeVariableName.get((TypeVariable) tp.asType());
      builder.addTypeVariable(tvn);
    }
    for (TypeMirror thrownType : factoryMethod.getThrownTypes()) {
      builder.addException(TypeName.get(thrownType));
    }
    StringBuilder call = new StringBuilder("return $T.create(scope");
    boolean first = true;
    for (VariableElement param : factoryMethod.getParameters()) {
      ParameterSpec p = ParameterSpec.get(param);
      if (first) {
        first = false;
        continue;
      }
      call.append(", ");
      call.append(p.name);
      builder.addParameter(p);
    }
    call.append(")");
    builder.addStatement(call.toString(), opClassName);
    return builder.build();
  }

  private String buildOpMethodJavadoc(ClassName opClassName, ExecutableElement factoryMethod) {
    StringBuilder javadoc = new StringBuilder();
    javadoc.append("Builds an {@link ").append(opClassName.simpleName()).append("} operation\n\n");

    // Add all javadoc tags found in the operator factory method but the first one, which should be
    // in all cases the
    // 'scope' parameter that is implicitly passed by this API
    Matcher tagMatcher = JAVADOC_TAG_PATTERN.matcher(elements.getDocComment(factoryMethod));
    boolean firstParam = true;

    while (tagMatcher.find()) {
      String tag = tagMatcher.group();
      if (tag.startsWith("@param") && firstParam) {
        firstParam = false;
      } else {
        javadoc.append(tag).append('\n');
      }
    }
    javadoc.append("@see ").append(opClassName).append("\n");

    return javadoc.toString();
  }

  private static TypeSpec buildGroupClass(String group, Collection<MethodSpec> methods) {
    MethodSpec.Builder ctorBuilder =
        MethodSpec.constructorBuilder()
            .addParameter(T_SCOPE, "scope")
            .addStatement("this.scope = scope");

    TypeSpec.Builder builder =
        TypeSpec.classBuilder(CaseFormat.LOWER_CAMEL.to(CaseFormat.UPPER_CAMEL, group) + "Ops")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addJavadoc(
                "An API for building {@code $L} operations as {@link $T Op}s\n\n"
                    + "@see {@link $T}\n",
                group,
                T_OP,
                T_OPS)
            .addMethods(methods)
            .addMethod(ctorBuilder.build());

    builder.addField(
        FieldSpec.builder(T_SCOPE, "scope").addModifiers(Modifier.PRIVATE, Modifier.FINAL).build());

    return builder.build();
  }

  private static TypeSpec buildTopClass(
      Map<String, ClassName> groupToClass, Collection<MethodSpec> methods) {
    MethodSpec.Builder ctorBuilder =
        MethodSpec.constructorBuilder()
            .addModifiers(Modifier.PRIVATE)
            .addParameter(T_SCOPE, "scope")
            .addStatement("this.scope = scope", T_SCOPE);

    for (Map.Entry<String, ClassName> entry : groupToClass.entrySet()) {
      ctorBuilder.addStatement("$L = new $T(scope)", entry.getKey(), entry.getValue());
    }

    TypeSpec.Builder opsBuilder =
        TypeSpec.classBuilder("Ops")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .addJavadoc(
                "An API for building operations as {@link $T Op}s\n<p>\n"
                    + "Any operation wrapper found in the classpath properly annotated as an"
                    + "{@link $T @Operator} is exposed\n"
                    + "by this API or one of its subgroup.\n<p>Example usage:\n<pre>{@code\n"
                    + "try (Graph g = new Graph()) {\n"
                    + "  Ops ops = Ops.create(g);\n"
                    + "  // Operations are typed classes with convenience\n"
                    + "  // builders in Ops.\n"
                    + "  Constant three = ops.constant(3);\n"
                    + "  // Single-result operations implement the Operand\n"
                    + "  // interface, so this works too.\n"
                    + "  Operand four = ops.constant(4);\n"
                    + "  // Most builders are found within a group, and accept\n"
                    + "  // Operand types as operands\n"
                    + "  Operand nine = ops.math.add(four, ops.constant(5));\n"
                    + "  // Multi-result operations however offer methods to\n"
                    + "  // select a particular result for use.\n"
                    + "  Operand result = \n"
                    + "      ops.math.add(ops.unique(s, a).y(), b);\n"
                    + "  // Optional attributes\n"
                    + "  ops.linalg.matMul(a, b, MatMul.transposeA(true));\n"
                    + "  // Naming operators\n"
                    + "  ops.withName(\"foo\").constant(5); // name \"foo\"\n"
                    + "  // Names can exist in a hierarchy\n"
                    + "  Ops sub = ops.withSubScope(\"sub\");\n"
                    + "  sub.withName(\"bar\").constant(4); // \"sub/bar\"\n"
                    + "}\n"
                    + "}</pre>\n",
                T_OP,
                T_OPERATOR)
            .addMethods(methods)
            .addMethod(ctorBuilder.build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withSubScope")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(T_STRING, "childScopeName")
            .returns(T_OPS)
            .addStatement("return new $T(scope.withSubScope(childScopeName))", T_OPS)
            .addJavadoc(
                "Returns an API that builds operations with the provided name prefix.\n"
                    + "\n@see {@link $T#withSubScope(String)}\n",
                T_SCOPE)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withName")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(T_STRING, "opName")
            .returns(T_OPS)
            .addStatement("return new Ops(scope.withName(opName))")
            .addJavadoc(
                "Returns an API that uses the provided name for an op.\n\n"
                    + "@see {@link $T#withName(String)}\n",
                T_SCOPE)
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("withControlDependencies")
            .addModifiers(Modifier.PUBLIC)
            .addParameter(T_ITERABLE_OPERAND, "controls")
            .returns(T_OPS)
            .addStatement("return new Ops(scope.withControlDependencies(controls))")
            .addJavadoc(
                "Returns an API that adds operations to the graph with the provided control"
                    + " dependencies.\n\n"
                    + "@see {@link $T#withControlDependencies(Iterable<Operand<?>>)}\n",
                T_SCOPE)
            .build());

    opsBuilder.addField(
        FieldSpec.builder(T_SCOPE, "scope").addModifiers(Modifier.PRIVATE, Modifier.FINAL).build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("scope")
            .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
            .returns(T_SCOPE)
            .addStatement("return scope")
            .addJavadoc("Returns the current {@link $T scope} of this API\n", T_SCOPE)
            .build());

    for (Map.Entry<String, ClassName> entry : groupToClass.entrySet()) {
      opsBuilder.addField(
          FieldSpec.builder(entry.getValue(), entry.getKey())
              .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
              .build());

      opsBuilder.addMethod(
          MethodSpec.methodBuilder(entry.getKey())
              .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
              .returns(entry.getValue())
              .addStatement("return $L", entry.getKey())
              .addJavadoc("Returns an API for building {@code $L} operations\n", entry.getKey())
              .build());
    }

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("create")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .addParameter(T_EXEC_ENV, "env")
            .returns(T_OPS)
            .addStatement("return new Ops(new $T(env))", T_SCOPE)
            .addJavadoc(
                "Creates an API for building operations in the provided execution environment\n")
            .build());

    opsBuilder.addMethod(
        MethodSpec.methodBuilder("create")
            .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
            .returns(T_OPS)
            .addStatement("return new Ops(new $T($T.getDefault()))", T_SCOPE, T_EAGER_SESSION)
            .addJavadoc(
                "Creates an API for building operations in the default eager execution"
                    + " environment\n\n"
                    + "<p>Invoking this method is equivalent to {@code"
                    + " Ops.create(EagerSession.getDefault())}.\n")
            .build());

    return opsBuilder.build();
  }

  private static AnnotationMirror getAnnotationMirror(Element element, TypeElement annotation) {
    for (AnnotationMirror am : element.getAnnotationMirrors()) {
      if (am.getAnnotationType().asElement().equals(annotation)) {
        return am;
      }
    }
    throw new IllegalArgumentException(
        "Annotation "
            + annotation.getSimpleName()
            + " not present on element "
            + element.getSimpleName());
  }

  private static String getAnnotationElementValueAsString(String elementName, AnnotationMirror am) {
    for (Map.Entry<? extends ExecutableElement, ? extends AnnotationValue> entry :
        am.getElementValues().entrySet()) {
      if (entry.getKey().getSimpleName().contentEquals(elementName)) {
        return entry.getValue().getValue().toString();
      }
    }
    return "";
  }
}
