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

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Filer;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.TypeElement;
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
    return SourceVersion.latestSupported();
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    messager = processingEnv.getMessager();
    filer = processingEnv.getFiler();
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
      return true;
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
      return true;
    }

    // Collect all classes tagged with our annotation.
    Set<TypeElement> opClasses = new HashSet<TypeElement>();
    if (!collectOpClasses(roundEnv, opClasses, annotation)) {
      return true;
    }

    // Nothing to do when there are no tagged classes.
    if (opClasses.isEmpty()) {
      return true;
    }

    // TODO:(kbsriram) validate operator classes and generate Op API.
    writeApi();
    hasRun = true;
    return true;
  }

  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return Collections.singleton(String.format("%s.annotation.Operator", OP_PACKAGE));
  }

  private void writeApi() {
    // Generate an empty class for now and get the build working correctly. This will be changed to
    // generate the actual API once we've done with build-related changes.
    // TODO:(kbsriram)
    try (PrintWriter writer =
        new PrintWriter(filer.createSourceFile(String.format("%s.Ops", OP_PACKAGE)).openWriter())) {
      writer.println(String.format("package %s;", OP_PACKAGE));
      writer.println("public class Ops{}");
    } catch (IOException e) {
      error(null, "Unexpected failure generating API: %s", e.getMessage());
    }
  }

  private boolean collectOpClasses(
      RoundEnvironment roundEnv, Set<TypeElement> opClasses, TypeElement annotation) {
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
      opClasses.add((TypeElement) e);
    }
    return result;
  }

  private void error(Element e, String message, Object... args) {
    if (args != null && args.length > 0) {
      message = String.format(message, args);
    }
    messager.printMessage(Kind.ERROR, message, e);
  }

  private Filer filer;
  private Messager messager;
  private boolean hasRun = false;
  private static final String OP_PACKAGE = "org.tensorflow.op";
}
