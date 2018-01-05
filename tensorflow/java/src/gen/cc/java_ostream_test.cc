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

#include <vector>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/java_ostream.h"

namespace tensorflow {
namespace java {
namespace {

TEST(StreamTest, BlocksAndLines) {
  SourceBufferWriter src_writer;
  JavaOutputStream src_ostream(&src_writer);

  src_ostream << "int i = 0;" << endl
      << "int j = 10;" << endl
      << "if (true)"
      << beginb
      << "int aLongWayToTen = 0;" << endl
      << "while (++i <= j)"
      << beginb
      << "++aLongWayToTen;" << endl
      << endb
      << endb;

  const char* expected =
      "int i = 0;\n"
      "int j = 10;\n"
      "if (true) {\n"
      "  int aLongWayToTen = 0;\n"
      "  while (++i <= j) {\n"
      "    ++aLongWayToTen;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(StreamTest, Types) {
  SourceBufferWriter src_writer;
  JavaOutputStream src_ostream(&src_writer);

  Type generic = Type::Generic("T").add_supertype(Type::Class("Number"));
  src_ostream << Type::Int() << ", "
      << Type::Class("String") << ", "
      << generic << ", "
      << Type::ListOf(generic) << ", "
      << Type::ListOf(Type::IterableOf(generic)) << ", "
      << Type::ListOf(Type::Generic());

  const char* expected =
      "int, String, T, List<T>, List<Iterable<T>>, List<?>";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(StreamTest, Snippets) {
  SourceBufferWriter src_writer;
  JavaOutputStream src_ostream(&src_writer);

  Snippet snippet =
      Snippet::Create(io::JoinPath(kGenResourcePath, "test.snippet.java"));
  src_ostream << snippet
      << beginb
      << snippet
      << endb;

  const char* expected =
      "// Here is a little snippet\n"
      "System.out.println(\"Hello!\");\n"
      "{\n"
      "  // Here is a little snippet\n"
      "  System.out.println(\"Hello!\");\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, SimpleClass) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  src_ostream.BeginClass(clazz, nullptr, PUBLIC)->EndClass();

  const char* expected = "public class Test {\n}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, SimpleClassWithPackageAndImports) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test", "org.test");
  std::vector<Type> deps;
  deps.push_back(Type::Class("TypeA", "org.test.sub"));
  deps.push_back(Type::Class("TypeA", "org.test.sub"));  // a second time
  deps.push_back(Type::Class("TypeB", "org.other"));
  deps.push_back(Type::Class("SamePackageType", "org.test"));
  deps.push_back(Type::Class("NoPackageType"));
  src_ostream.BeginClass(clazz, &deps, PUBLIC)->EndClass();

  const char* expected =
      "package org.test;\n\n"
      "import org.other.TypeB;\n"
      "import org.test.sub.TypeA;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}


TEST(WriteClass, AnnotatedAndDocumentedClass) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  clazz.description("This class has a\n<p>\nmultiline description.");
  clazz.add_annotation(Annotation::Create("Bean"));
  clazz.add_annotation(Annotation::Create("SuppressWarnings")
      .attributes("\"rawtypes\""));
  src_ostream.BeginClass(clazz, nullptr, PUBLIC)->EndClass();

  const char* expected =
      "/**\n"
      " * This class has a\n"
      " * <p>\n"
      " * multiline description.\n"
      " **/\n"
      "@Bean\n"
      "@SuppressWarnings(\"rawtypes\")\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, ParameterizedClass) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  clazz.add_parameter(Type::Generic("T"));
  clazz.add_parameter(Type::Generic("U").add_supertype(Type::Class("Number")));
  src_ostream.BeginClass(clazz, nullptr, PUBLIC)->EndClass();

  const char* expected = "public class Test<T, U extends Number> {\n}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, ParameterizedClassAndSupertypes) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  Type type_t = Type::Generic("T");
  clazz.add_parameter(type_t);
  Type type_u = Type::Generic("U").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_u);
  clazz.add_supertype(Type::Interface("Parametrizable").add_parameter(type_u));
  clazz.add_supertype(Type::Interface("Runnable"));
  clazz.add_supertype(Type::Class("SuperTest").add_parameter(type_t));
  src_ostream.BeginClass(clazz, nullptr, PUBLIC)->EndClass();

  const char* expected =
      "public class Test<T, U extends Number>"
      " extends SuperTest<T> implements Parametrizable<U>, Runnable {\n}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, ParameterizedClassFields) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  std::vector<Variable> static_fields;
  static_fields.push_back(Variable::Create("field1", Type::Class("String")));
  std::vector<Variable> member_fields;
  member_fields.push_back(Variable::Create("field2", Type::Class("String")));
  member_fields.push_back(Variable::Create("field3", type_t));

  class_ostream->WriteFields(static_fields, STATIC | PUBLIC | FINAL)
      ->WriteFields(member_fields, PRIVATE)
      ->EndClass();

  const char* expected =
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static final String field1;\n"
      "  \n"
      "  private String field2;\n"
      "  private T field3;\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, SimpleInnerClass) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Type inner_class = Type::Class("InnerTest");
  class_ostream->BeginInnerClass(inner_class, PUBLIC)->EndClass();
  class_ostream->EndClass();

  const char* expected =
      "public class Test {\n"
      "  \n"
      "  public class InnerTest {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteClass, StaticParameterizedInnerClass) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Type inner_class = Type::Class("InnerTest");
  inner_class.add_parameter(type_t);
  class_ostream->BeginInnerClass(inner_class, PUBLIC | STATIC)->EndClass();
  class_ostream->EndClass();

  const char* expected =
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static class InnerTest<T extends Number> {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteMethod, SimpleMethod) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Method method = Method::Create("doNothing", Type::Void());
  class_ostream->BeginMethod(method, PUBLIC)->EndMethod();
  class_ostream->EndClass();

  const char* expected =
      "public class Test {\n"
      "  \n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteMethod, AnnotatedAndDocumentedMethod) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Method method = Method::Create("doNothing", Type::Void());
  method.description("This method has a\n<p>\nmultiline description.");
  method.add_annotation(Annotation::Create("Override"));
  method.add_annotation(Annotation::Create("SuppressWarnings")
      .attributes("\"rawtypes\""));
  class_ostream->BeginMethod(method, PUBLIC)->EndMethod();
  class_ostream->EndClass();

  const char* expected =
      "public class Test {\n"
      "  \n"
      "  /**\n"
      "   * This method has a\n"
      "   * <p>\n"
      "   * multiline description.\n"
      "   **/\n"
      "  @Override\n"
      "  @SuppressWarnings(\"rawtypes\")\n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteMethod, DocumentedMethodWithArguments) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Method method = Method::Create("boolToInt", Type::Int());
  method.description("Converts a boolean to an int");
  method.return_description("int value for this boolean");
  method.add_argument(Variable::Create("b", Type::Boolean()));
  Variable reverse = Variable::Create("reverse", Type::Boolean());
  reverse.description("if true, value is reversed");
  method.add_argument(reverse);
  MethodOutputStream* method_ostream =
      class_ostream->BeginMethod(method, PUBLIC);
  *method_ostream << "if (b && !reverse)"
      << beginb
      << "return 1;" << endl
      << endb
      << "return 0;" << endl;
  method_ostream->EndMethod();
  class_ostream->EndClass();

  const char* expected =
      "public class Test {\n"
      "  \n"
      "  /**\n"
      "   * Converts a boolean to an int\n"
      "   * \n"
      "   * @param b\n"
      "   * @param reverse if true, value is reversed\n"
      "   * @return int value for this boolean\n"
      "   **/\n"
      "  public int boolToInt(boolean b, boolean reverse) {\n"
      "    if (b && !reverse) {\n"
      "      return 1;\n"
      "    }\n"
      "    return 0;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteMethod, ParameterizedMethod) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Method method = Method::Create("doNothing", type_t);
  MethodOutputStream* method_ostream =
      class_ostream->BeginMethod(method, PUBLIC);
  *method_ostream << "return null;" << endl;
  method_ostream->EndMethod();
  class_ostream->EndClass();

  const char* expected =
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

TEST(WriteMethod, StaticParameterizedMethod) {
  SourceBufferWriter src_writer;
  SourceOutputStream src_ostream(&src_writer);

  Type clazz = Type::Class("Test");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  ClassOutputStream* class_ostream
      = src_ostream.BeginClass(clazz, nullptr, PUBLIC);

  Method method = Method::Create("doNothing", type_t);
  MethodOutputStream* method_ostream =
      class_ostream->BeginMethod(method, PUBLIC | STATIC);
  *method_ostream << "return null;" << endl;
  method_ostream->EndMethod();
  class_ostream->EndClass();

  const char* expected =
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static <T extends Number> T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, src_writer.str().data());
}

}  // namespace
}  // namespace java
}  // namespace tensorflow
