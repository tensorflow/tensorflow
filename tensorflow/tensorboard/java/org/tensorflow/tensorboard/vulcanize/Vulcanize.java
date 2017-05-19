// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.tensorflow.tensorboard.vulcanize;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Verify.verifyNotNull;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.CharMatcher;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.javascript.jscomp.BasicErrorManager;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.CompilerOptions.LanguageMode;
import com.google.javascript.jscomp.CompilerOptions.Reach;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.PropertyRenamingPolicy;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.VariableRenamingPolicy;
import com.google.protobuf.TextFormat;
import io.bazel.rules.closure.Webpath;
import io.bazel.rules.closure.webfiles.BuildInfo.Webfiles;
import io.bazel.rules.closure.webfiles.BuildInfo.WebfilesSource;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Html5Printer;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.Parser;
import org.jsoup.parser.Tag;

/** Simple one-off solution for TensorBoard vulcanization. */
public final class Vulcanize {

  private static final Parser parser = Parser.htmlParser();
  private static final Map<Webpath, Path> webfiles = new HashMap<>();
  private static final Set<Webpath> alreadyInlined = new HashSet<>();
  private static final Set<String> legalese = new HashSet<>();
  private static final List<String> licenses = new ArrayList<>();
  private static final List<Webpath> stack = new ArrayList<>();
  private static Webpath outputPath;
  private static Node licenseComment;
  private static boolean nominify;

  public static void main(String[] args) throws IOException {
    Webpath inputPath = Webpath.get(args[0]);
    outputPath = Webpath.get(args[1]);
    Path output = Paths.get(args[2]);
    for (int i = 3; i < args.length; i++) {
      Webfiles manifest = loadWebfilesPbtxt(Paths.get(args[i]));
      for (WebfilesSource src : manifest.getSrcList()) {
        webfiles.put(Webpath.get(src.getWebpath()), Paths.get(src.getPath()));
      }
    }
    stack.add(inputPath);
    Document document = parse(Files.readAllBytes(webfiles.get(inputPath)));
    transform(document);
    if (licenseComment != null) {
      licenseComment.attr("comment", String.format("\n%s\n", Joiner.on("\n\n").join(licenses)));
    }
    Files.write(
        output,
        Html5Printer.stringify(document).getBytes(UTF_8),
        StandardOpenOption.WRITE,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING);
  }

  private static void transform(Node root) throws IOException {
    Node node = checkNotNull(root);
    Node newNode;
    while (true) {
      newNode = enterNode(node);
      if (node.equals(root)) {
        root = newNode;
      }
      node = newNode;
      if (node.childNodeSize() > 0) {
        node = node.childNode(0);
      } else {
        while (true) {
          newNode = leaveNode(node);
          if (node.equals(root)) {
            root = newNode;
          }
          node = newNode;
          if (node.equals(root)) {
            return;
          }
          Node next = node.nextSibling();
          if (next == null) {
            if (node.parentNode() == null) {
              return;
            }
            node = verifyNotNull(node.parentNode(), "unexpected root: %s", node);
          } else {
            node = next;
            break;
          }
        }
      }
    }
  }

  private static Node enterNode(Node node) throws IOException {
    Node newNode = node;
    if (node instanceof Element) {
      if (node.nodeName().equals("link") && node.attr("rel").equals("import")) {
        // Inline HTML.
        Webpath href = me().lookup(Webpath.get(node.attr("href")));
        if (alreadyInlined.add(href)) {
          newNode =
              parse(Files.readAllBytes(checkNotNull(webfiles.get(href), "%s in %s", href, me())));
          stack.add(href);
          node.replaceWith(newNode);
        } else {
          newNode = new TextNode("", node.baseUri());
          node.replaceWith(newNode);
        }
      } else if (node.nodeName().equals("script")) {
        nominify = node.hasAttr("nominify");
        node.removeAttr("nominify");
        Webpath src;
        String script;
        if (node.attr("src").isEmpty()) {
          // Minify JavaScript.
          StringBuilder sb = new StringBuilder();
          for (Node child : node.childNodes()) {
            if (child instanceof DataNode) {
              sb.append(((DataNode) child).getWholeData());
            }
          }
          src = me();
          script = sb.toString();
        } else {
          // Inline JavaScript.
          src = me().lookup(Webpath.get(node.attr("src")));
          Path other = webfiles.get(src);
          if (other != null) {
            script = new String(Files.readAllBytes(other), UTF_8);
            node.removeAttr("src");
          } else {
            src = me();
            script = "";
          }
        }
        script = minify(src, script);
        newNode =
            new Element(Tag.valueOf("script"), node.baseUri(), node.attributes())
                .appendChild(new DataNode(script, node.baseUri()));
        node.replaceWith(newNode);
      } else if (node.nodeName().equals("link")
          && node.attr("rel").equals("stylesheet")
          && !node.attr("href").isEmpty()) {
        // Inline CSS.
        Webpath href = me().lookup(Webpath.get(node.attr("href")));
        Path other = webfiles.get(href);
        if (other != null) {
          newNode =
              new Element(Tag.valueOf("style"), node.baseUri(), node.attributes())
                  .appendChild(
                      new DataNode(new String(Files.readAllBytes(other), UTF_8), node.baseUri()));
          newNode.removeAttr("rel");
          newNode.removeAttr("href");
          node.replaceWith(newNode);
        }
      }
      rootifyAttribute(newNode, "href");
      rootifyAttribute(newNode, "src");
      rootifyAttribute(newNode, "action");
      rootifyAttribute(newNode, "assetpath");
    } else if (node instanceof Comment) {
      String text = ((Comment) node).getData();
      if (text.contains("@license")) {
        handleLicense(text);
        if (licenseComment == null) {
          licenseComment = node;
        } else {
          newNode = new TextNode("", node.baseUri());
          node.replaceWith(newNode);
        }
      } else {
        newNode = new TextNode("", node.baseUri());
        node.replaceWith(newNode);
      }
    }
    return newNode;
  }

  private static String minify(Webpath src, String script) {
    if (nominify) {
      return script;
    }
    Compiler compiler = new Compiler(new JsPrintlessErrorManager());
    CompilerOptions options = new CompilerOptions();
    options.skipAllCompilerPasses(); // too lazy to get externs
    options.setLanguageIn(LanguageMode.ECMASCRIPT_2016);
    options.setLanguageOut(LanguageMode.ECMASCRIPT5);
    options.setContinueAfterErrors(true);
    options.setManageClosureDependencies(false);
    options.setRenamingPolicy(VariableRenamingPolicy.LOCAL, PropertyRenamingPolicy.OFF);
    options.setShadowVariables(true);
    options.setInlineVariables(Reach.LOCAL_ONLY);
    options.setFlowSensitiveInlineVariables(true);
    options.setInlineFunctions(Reach.LOCAL_ONLY);
    options.setAssumeClosuresOnlyCaptureReferences(false);
    options.setCheckGlobalThisLevel(CheckLevel.OFF);
    options.setFoldConstants(true);
    options.setCoalesceVariableNames(true);
    options.setDeadAssignmentElimination(true);
    options.setCollapseVariableDeclarations(true);
    options.setConvertToDottedProperties(true);
    options.setLabelRenaming(true);
    options.setRemoveDeadCode(true);
    options.setOptimizeArgumentsArray(true);
    options.setRemoveUnusedVariables(Reach.LOCAL_ONLY);
    options.setCollapseObjectLiterals(true);
    options.setProtectHiddenSideEffects(true);
    //options.setPrettyPrint(true);
    compiler.disableThreads();
    compiler.compile(
        ImmutableList.<SourceFile>of(),
        ImmutableList.of(SourceFile.fromCode(src.toString(), script)),
        options);
    return compiler.toSource();
  }

  private static void handleLicense(String text) {
    if (legalese.add(CharMatcher.whitespace().removeFrom(text))) {
      licenses.add(CharMatcher.anyOf("\r\n").trimFrom(text));
    }
  }

  private static Node leaveNode(Node node) {
    if (node instanceof Document) {
      stack.remove(stack.size() - 1);
    }
    return node;
  }

  private static Webpath me() {
    return Iterables.getLast(stack);
  }

  private static void rootifyAttribute(Node node, String attribute) {
    String value = node.attr(attribute);
    if (value.isEmpty()) {
      return;
    }
    Webpath uri = Webpath.get(value);
    if (webfiles.containsKey(uri)) {
      node.attr(attribute, outputPath.getParent().relativize(uri).toString());
    }
  }

  private static Document parse(byte[] bytes) {
    return parse(new ByteArrayInputStream(bytes));
  }

  private static Document parse(InputStream input) {
    Document document;
    try {
      document = Jsoup.parse(input, null, "", parser);
    } catch (IOException e) {
      throw new AssertionError("I/O error when parsing byte array D:", e);
    }
    document.outputSettings().indentAmount(0);
    document.outputSettings().prettyPrint(false);
    return document;
  }

  private static Webfiles loadWebfilesPbtxt(Path path) throws IOException {
    Webfiles.Builder build = Webfiles.newBuilder();
    TextFormat.getParser().merge(new String(Files.readAllBytes(path), UTF_8), build);
    return build.build();
  }

  private static final class JsPrintlessErrorManager extends BasicErrorManager {

    @Override
    public void println(CheckLevel level, JSError error) {}

    @Override
    public void printSummary() {}
  }
}
