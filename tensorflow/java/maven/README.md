# TensorFlow for Java using Maven

The
[TensorFlow Java API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
is available on Maven Central and JCenter through artifacts uploaded to
[OSS Sonatype](https://oss.sonatype.org/content/repositories/releases/org/tensorflow/) and
[Bintray](https://bintray.com/google/tensorflow/tensorflow) respectively. This
document describes the process of updating the release artifacts. It does _not_
describe how to use the artifacts, for which the reader is referred to
the
[TensorFlow for Java installation instructions](https://www.tensorflow.org/code/tensorflow/java/README.md).

## Background

TensorFlow source (which is primarily in C++) is built using
[bazel](https://bazel.build) and not [maven](https://maven.apache.org/).  The
Java API wraps over this native code and thus depends on platform (OS,
architecture) specific native code.

Hence, the process for building and uploading release artifacts is not a single
`mvn deploy` command.

## Artifact Structure

There are seven artifacts and thus `pom.xml`s involved in this release:

1.  `tensorflow`: The single dependency for projects requiring TensorFlow for
    Java. This convenience package depends on `libtensorflow` and
    `libtensorflow_jni`. Typically, this is the single dependency that should
    be used by client programs (unless GPU support is required).

2.  `libtensorflow`: Java-only code for the [TensorFlow Java API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary).
    The `.jar` itself has no native code, but requires the native code be either
    already installed on the system or made available through
    `libtensorflow_jni`.

3.  `libtensorflow_jni`: The native libraries required by `libtensorflow`.
    Native code for all supported platforms is packaged into a single `.jar`.

4.  `libtensorflow_jni_gpu`: The native libraries required by `libtensorflow`
    with GPU (CUDA) support enabled. Programs requiring GPU-enabled TensorFlow
    should add a dependency on `libtensorflow` and `libtensorflow_jni_gpu`.
    As of January 2018, this artifact is *Linux only*.

5.  `proto`: Generated Java code for TensorFlow protocol buffers
    (e.g., `MetaGraphDef`, `ConfigProto` etc.)

6. `tensorflow-android`: A package geared towards
    supporting [TensorFlow on Android](../../contrib/android/README.md), and is
    a self-contained Android AAR library containing all necessary native and
    Java code.

7.  [`parentpom`](https://maven.apache.org/pom/index.html): Common settings
    shared by all of the above.

8. `hadoop`: The TensorFlow TFRecord InputFormat/OutputFormat for Apache Hadoop.
    The source code for this package is available in the [TensorFlow Ecosystem](https://github.com/tensorflow/ecosystem/tree/master/hadoop)

9. `spark-connector`: A Scala library for loading and storing TensorFlow TFRecord
    using Apache Spark DataFrames. The source code for this package is available
    in the [TensorFlow Ecosystem](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector)

## Updating the release

The Maven artifacts are created from files built as part of the TensorFlow
release process (which uses `bazel`). The author's lack of familiarity with
Maven best practices combined with the use of a different build system means
that this process is possibly not ideal, but it's what we've got.  Suggestions
are welcome.

In order to isolate the environment used for building, all release processes are
conducted in a [Docker](https://www.docker.com) container.

### Pre-requisites

-   `docker`
-   An account at [oss.sonatype.org](https://oss.sonatype.org/), that has
    permissions to update artifacts in the `org.tensorflow` group. If your
    account does not have permissions, then you'll need to ask someone who does
    to [file a ticket](https://issues.sonatype.org/) to add to the permissions
    ([sample ticket](https://issues.sonatype.org/browse/MVNCENTRAL-1637)).
-   An account at [bintray.com](https://bintray.com) that has permissions to
    update the [tensorflow repository](https://bintray.com/google/tensorflow).
    If your account does not have permissions, then you'll need to ask one of
    the [organization administrators](https://bintray.com/google) to give you
    permissions to update the `tensorflow` repository. Please keep the
    [repository option](https://bintray.com/google/tensorflow/edit?tab=general)
    to *"GPG sign uploaded files using Bintray's public/private key pair"*
    **unchecked**, otherwise it will conflict with locally signed artifacts.
-   A GPG signing key, required
    [to sign the release artifacts](http://central.sonatype.org/pages/apache-maven.html#gpg-signed-components).

### Deploying to Sonatype and Bintray

1.  Create a file with your OSSRH credentials and
    [Bintray API key](https://bintray.com/docs/usermanual/interacting/interacting_interacting.html#anchorAPIKEY)
    (or perhaps you use `mvn` and have it in `~/.m2/settings.xml`):

    ```sh
    SONATYPE_USERNAME="your_sonatype.org_username_here"
    SONATYPE_PASSWORD="your_sonatype.org_password_here"
    BINTRAY_USERNAME="your_bintray_username_here"
    BINTRAY_API_KEY="your_bintray_api_key_here"
    GPG_PASSPHRASE="your_gpg_passphrase_here"
    cat >/tmp/settings.xml <<EOF
    <settings>
      <servers>
        <server>
          <id>ossrh</id>
          <username>${SONATYPE_USERNAME}</username>
          <password>${SONATYPE_PASSWORD}</password>
        </server>
        <server>
          <id>bintray</id>
          <username>${BINTRAY_USERNAME}</username>
          <password>${BINTRAY_API_KEY}</password>
        </server>
      </servers>
      <properties>
        <gpg.executable>gpg2</gpg.executable>
        <gpg.passphrase>${GPG_PASSPHRASE}</gpg.passphrase>
      </properties>
    </settings>
    EOF
    ```

2.  Run the `release.sh` script.

3.  If the script above succeeds then the artifacts would have been uploaded to
    the private staging repository in Sonatype, and as unpublished artifacts in
    Bintray. After verifying the release, you should finalize or abort the
    release on both sites.

4.  Visit https://oss.sonatype.org/#stagingRepositories, find the `org.tensorflow`
    release and click on either `Release` to finalize the release, or `Drop` to
    abort.

5.  Visit https://bintray.com/google/tensorflow/tensorflow, and select the
    version you just uploaded. Notice there's a message about unpublished
    artifacts. Click on either `Publish` to finalize the release, or `Discard`
    to abort.

6.  Some things of note:
    - For details, look at the [Sonatype guide](http://central.sonatype.org/pages/releasing-the-deployment.html).
    - Syncing with [Maven Central](http://repo1.maven.org/maven2/org/tensorflow/)
      can take 10 minutes to 2 hours (as per the [OSSRH
      guide](http://central.sonatype.org/pages/ossrh-guide.html#releasing-to-central)).
    - For Bintray details, refer to their guide on
      [managing uploaded content](https://bintray.com/docs/usermanual/uploads/uploads_managinguploadedcontent.html#_publishing).

7.  Upon successful release, commit changes to all the `pom.xml` files
    (which should have the updated version number).

### Skip deploying to a repository

Should you need, setting environment variables `DEPLOY_OSSRH=0` or
`DEPLOY_BINTRAY=0` when calling `release.sh` will skip deploying to OSSRH or
Bintray respectively. Note that snapshots are only uploaded to OSSRH, so you
cannot skip deploying to OSSRH for a `-SNAPSHOT` version.

## The overall flow

This section provides some pointers around how artifacts are currently
assembled.

All native and java code is first built and tested by the release process
which run various scripts under the [`tools/ci_build`](../../tools/ci_build/)
directory. Of particular interest may be
`tools/ci_build/builds/libtensorflow.sh` which bundles Java-related build
sources and outputs into archives, and `tools/ci_build/builds/android_full.sh`
which produces an Android AAR package.

Maven artifacts however are not created in Jenkins. Instead, artifacts are
created and deployed externally on-demand, when a maintainer runs the
`release.sh` script.

This script spins up a Docker instance which downloads the archives created by
successful runs of various `tools/ci_build` scripts on the Tensorflow Jenkins
server.

It organizes these archives locally into a maven-friendly layout, and runs `mvn
deploy` to create maven artifacts within the container. Native libraries built
in Jenkins are used as-is, but srcjars for java code are used to compile class
files and generate javadocs.) It also downloads the Android AAR from the Jenkins
server and directly deploys it via `mvn gpg:sign-and-deploy-file`.

`release.sh` then stages these artifacts to OSSRH and Bintray, and if all goes
well a maintainer can log into both sites to promote them as a new release.

There is a small change to the flow for a standard (rather than a `-SNAPSHOT`)
release. Rather than downloading archives directly from jobs on the Jenkins
server, the script uses a static repository of QA-blessed archives.

## References

-   [Sonatype guide](http://central.sonatype.org/pages/ossrh-guide.html) for
    hosting releases.
-   [Ticket that created the `org/tensorflow` configuration](https://issues.sonatype.org/browse/OSSRH-28072) on OSSRH.
-   The [Bintray User Manual](https://bintray.com/docs/usermanual/index.html)
