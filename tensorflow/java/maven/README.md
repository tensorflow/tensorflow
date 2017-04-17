# TensorFlow for Java using Maven

The [TensorFlow Java
API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
is available through artifacts uploaded to [Maven
Central](https://oss.sonatype.org/content/repositories/snapshots/org/tensorflow/).
This document describes the process of updating the release artifacts. It does
_not_ describe how to use the artifacts, for which the reader is referred to the
[TensorFlow for Java installation instructions](https://www.tensorflow.org/code/tensorflow/java/README.md).

## Background

TensorFlow source (which is primarily in C++) is built using
[bazel](https://bazel.build) and not [maven](https://maven.apache.org/).  The
Java API wraps over this native code and thus depends on platform (OS,
architecture) specific native code.

Hence, the process for building and uploading release artifacts is not a single
`mvn deploy` command.

## Artifact Structure

There are four artifacts and thus `pom.xml`s involved in this release:

1.  `tensorflow`: The single dependency for projects requiring TensorFlow for
    Java. This convenience package depends on the two below, and is the one that
    should typically be used in other programs.

2.  `libtensorflow`: Java-only code for the [TensorFlow Java API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary).
    The `.jar` itself has no native code, but requires the native code be either
    already installed on the system or made available through
    `libtensorflow_jni`.

3.  `libtensorflow_jni`: The native libraries required by `libtensorflow`.
    Native code for all supported platforms is packaged into a single `.jar`.

4.  [`parentpom`](https://maven.apache.org/pom/index.html): Common settings
    shared between the above three.

## Updating the release

The TensorFlow artifacts at Maven Central are created from files built as part
of the TensorFlow release process (which uses `bazel`). The author's lack of
familiarity with Maven best practices combined with the use of a different build
system means that this process is possibly not ideal, but it's what we've got.
Suggestions are welcome.

In order to isolate the environment used for building, all release processes are
conducted in a [Docker](https://www.docker.com) container.

### Pre-requisites

-   `docker`
-   An account at [oss.sonatype.org](https://oss.sonatype.org/), that has
    permissions to update artifacts in the `org.tensorflow` group. If your
    account does not have permissions, then you'll need to ask someone who does
    to [file a ticket](https://issues.sonatype.org/) to add to the permissions
    ([sample ticket](https://issues.sonatype.org/browse/MVNCENTRAL-1637)).
-   A GPG signing key, required [to sign the release artifacts](http://central.sonatype.org/pages/apache-maven.html#gpg-signed-components).

### Deploying to Maven Central

1.  Create a file with your OSSRH credentials (or perhaps you use `mvn` and have
    it in `~/.m2/settings.xml`):

    ```sh
    SONATYPE_USERNAME="your_sonatype.org_username_here"
    SONATYPE_PASSWORD="your_sonatype.org_password_here"
    GPG_PASSPHRASE="your_gpg_passphrase_here"
    cat >/tmp/settings.xml <<EOF
    <settings>
      <servers>
        <server>
          <id>ossrh</id>
          <username>${SONATYPE_USERNAME}</username>
          <password>${SONATYPE_PASSWORD}</password>
        </server>
      </servers>
      <profiles>
        <profile>
          <id>ossrh</id>
          <activation>
            <activeByDefault>true</activeByDefault>
          </activation>
          <properties>
            <gpg.executable>gpg2</gpg.executable>
            <gpg.passphrase>${GPG_PASSPHRASE}</gpg.passphrase>
          </properties>
        </profile>
      </profiles>
    </settings>
    EOF
    ```

2.  Run the `release.sh` script.

3.  If the script above succeeds then the artifacts would have been uploaded to
    the private staging repository. After verifying the release, visit
    https://oss.sonatype.org/#stagingRepositories, find the `org.tensorflow`
    release and click on either `Release` to finalize the release, or `Drop` to
    abort. Some things of note:

    - For details, look at the [Sonatype guide](http://central.sonatype.org/pages/releasing-the-deployment.html).
    - Syncing with [Maven Central](http://repo1.maven.org/maven2/org/tensorflow/)
      can take 10 minutes to 2 hours (as per the [OSSRH
      guide](http://central.sonatype.org/pages/ossrh-guide.html#releasing-to-central)).

4.  Upon successful release, commit changes to all the `pom.xml` files
    (which should have the updated version number).

### Snapshots

If the `TF_VERSION` provided to the `release.sh` script ends in `-SNAPSHOT`,
then instead of using official release files, the nightly build artifacts from
https://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/ and
https://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/ will
be used to upload to the Maven Central snapshots repository.


## References

-   [Sonatype guide](http://central.sonatype.org/pages/ossrh-guide.html) for
    hosting releases.
-   [Ticket that created the `org/tensorflow` configuration](https://issues.sonatype.org/browse/OSSRH-28072) on OSSRH.
