// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/internal/version"
)

var (
	regenerate   = flag.Bool("regenerate", false, "regenerate files")
	buildRelease = flag.Bool("buildRelease", false, "build release binaries")

	protobufVersion = "26.0-rc2"

	golangVersions = func() []string {
		// Version policy: same version as is in the x/ repos' go.mod.
		return []string{
			"1.17.13",
			"1.18.10",
			"1.19.13",
			"1.20.12",
			"1.21.5",
		}
	}()
	golangLatest = golangVersions[len(golangVersions)-1]

	staticcheckVersion = "2023.1.6"
	staticcheckSHA256s = map[string]string{
		"darwin/amd64": "b14a0cbd3c238713f5f9db41550893ea7d75d8d7822491c7f4e33e2fe43f6305",
		"darwin/arm64": "f1c869abe6be2c6ab727dc9d6049766c947534766d71a1798c12a37526ea2b6f",
		"linux/386":    "02859a7c44c7b5ab41a70d9b8107c01ab8d2c94075bae3d0b02157aff743ca42",
		"linux/amd64":  "45337834da5dc7b8eff01cb6b3837e3759503cfbb8edf36b09e42f32bccb1f6e",
	}

	// purgeTimeout determines the maximum age of unused sub-directories.
	purgeTimeout = 30 * 24 * time.Hour // 1 month

	// Variables initialized by mustInitDeps.
	modulePath   string
	protobufPath string
)

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	if os.Getenv("GO_BUILDER_NAME") != "" {
		// To start off, run on longtest builders, not longtest-race ones.
		if race() {
			t.Skip("skipping integration test in race mode on builders")
		}
		// When on a builder, run even if it's not explicitly requested
		// provided our caller isn't already running it.
		if os.Getenv("GO_PROTOBUF_INTEGRATION_TEST_RUNNING") == "1" {
			t.Skip("protobuf integration test is already running, skipping nested invocation")
		}
		os.Setenv("GO_PROTOBUF_INTEGRATION_TEST_RUNNING", "1")
	} else if flag.Lookup("test.run").Value.String() != "^TestIntegration$" {
		t.Skip("not running integration test if not explicitly requested via test.bash")
	}

	mustInitDeps(t)
	mustHandleFlags(t)

	// Report dirt in the working tree quickly, rather than after
	// going through all the presubmits.
	//
	// Fail the test late, so we can test uncommitted changes with -failfast.
	gitDiff := mustRunCommand(t, "git", "diff", "HEAD")
	if strings.TrimSpace(gitDiff) != "" {
		fmt.Printf("WARNING: working tree contains uncommitted changes:\n%v\n", gitDiff)
	}
	gitUntracked := mustRunCommand(t, "git", "ls-files", "--others", "--exclude-standard")
	if strings.TrimSpace(gitUntracked) != "" {
		fmt.Printf("WARNING: working tree contains untracked files:\n%v\n", gitUntracked)
	}

	// Do the relatively fast checks up-front.
	t.Run("GeneratedGoFiles", func(t *testing.T) {
		diff := mustRunCommand(t, "go", "run", "-tags", "protolegacy", "./internal/cmd/generate-types")
		if strings.TrimSpace(diff) != "" {
			t.Fatalf("stale generated files:\n%v", diff)
		}
		diff = mustRunCommand(t, "go", "run", "-tags", "protolegacy", "./internal/cmd/generate-protos")
		if strings.TrimSpace(diff) != "" {
			t.Fatalf("stale generated files:\n%v", diff)
		}
	})
	t.Run("FormattedGoFiles", func(t *testing.T) {
		files := strings.Split(strings.TrimSpace(mustRunCommand(t, "git", "ls-files", "*.go")), "\n")
		diff := mustRunCommand(t, append([]string{"gofmt", "-d"}, files...)...)
		if strings.TrimSpace(diff) != "" {
			t.Fatalf("unformatted source files:\n%v", diff)
		}
	})
	t.Run("CopyrightHeaders", func(t *testing.T) {
		files := strings.Split(strings.TrimSpace(mustRunCommand(t, "git", "ls-files", "*.go", "*.proto")), "\n")
		mustHaveCopyrightHeader(t, files)
	})

	var wg sync.WaitGroup
	sema := make(chan bool, (runtime.NumCPU()+1)/2)
	for i := range golangVersions {
		goVersion := golangVersions[i]
		goLabel := "Go" + goVersion
		runGo := func(label string, cmd command, args ...string) {
			wg.Add(1)
			sema <- true
			go func() {
				defer wg.Done()
				defer func() { <-sema }()
				t.Run(goLabel+"/"+label, func(t *testing.T) {
					args[0] += goVersion
					cmd.mustRun(t, args...)
				})
			}()
		}

		runGo("Normal", command{}, "go", "test", "-race", "./...")
		runGo("PureGo", command{}, "go", "test", "-race", "-tags", "purego", "./...")
		runGo("Reflect", command{}, "go", "test", "-race", "-tags", "protoreflect", "./...")
		if goVersion == golangLatest {
			runGo("ProtoLegacy", command{}, "go", "test", "-race", "-tags", "protolegacy", "./...")
			runGo("ProtocGenGo", command{Dir: "cmd/protoc-gen-go/testdata"}, "go", "test")
			runGo("Conformance", command{Dir: "internal/conformance"}, "go", "test", "-execute")

			// Only run the 32-bit compatibility tests for Linux;
			// avoid Darwin since 10.15 dropped support i386 code execution.
			if runtime.GOOS == "linux" {
				runGo("Arch32Bit", command{Env: append(os.Environ(), "GOARCH=386")}, "go", "test", "./...")
			}
		}
	}
	wg.Wait()

	t.Run("GoStaticCheck", func(t *testing.T) {
		checks := []string{
			"all",     // start with all checks enabled
			"-SA1019", // disable deprecated usage check
			"-S*",     // disable code simplification checks
			"-ST*",    // disable coding style checks
			"-U*",     // disable unused declaration checks
		}
		out := mustRunCommand(t, "staticcheck", "-checks="+strings.Join(checks, ","), "-fail=none", "./...")

		// Filter out findings from certain paths.
		var findings []string
		for _, finding := range strings.Split(strings.TrimSpace(out), "\n") {
			switch {
			case strings.HasPrefix(finding, "internal/testprotos/legacy/"):
			default:
				findings = append(findings, finding)
			}
		}
		if len(findings) > 0 {
			t.Fatalf("staticcheck findings:\n%v", strings.Join(findings, "\n"))
		}
	})
	t.Run("CommittedGitChanges", func(t *testing.T) {
		if strings.TrimSpace(gitDiff) != "" {
			t.Fatalf("uncommitted changes")
		}
	})
	t.Run("TrackedGitFiles", func(t *testing.T) {
		if strings.TrimSpace(gitUntracked) != "" {
			t.Fatalf("untracked files")
		}
	})
}

func mustInitDeps(t *testing.T) {
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}

	// Determine the directory to place the test directory.
	repoRoot, err := os.Getwd()
	check(err)
	testDir := filepath.Join(repoRoot, ".cache")
	check(os.MkdirAll(testDir, 0775))

	// Delete the current directory if non-empty,
	// which only occurs if a dependency failed to initialize properly.
	var workingDir string
	finishedDirs := map[string]bool{}
	defer func() {
		if workingDir != "" {
			os.RemoveAll(workingDir) // best-effort
		}
	}()
	startWork := func(name string) string {
		workingDir = filepath.Join(testDir, name)
		return workingDir
	}
	finishWork := func() {
		finishedDirs[workingDir] = true
		workingDir = ""
	}

	// Delete other sub-directories that are no longer relevant.
	defer func() {
		now := time.Now()
		fis, _ := ioutil.ReadDir(testDir)
		for _, fi := range fis {
			dir := filepath.Join(testDir, fi.Name())
			if finishedDirs[dir] {
				os.Chtimes(dir, now, now) // best-effort
				continue
			}
			if now.Sub(fi.ModTime()) < purgeTimeout {
				continue
			}
			fmt.Printf("delete %v\n", fi.Name())
			os.RemoveAll(dir) // best-effort
		}
	}()

	// The bin directory contains symlinks to each tool by version.
	// It is safe to delete this directory and run the test script from scratch.
	binPath := startWork("bin")
	check(os.RemoveAll(binPath))
	check(os.Mkdir(binPath, 0775))
	check(os.Setenv("PATH", binPath+":"+os.Getenv("PATH")))
	registerBinary := func(name, path string) {
		check(os.Symlink(path, filepath.Join(binPath, name)))
	}
	finishWork()

	// Get the protobuf toolchain.
	protobufPath = startWork("protobuf-" + protobufVersion)
	if _, err := os.Stat(protobufPath); err != nil {
		fmt.Printf("download %v\n", filepath.Base(protobufPath))
		checkoutVersion := protobufVersion
		if isCommit := strings.Trim(protobufVersion, "0123456789abcdef") == ""; !isCommit {
			// release tags have "v" prefix
			checkoutVersion = "v" + protobufVersion
		}
		command{Dir: testDir}.mustRun(t, "git", "clone", "https://github.com/protocolbuffers/protobuf", "protobuf-"+protobufVersion)
		command{Dir: protobufPath}.mustRun(t, "git", "checkout", checkoutVersion)

		if os.Getenv("GO_BUILDER_NAME") != "" {
			// If this is running on the Go build infrastructure,
			// use pre-built versions of these binaries that the
			// builders are configured to provide in $PATH.
			protocPath, err := exec.LookPath("protoc")
			check(err)
			confTestRunnerPath, err := exec.LookPath("conformance_test_runner")
			check(err)
			check(os.MkdirAll(filepath.Join(protobufPath, "bazel-bin", "conformance"), 0775))
			check(os.Symlink(protocPath, filepath.Join(protobufPath, "bazel-bin", "protoc")))
			check(os.Symlink(confTestRunnerPath, filepath.Join(protobufPath, "bazel-bin", "conformance", "conformance_test_runner")))
		} else {
			// In other environments, download and build the protobuf toolchain.
			// We avoid downloading the pre-compiled binaries since they do not contain
			// the conformance test runner.
			fmt.Printf("build %v\n", filepath.Base(protobufPath))
			env := os.Environ()
			if runtime.GOOS == "darwin" {
				// Adding this environment variable appears to be necessary for macOS builds.
				env = append(env, "CC=clang")
			}
			command{
				Dir: protobufPath,
				Env: env,
			}.mustRun(t, "bazel", "build", ":protoc", "//conformance:conformance_test_runner")
		}
	}
	check(os.Setenv("PROTOBUF_ROOT", protobufPath)) // for generate-protos
	registerBinary("conform-test-runner", filepath.Join(protobufPath, "bazel-bin", "conformance", "conformance_test_runner"))
	registerBinary("protoc", filepath.Join(protobufPath, "bazel-bin", "protoc"))
	finishWork()

	// Download each Go toolchain version.
	for _, v := range golangVersions {
		goDir := startWork("go" + v)
		if _, err := os.Stat(goDir); err != nil {
			fmt.Printf("download %v\n", filepath.Base(goDir))
			url := fmt.Sprintf("https://dl.google.com/go/go%v.%v-%v.tar.gz", v, runtime.GOOS, runtime.GOARCH)
			downloadArchive(check, goDir, url, "go", "") // skip SHA256 check as we fetch over https from a trusted domain
		}
		registerBinary("go"+v, filepath.Join(goDir, "bin", "go"))
		finishWork()
	}
	registerBinary("go", filepath.Join(testDir, "go"+golangLatest, "bin", "go"))
	registerBinary("gofmt", filepath.Join(testDir, "go"+golangLatest, "bin", "gofmt"))

	// Download the staticcheck tool.
	checkDir := startWork("staticcheck-" + staticcheckVersion)
	if _, err := os.Stat(checkDir); err != nil {
		fmt.Printf("download %v\n", filepath.Base(checkDir))
		url := fmt.Sprintf("https://github.com/dominikh/go-tools/releases/download/%v/staticcheck_%v_%v.tar.gz", staticcheckVersion, runtime.GOOS, runtime.GOARCH)
		downloadArchive(check, checkDir, url, "staticcheck", staticcheckSHA256s[runtime.GOOS+"/"+runtime.GOARCH])
	}
	registerBinary("staticcheck", filepath.Join(checkDir, "staticcheck"))
	finishWork()

	// GitHub actions sets GOROOT, which confuses invocations of the Go toolchain.
	// Explicitly clear GOROOT, so each toolchain uses their default GOROOT.
	check(os.Unsetenv("GOROOT"))

	// Set a cache directory outside the test directory.
	check(os.Setenv("GOCACHE", filepath.Join(repoRoot, ".gocache")))
}

func downloadFile(check func(error), dstPath, srcURL string, perm fs.FileMode) {
	resp, err := http.Get(srcURL)
	check(err)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		check(fmt.Errorf("GET %q: non-200 OK status code: %v body: %q", srcURL, resp.Status, body))
	}

	check(os.MkdirAll(filepath.Dir(dstPath), 0775))
	f, err := os.OpenFile(dstPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	check(err)

	_, err = io.Copy(f, resp.Body)
	check(err)

	check(f.Close())
}

func downloadArchive(check func(error), dstPath, srcURL, skipPrefix, wantSHA256 string) {
	check(os.RemoveAll(dstPath))

	resp, err := http.Get(srcURL)
	check(err)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		check(fmt.Errorf("GET %q: non-200 OK status code: %v body: %q", srcURL, resp.Status, body))
	}

	var r io.Reader = resp.Body
	if wantSHA256 != "" {
		b, err := ioutil.ReadAll(resp.Body)
		check(err)
		r = bytes.NewReader(b)

		if gotSHA256 := fmt.Sprintf("%x", sha256.Sum256(b)); gotSHA256 != wantSHA256 {
			check(fmt.Errorf("checksum validation error:\ngot  %v\nwant %v", gotSHA256, wantSHA256))
		}
	}

	zr, err := gzip.NewReader(r)
	check(err)

	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return
		}
		check(err)

		// Skip directories or files outside the prefix directory.
		if len(skipPrefix) > 0 {
			if !strings.HasPrefix(h.Name, skipPrefix) {
				continue
			}
			if len(h.Name) > len(skipPrefix) && h.Name[len(skipPrefix)] != '/' {
				continue
			}
		}

		path := strings.TrimPrefix(strings.TrimPrefix(h.Name, skipPrefix), "/")
		path = filepath.Join(dstPath, filepath.FromSlash(path))
		mode := os.FileMode(h.Mode & 0777)
		switch h.Typeflag {
		case tar.TypeReg:
			b, err := ioutil.ReadAll(tr)
			check(err)
			check(ioutil.WriteFile(path, b, mode))
		case tar.TypeDir:
			check(os.Mkdir(path, mode))
		}
	}
}

func mustHandleFlags(t *testing.T) {
	if *regenerate {
		t.Run("Generate", func(t *testing.T) {
			fmt.Print(mustRunCommand(t, "go", "generate", "./internal/cmd/generate-types"))
			fmt.Print(mustRunCommand(t, "go", "generate", "./internal/cmd/generate-protos"))
			files := strings.Split(strings.TrimSpace(mustRunCommand(t, "git", "ls-files", "*.go")), "\n")
			mustRunCommand(t, append([]string{"gofmt", "-w"}, files...)...)
		})
	}
	if *buildRelease {
		t.Run("BuildRelease", func(t *testing.T) {
			v := version.String()
			for _, goos := range []string{"linux", "darwin", "windows"} {
				for _, goarch := range []string{"386", "amd64", "arm64"} {
					// Avoid Darwin since 10.15 dropped support for i386.
					if goos == "darwin" && goarch == "386" {
						continue
					}

					binPath := filepath.Join("bin", fmt.Sprintf("protoc-gen-go.%v.%v.%v", v, goos, goarch))

					// Build the binary.
					cmd := command{Env: append(os.Environ(), "GOOS="+goos, "GOARCH="+goarch)}
					cmd.mustRun(t, "go", "build", "-trimpath", "-ldflags", "-s -w -buildid=", "-o", binPath, "./cmd/protoc-gen-go")

					// Archive and compress the binary.
					in, err := ioutil.ReadFile(binPath)
					if err != nil {
						t.Fatal(err)
					}
					out := new(bytes.Buffer)
					suffix := ""
					comment := fmt.Sprintf("protoc-gen-go VERSION=%v GOOS=%v GOARCH=%v", v, goos, goarch)
					switch goos {
					case "windows":
						suffix = ".zip"
						zw := zip.NewWriter(out)
						zw.SetComment(comment)
						fw, _ := zw.Create("protoc-gen-go.exe")
						fw.Write(in)
						zw.Close()
					default:
						suffix = ".tar.gz"
						gz, _ := gzip.NewWriterLevel(out, gzip.BestCompression)
						gz.Comment = comment
						tw := tar.NewWriter(gz)
						tw.WriteHeader(&tar.Header{
							Name: "protoc-gen-go",
							Mode: int64(0775),
							Size: int64(len(in)),
						})
						tw.Write(in)
						tw.Close()
						gz.Close()
					}
					if err := ioutil.WriteFile(binPath+suffix, out.Bytes(), 0664); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
	if *regenerate || *buildRelease {
		t.SkipNow()
	}
}

var copyrightRegex = []*regexp.Regexp{
	regexp.MustCompile(`^// Copyright \d\d\d\d The Go Authors\. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file\.
`),
	// Generated .pb.go files from main protobuf repo.
	regexp.MustCompile(`^// Protocol Buffers - Google's data interchange format
// Copyright \d\d\d\d Google Inc\.  All rights reserved\.
`),
}

func mustHaveCopyrightHeader(t *testing.T, files []string) {
	var bad []string
File:
	for _, file := range files {
		b, err := ioutil.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, re := range copyrightRegex {
			if loc := re.FindIndex(b); loc != nil && loc[0] == 0 {
				continue File
			}
		}
		bad = append(bad, file)
	}
	if len(bad) > 0 {
		t.Fatalf("files with missing/bad copyright headers:\n  %v", strings.Join(bad, "\n  "))
	}
}

type command struct {
	Dir string
	Env []string
}

func (c command) mustRun(t *testing.T, args ...string) string {
	t.Helper()
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = "."
	if c.Dir != "" {
		cmd.Dir = c.Dir
	}
	cmd.Env = os.Environ()
	if c.Env != nil {
		cmd.Env = c.Env
	}
	cmd.Env = append(cmd.Env, "PWD="+cmd.Dir)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("executing (%v): %v\n%s%s", strings.Join(args, " "), err, stdout.String(), stderr.String())
	}
	return stdout.String()
}

func mustRunCommand(t *testing.T, args ...string) string {
	t.Helper()
	return command{}.mustRun(t, args...)
}

// race is an approximation of whether the race detector is on.
// It's used to skip the integration test on builders, without
// preventing the integration test from running under the race
// detector as a '//go:build !race' build constraint would.
func race() bool {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return false
	}
	// Use reflect because the debug.BuildInfo.Settings field
	// isn't available in Go 1.17.
	s := reflect.ValueOf(bi).Elem().FieldByName("Settings")
	if !s.IsValid() {
		return false
	}
	for i := 0; i < s.Len(); i++ {
		if s.Index(i).FieldByName("Key").String() == "-race" {
			return s.Index(i).FieldByName("Value").String() == "true"
		}
	}
	return false
}
