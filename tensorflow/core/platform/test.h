#ifndef TENSORFLOW_PLATFORM_TEST_H_
#define TENSORFLOW_PLATFORM_TEST_H_

namespace tensorflow {
namespace testing {

// Return a temporary directory suitable for temporary testing files.
string TmpDir();

// Return a random number generator seed to use in randomized tests.
// Returns the same value for the lifetime of the process.
int RandomSeed();

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_TEST_H_
