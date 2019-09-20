#ifndef FUZZER_ASSERT_IMPL_H_
#define FUZZER_ASSERT_IMPL_H_

// Declare Debug/Release independed assert macro.
#define fuzzer_assert_impl(x) (!!(x) ? static_cast<void>(0) : __builtin_trap())

extern "C" void __builtin_trap(void);

#endif // !FUZZER_ASSERT_IMPL_H_
