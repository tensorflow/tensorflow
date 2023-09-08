# Proofs for TryRemoveUpcastAndDowncastSurroundingBinaryOp

We want to verify for which types and ops the operation
`upcasting_convert -> binary_op -> downcasting_convert` is equivalent to
`binary_op` in the original precision.

XLA types analyzed:
 - s8
 - s16
 - s32
 - s64
 - u8
 - u16
 - u32
 - u64

Binary ops:
- Add
- Subtract
- Multiply
- Divide
- Remainder

## Integer Tests

For integers, we verified using [Alive2](https://alive2.llvm.org/ce/). Tests
were only done with types `i8`->`i16`->`i8`; this should generalize to other
integer types. For integer tests, the online version of the tool was sufficient.

### Add/Sub/Mul

Below is a test for signed integer addition. To test signed and unsigned, I
changed `sext` to `zext`. For subtraction and multiplication, I replaced the
`add` with `sub` or `mul` respectively. All of these show that these
transformations are safe. I did not use `nsw` or `nuw` because we do not use
them in XLA. In XLA, when creating add/sub/mul, we pass no additional arguments
apart from rhs and lhs, which means that nsw and nuw are false.

```
define i8 @src(i8, i8) {
  %cast1 = sext i8 %0 to i16
  %cast2 = sext i8 %1 to i16
  %sum = add i16 %cast1, %cast2
  %trunc = trunc i16 %sum to i8
  ret i8 %trunc
}

define i8 @tgt(i8, i8) {
  %r = add i8 %0, %1
  ret i8 %r
}
```

### Div

Only unsigned division is safe, there can be overflow in signed integer
division. This test shows that it “seems to be correct” for unsigned integers.

```
define i8 @src(i8, i8) {
  %cast1 = zext i8 %0 to i16
  %cast2 = zext i8 %1 to i16
  %sum = udiv i16 %cast1, %cast2
  %trunc = trunc i16 %sum to i8
  ret i8 %trunc
}

define i8 @tgt(i8, i8) {
  %r = udiv i8 %0, %1
  ret i8 %r
}
```

### Remainder

Similarly for the remainder op, unsigned is safe and signed is not safe. This
test shows that it “seems to be correct” for unsigned integers.

```
define i8 @src(i8, i8) {
  %cast1 = zext i8 %0 to i16
  %cast2 = zext i8 %1 to i16
  %sum = urem i16 %cast1, %cast2
  %trunc = trunc i16 %sum to i8
  ret i8 %trunc
}

define i8 @tgt(i8, i8) {
  %r = urem i8 %0, %1
  ret i8 %r
}
```

## Next Steps

Assess the validity of floating point and complex types for these
transformations.

### Floating Point Tests

Floating point tests timeout on the online Alive2 tool. Cloning, building, and
running the tool locally without timeout on the below test has taken at least
10,000 minutes without finishing:

```
  %cast1 = fpext half %0 to float
  %cast2 = fpext half %1 to float
  %sum = fadd float %cast1, %cast2
  %trunc = fptrunc float %sum to half
  ret half %trunc
=>
  %r = fadd half %0, %1
  ret half %r
```