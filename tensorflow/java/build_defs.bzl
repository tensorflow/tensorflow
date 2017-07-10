# -*- Python -*-

# Pin to Java 1.7 to ensure broader compatibility for the Java bindings on
# Android. Note also that the android_library bazel rule currently enforces
# java 7
# https://github.com/bazelbuild/bazel/blob/6c1106b1a721516d3b3db54d2e1c31b44a76fbb1/src/main/java/com/google/devtools/build/lib/bazel/rules/android/BazelAndroidSemantics.java#L73

JAVA_VERSION_OPTS = [
    "-source 7 -target 7",
]

# A more robust set of lint and errorprone checks when building
# Java source to improve code consistency.

XLINT_OPTS = [
    "-Werror",
    "-Xlint:all",
    "-Xlint:-serial",
    "-Xlint:-try",
]

# The bazel errorprone plugin currently only enables default errorChecks
# https://github.com/bazelbuild/bazel/blob/97975603e5ff2247e6bb352e3afd27fea38f108d/src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins/errorprone/ErrorPronePlugin.java#L52
#
# Default errorChecks are errorprone checkers listed under ENABLED_ERRORS at
# https://github.com/google/error-prone/blob/c6f24bc387989158d99af28e7ae86755e56c5f38/core/src/main/java/com/google/errorprone/scanner/BuiltInCheckerSuppliers.java#L273
#
# Here we enable all available errorprone checks to converge on a consistent
# code style.
# https://github.com/google/error-prone/blob/c6f24bc387989158d99af28e7ae86755e56c5f38/core/src/main/java/com/google/errorprone/scanner/BuiltInCheckerSuppliers.java#L260

# This list is from ENABLED_WARNINGS in
# com/google/errorprone/scanner/BuiltInCheckerSuppliers.java
EP_ENABLED_WARNINGS = [
    "-Xep:AmbiguousMethodReference:ERROR",
    "-Xep:ArgumentSelectionDefectChecker:ERROR",
    "-Xep:AssertEqualsArgumentOrderChecker:ERROR",
    "-Xep:BadAnnotationImplementation:ERROR",
    "-Xep:BadComparable:ERROR",
    "-Xep:BoxedPrimitiveConstructor:ERROR",
    "-Xep:CannotMockFinalClass:ERROR",
    "-Xep:ClassCanBeStatic:ERROR",
    "-Xep:ClassNewInstance:ERROR",
    "-Xep:DefaultCharset:ERROR",
    "-Xep:DoubleCheckedLocking:ERROR",
    "-Xep:ElementsCountedInLoop:ERROR",
    "-Xep:EqualsHashCode:ERROR",
    "-Xep:EqualsIncompatibleType:ERROR",
    "-Xep:Finally:ERROR",
    "-Xep:FloatingPointLiteralPrecision:ERROR",
    "-Xep:FragmentInjection:ERROR",
    "-Xep:FragmentNotInstantiable:ERROR",
    "-Xep:FunctionalInterfaceClash:ERROR",
    "-Xep:FutureReturnValueIgnored:ERROR",
    "-Xep:GetClassOnEnum:ERROR",
    "-Xep:ImmutableAnnotationChecker:ERROR",
    "-Xep:ImmutableEnumChecker:ERROR",
    "-Xep:IncompatibleModifiers:ERROR",
    "-Xep:InjectOnConstructorOfAbstractClass:ERROR",
    "-Xep:InputStreamSlowMultibyteRead:ERROR",
    "-Xep:IterableAndIterator:ERROR",
    "-Xep:JavaLangClash:ERROR",
    "-Xep:JUnit3FloatingPointComparisonWithoutDelta:ERROR",
    "-Xep:JUnitAmbiguousTestClass:ERROR",
    "-Xep:LiteralClassName:ERROR",
    "-Xep:LogicalAssignment:ERROR",
    "-Xep:MissingFail:ERROR",
    "-Xep:MissingOverride:ERROR",
    "-Xep:MutableConstantField:ERROR",
    "-Xep:NamedParameters:ERROR",
    "-Xep:NarrowingCompoundAssignment:ERROR",
    "-Xep:NonAtomicVolatileUpdate:ERROR",
    "-Xep:NonOverridingEquals:ERROR",
    "-Xep:NullableConstructor:ERROR",
    "-Xep:NullablePrimitive:ERROR",
    "-Xep:NullableVoid:ERROR",
    "-Xep:OperatorPrecedence:ERROR",
    "-Xep:OverridesGuiceInjectableMethod:ERROR",
    "-Xep:PreconditionsInvalidPlaceholder:ERROR",
    "-Xep:ProtoFieldPreconditionsCheckNotNull:ERROR",
    "-Xep:ReferenceEquality:ERROR",
    "-Xep:RequiredModifiers:ERROR",
    "-Xep:ShortCircuitBoolean:ERROR",
    "-Xep:SimpleDateFormatConstant:ERROR",
    "-Xep:StaticGuardedByInstance:ERROR",
    "-Xep:SynchronizeOnNonFinalField:ERROR",
    "-Xep:TruthConstantAsserts:ERROR",
    "-Xep:TypeParameterShadowing:ERROR",
    "-Xep:TypeParameterUnusedInFormals:ERROR",
    "-Xep:UnsynchronizedOverridesSynchronized:ERROR",
    "-Xep:URLEqualsHashCode:ERROR",
    "-Xep:WaitNotInLoop:ERROR",
]

# This list is from DISABLED_CHECKS in
# com/google/errorprone/scanner/BuiltInCheckerSuppliers.java
EP_DISABLED_CHECKS = [
    "-Xep:AutoFactoryAtInject:ERROR",
    "-Xep:AssertFalse:ERROR",
    "-Xep:AssistedInjectAndInjectOnConstructors:ERROR",
    "-Xep:AssistedInjectAndInjectOnSameConstructor:ERROR",
    "-Xep:BigDecimalLiteralDouble:ERROR",
    "-Xep:BindingToUnqualifiedCommonType:ERROR",
    "-Xep:ClassName:ERROR",
    "-Xep:ComparisonContractViolated:ERROR",
    "-Xep:ConstantField:ERROR",
    "-Xep:ConstructorInvokesOverridable:ERROR",
    # False positives, disabled
    # "-Xep:ConstructorLeaksThis:ERROR",
    "-Xep:DepAnn:ERROR",
    "-Xep:DivZero:ERROR",
    "-Xep:EmptyIfStatement:ERROR",
    "-Xep:EmptySetMultibindingContributions:ERROR",
    "-Xep:EmptyTopLevelDeclaration:ERROR",
    "-Xep:ExpectedExceptionChecker:ERROR",
    "-Xep:HardCodedSdCardPath:ERROR",
    "-Xep:InjectedConstructorAnnotations:ERROR",
    "-Xep:InsecureCipherMode:ERROR",
    "-Xep:InvalidTargetingOnScopingAnnotation:ERROR",
    "-Xep:IterablePathParameter:ERROR",
    "-Xep:JMockTestWithoutRunWithOrRuleAnnotation:ERROR",
    "-Xep:JavaxInjectOnFinalField:ERROR",
    "-Xep:LockMethodChecker:ERROR",
    "-Xep:LongLiteralLowerCaseSuffix:ERROR",
    "-Xep:MethodCanBeStatic:ERROR",
    "-Xep:MissingDefault:ERROR",
    "-Xep:MixedArrayDimensions:ERROR",
    "-Xep:MoreThanOneQualifier:ERROR",
    "-Xep:MultiVariableDeclaration:ERROR",
    "-Xep:MultipleTopLevelClasses:ERROR",
    "-Xep:NoAllocationChecker:ERROR",
    "-Xep:NonCanonicalStaticMemberImport:ERROR",
    "-Xep:NumericEquality:ERROR",
    "-Xep:PackageLocation:ERROR",
    "-Xep:PrimitiveArrayPassedToVarargsMethod:ERROR",
    "-Xep:PrivateConstructorForUtilityClass:ERROR",
    "-Xep:PrivateConstructorForNoninstantiableModule:ERROR",
    "-Xep:ProtoStringFieldReferenceEquality:ERROR",
    "-Xep:QualifierOrScopeOnInjectMethod:ERROR",
    "-Xep:QualifierWithTypeUse:ERROR",
    "-Xep:RedundantThrows:ERROR",
    "-Xep:RemoveUnusedImports:ERROR",
    "-Xep:ScopeAnnotationOnInterfaceOrAbstractClass:ERROR",
    "-Xep:ScopeOrQualifierAnnotationRetention:ERROR",
    "-Xep:StaticQualifiedUsingExpression:ERROR",
    "-Xep:StaticOrDefaultInterfaceMethod:ERROR",
    "-Xep:StringEquality:ERROR",
    "-Xep:TestExceptionChecker:ERROR",
    # TODO: stylistic changes in code
    # "-Xep:ThrowsUncheckedException:ERROR",
    # "-Xep:UngroupedOverloads:ERROR",
    "-Xep:UnlockMethodChecker:ERROR",
    "-Xep:UnnecessaryDefaultInEnumSwitch:ERROR",
    "-Xep:UnnecessaryStaticImport:ERROR",
    "-Xep:UseBinds:ERROR",
    "-Xep:VarChecker:ERROR",
    "-Xep:WildcardImport:ERROR",
    "-Xep:WrongParameterPackage:ERROR",
]

EP_OPTS = EP_ENABLED_WARNINGS + EP_DISABLED_CHECKS

JAVACOPTS = JAVA_VERSION_OPTS + XLINT_OPTS + EP_OPTS
