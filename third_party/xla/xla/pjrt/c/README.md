# PJRT - Uniform Device API

PJRT C API is the uniform Device API that we want to add to the ML ecosystem.
The long term vision is that: (1) frameworks (TF, JAX, etc.) will call PJRT,
which has device-specific implementations that are opaque to the frameworks; (2)
each device focuses on implementing PJRT APIs as PJRT plugins, which can be
opaque to the frameworks.

## Communication channel

*   Please file issues in the [OpenXla/xla repo](https://github.com/openxla/xla).
*   Join the [pjrt-announcement maillist](https://groups.google.com/g/pjrt-announce/).

## Resources

*   [PJRT C API header](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h)
*   [PJRT C API changelog](https://github.com/openxla/xla/blob/main/xla/pjrt/c/CHANGELOG.md)
*   [PJRT C++ API Overview](https://github.com/openxla/xla/blob/main/xla/docs/pjrt/cpp_api_overview.md)
*   [PJRT integration guide](https://github.com/openxla/xla/blob/main/xla/docs/pjrt/pjrt_integration.md)
*   [PJRT design docs](https://drive.google.com/drive/folders/18M944-QQPk1E34qRyIjkqDRDnpMa3miN)
*   [PJRT API ABI versioning and compatibility](https://docs.google.com/document/d/1TKB5NyGtdzrpgw5mpyFjVAhJjpSNdF31T6pjPl_UT2o/edit)
*   [PJRT Plugin Mechanism design doc](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit)
*   [OpenXLA/IREE PJRT plugin implementation](https://github.com/openxla/openxla-pjrt-plugin)
