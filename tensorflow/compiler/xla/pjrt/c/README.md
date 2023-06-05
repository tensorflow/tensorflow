# PJRT - Uniform Device API

PJRT C API is the uniform Device API that we want to add to the ML ecosystem.
The long term vision is that: (1) frameworks (TF, JAX, etc.) will call PJRT,
which has device-specific implementations that are opaque to the frameworks; (2)
each device focuses on implementing PJRT APIs as PJRT plugins, which can be
opaque to the frameworks.

## Communication channel

*   Please file issues in the [OpenXla/xla repo](https://github.com/openxla/xla).
*   Join discussion in the #pjrt-plugin channel of the [IREE discord server](https://github.com/openxla/iree/#communication-channels).

## Resources

*   [OpenXLA/IREE PJRT plugin implementation](https://github.com/openxla/openxla-pjrt-plugin)
*   [PJRT integration guide](https://docs.google.com/document/d/1KV_p6aa-u5v_U71_SdtyBDdH-qncY-dUr47-wIUJQ_s/edit?resourcekey=0-mzaosg6Id0JLpEoT_O47LA#heading=h.xsz0btxjkdo3)
*   [PJRT Plugin Mechanism design doc](https://docs.google.com/document/d/1Qdptisz1tUPGn1qFAVgCV2omnfjN01zoQPwKLdlizas/edit)
