tensorflow {
  global:
    *tensorflow*;
    *perftools*gputools*;
    *TF_*;
    *TFE_*;
    *nsync_*;
    *pywrap_xla*;
    *stream_executor*;
  local:
    *;
};
