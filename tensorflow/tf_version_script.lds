tensorflow {
  global:
    *tensorflow*;
    *perftools*gputools*;
    *xla*;
    *protobuf*Arena*AddListNode*;
    *protobuf*Arena*AllocateAligned*;
    *protobuf*internal*RepeatedPtrFieldBase*;
    *protobuf*internal*fixed_address_empty_string*;
    *protobuf*internal*LogMessage*;
  local:
    *;
};
