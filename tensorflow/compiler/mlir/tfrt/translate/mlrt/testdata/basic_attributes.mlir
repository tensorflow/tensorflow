func.func @simple_attributes() {
  "test_custom.attribute"() {value = "test string"} : () -> ()
  "test_custom.attribute"() {value = "ts"} : () -> ()
  "test_custom.attribute"() {value = 100 : i32} : () -> ()
  "test_custom.attribute"() {value = 200 : i64} : () -> ()
  "test_custom.attribute"() {value = 3.0 : f32} : () -> ()
  "test_custom.attribute"() {value = false} : () -> ()
  "test_custom.attribute"() {value = [0, 1, 2, 3, 4]} : () -> ()
  "test_custom.attribute"() {value = [0 : i32, 1 : i32, 2 : i32, 3 : i32]} : () -> ()
  "test_custom.attribute"() {value = ["string 0", "string 1"]} : () -> ()
  "test_custom.attribute"() {value = @callee} : () -> ()
  "test_custom.attribute"() {value = [@callee0, @callee1]} : () -> ()
  "test_custom.attribute"() {value = array<i32: 0, 1, 2>} : () -> ()
  "test_custom.attribute"() {value = array<i64: 0, 1, 2>} : () -> ()
  "test_custom.attribute"() {value = array<i32>} : () -> ()
  "test_custom.attribute"() {value = array<i1: true, false>} : () -> ()
  func.return
}

func.func @callee() {
  return
}

func.func @callee0() {
  return
}

func.func @callee1() {
  return
}
