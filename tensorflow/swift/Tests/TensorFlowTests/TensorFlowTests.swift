import XCTest
@testable import TensorFlow

class TensorFlowTests: XCTestCase {

    func testSession() {
        _ = Session()
    }

    static var allTests : [(String, (TensorFlowTests) -> () throws -> Void)] {
        return [
        ]
    }
}
