import UIKit

@UIApplicationMain

final class AppDelegate: UIResponder, UIApplicationDelegate {

  /// The main window of the app.
  var window: UIWindow?

  func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
  ) -> Bool {
    return true
  }
}

// MARK: - Extensions

#if !swift(>=4.2)
extension UIApplication {
  typealias LaunchOptionsKey = UIApplicationLaunchOptionsKey
}
#endif  // !swift(>=4.2)
