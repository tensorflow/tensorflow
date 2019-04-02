# Run `pod lib lint TensorFlowLiteObjC.podspec` to ensure this is a valid spec.

Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteObjC'
  s.version          = '0.1.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :git => 'https://github.com/tensorflow/tensorflow.git', :tag => 'v2.0.0-alpha0' }
  s.summary          = 'TensorFlow Lite for Objective-C'
  s.description      = <<-DESC

  TensorFlow Lite is TensorFlow's lightweight solution for Objective-C
  developers. It enables low-latency inference of on-device machine learning
  models with a small binary size and fast performance supporting hardware
  acceleration.
                       DESC

  s.ios.deployment_target = '9.0'

  s.module_name = 'TFLTensorFlowLite'
  s.static_framework = true

  base_dir = 'tensorflow/lite/experimental/objc/'
  s.public_header_files = base_dir + 'apis/*.h'
  s.source_files = base_dir + '{apis,sources}/*.{h,m,mm}'
  s.module_map = base_dir + 'apis/framework.modulemap'
  s.dependency 'TensorFlowLiteC', "#{s.version}"
  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' =>
      '"${PODS_TARGET_SRCROOT}" ' +
      '"${PODS_TARGET_SRCROOT}/' + base_dir  + 'apis"',
  }

  s.test_spec 'Tests' do |ts|
    ts.source_files = base_dir + 'tests/*.m'
  end
end
