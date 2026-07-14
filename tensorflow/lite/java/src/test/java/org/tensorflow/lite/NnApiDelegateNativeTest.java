/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegateImpl;

/** Unit tests for {@link org.tensorflow.lite.nnapi.NnApiDelegate}. */
@RunWith(JUnit4.class)
public final class NnApiDelegateNativeTest {
  @Before
  public void setUp() throws Exception {
    TestInit.init();
  }

  private static native boolean getAllowFp16Option(long delegateHandle);

  private static native boolean getDisallowNnapiCpuOption(long delegateHandle);

  private static native String getModelTokenOption(long delegateHandle);

  @Test
  public void testCorrectOptions() {
    // Create NNAPI delegate and DelegateImpl.
    NnApiDelegate.Options options = new NnApiDelegate.Options();
    options.setAllowFp16(false).setUseNnapiCpu(false).setModelToken("ABC");
    NnApiDelegateImpl delegateImpl = new NnApiDelegateImpl(options);
    NnApiDelegate nnapiDelegate = new NnApiDelegate(options);

    // Mock the Interpreter to return specific DelegateImpl.
    InterpreterFactoryApi mockInterpreterFactory = mock(InterpreterFactoryApi.class);
    when(mockInterpreterFactory.createNnApiDelegateImpl(eq(options))).thenReturn(delegateImpl);

    nnapiDelegate.initWithInterpreterFactoryApi(mockInterpreterFactory);

    // Verify mocking called exactly once.
    verify(mockInterpreterFactory, times(1)).createNnApiDelegateImpl(options);

    // Verify constructed delegate options is as expected with no errors.
    assertThat(nnapiDelegate.getNnapiErrno()).isEqualTo(0);
    assertThat(getAllowFp16Option(delegateImpl.getNativeHandle())).isFalse();
    assertThat(getDisallowNnapiCpuOption(delegateImpl.getNativeHandle())).isTrue();
    assertThat(getModelTokenOption(delegateImpl.getNativeHandle())).isEqualTo("ABC");
  }
}
