/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace FlatBuffers.Test
{
    static class Program
    {
        public static int Main(string[] args)
        {
            var testResults = new List<bool>();

            var testClasses = Assembly.GetExecutingAssembly().GetExportedTypes()
                .Where(t => t.IsClass && t.GetCustomAttributes(typeof (FlatBuffersTestClassAttribute), false).Length > 0);

            foreach (var testClass in testClasses)
            {
                var methods = testClass.GetMethods(BindingFlags.Public |
                                                         BindingFlags.Instance)
                          .Where(m => m.GetCustomAttributes(typeof(FlatBuffersTestMethodAttribute), false).Length > 0);

                var inst = Activator.CreateInstance(testClass);

                foreach (var method in methods)
                {
                    try
                    {
                        method.Invoke(inst, new object[] { });
                        testResults.Add(true);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("{0}: FAILED when invoking {1} with error {2}",
                            testClass.Name ,method.Name, ex.GetBaseException());
                        testResults.Add(false);
                    }
                }
            }

            var failedCount = testResults.Count(i => i == false);

            Console.WriteLine("{0} tests run, {1} failed", testResults.Count, failedCount);

            if (failedCount > 0)
            {
                return -1;
            }
            return 0;
        }
    }
}
