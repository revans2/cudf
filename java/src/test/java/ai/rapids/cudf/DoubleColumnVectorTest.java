/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.spy;

public class DoubleColumnVectorTest {

  @Test
  public void testCreateColumnVectorBuilder() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector doubleColumnVector = ColumnVector.build(DType.FLOAT64, 3,
        (b) -> b.append(1.0))) {
      assertFalse(doubleColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(2.1, 3.02, 5.003)) {
      assertFalse(doubleColumnVector.hasNulls());
      assertEquals(doubleColumnVector.getDouble(0), 2.1, 0.01);
      assertEquals(doubleColumnVector.getDouble(1), 3.02, 0.01);
      assertEquals(doubleColumnVector.getDouble(2), 5.003, 0.001);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(2.1, 3.02, 5.003)) {
      assertThrows(AssertionError.class, () -> doubleColumnVector.getDouble(3));
      assertFalse(doubleColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(2.1, 3.02, 5.003)) {
      assertFalse(doubleColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> doubleColumnVector.getDouble(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector cv =
             ColumnVector.fromBoxedDoubles(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, null, null)) {
      assertTrue(cv.hasNulls());
      assertEquals(2, cv.getNullCount());
      for (int i = 0; i < 6; i++) {
        assertFalse(cv.isNull(i));
      }
      assertTrue(cv.isNull(6));
      assertTrue(cv.isNull(7));
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.FLOAT64, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2.1).appendNull().appendArray(new double[]{5.003, 4.0}).build());
    }
  }
}
