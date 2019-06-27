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

public class FloatColumnVectorTest {

  @Test
  public void testCreateColumnVectorBuilder() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector floatColumnVector = ColumnVector.build(DType.FLOAT32, 3,
        (b) -> b.append(1.0f))) {
      assertFalse(floatColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
      assertFalse(floatColumnVector.hasNulls());
      assertEquals(floatColumnVector.getFloat(0), 2.1, 0.01);
      assertEquals(floatColumnVector.getFloat(1), 3.02, 0.01);
      assertEquals(floatColumnVector.getFloat(2), 5.003, 0.001);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
      assertThrows(AssertionError.class, () -> floatColumnVector.getFloat(3));
      assertFalse(floatColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector floatColumnVector = ColumnVector.fromFloats(2.1f, 3.02f, 5.003f)) {
      assertFalse(floatColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> floatColumnVector.getFloat(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector cv = ColumnVector.fromBoxedFloats(
        new Float[]{2f, 3f, 4f, 5f, 6f, 7f, null, null})) {
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
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.FLOAT32, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2.1f).appendNull().appendArray(5.003f, 4.0f).build());
    }
  }

  @Test
  public void testCastToFloat() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(new double[]{4.3, 3.8, 8});
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector floatColumnVector1 = doubleColumnVector.asFloats();
         ColumnVector floatColumnVector2 = shortColumnVector.asFloats()) {
      floatColumnVector1.ensureOnHost();
      floatColumnVector2.ensureOnHost();
      assertEquals(4.3, floatColumnVector1.getFloat(0), 0.001);
      assertEquals(3.8, floatColumnVector1.getFloat(1), 0.001);
      assertEquals(8, floatColumnVector1.getFloat(2));
      assertEquals(100, floatColumnVector2.getFloat(0));
    }
  }
}
