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

public class IntColumnVectorTest {

  @Test
  public void testCreateColumnVectorBuilder() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector intColumnVector = ColumnVector.build(DType.INT32, 3, (b) -> b.append(1))) {
      assertFalse(intColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector intColumnVector = ColumnVector.fromInts(2, 3, 5)) {
      assertFalse(intColumnVector.hasNulls());
      assertEquals(intColumnVector.getInt(0), 2);
      assertEquals(intColumnVector.getInt(1), 3);
      assertEquals(intColumnVector.getInt(2), 5);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector intColumnVector = ColumnVector.fromInts(2, 3, 5)) {
      assertThrows(AssertionError.class, () -> intColumnVector.getInt(3));
      assertFalse(intColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector intColumnVector = ColumnVector.fromInts(2, 3, 5)) {
      assertFalse(intColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> intColumnVector.getInt(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector cv = ColumnVector.fromBoxedInts(2, 3, 4, 5, 6, 7, null, null)) {
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
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT32, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2).appendNull().appendArray(new int[]{5, 4}).build());
    }
  }

  @Test
  public void testCastToInt() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector doubleColumnVector = ColumnVector.fromDoubles(new double[]{4.3, 3.8, 8});
         ColumnVector shortColumnVector = ColumnVector.fromShorts(new short[]{100});
         ColumnVector intColumnVector1 = doubleColumnVector.asInts();
         ColumnVector intColumnVector2 = shortColumnVector.asInts()) {
      intColumnVector1.ensureOnHost();
      intColumnVector2.ensureOnHost();
      assertEquals(4, intColumnVector1.getInt(0));
      assertEquals(3, intColumnVector1.getInt(1));
      assertEquals(8, intColumnVector1.getInt(2));
      assertEquals(100, intColumnVector2.getInt(0));
    }
  }
}
