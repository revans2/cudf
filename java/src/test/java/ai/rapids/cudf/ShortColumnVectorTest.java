
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

public class ShortColumnVectorTest {

  @Test
  public void testCreateColumnVectorBuilder() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector shortColumnVector = ColumnVector.build(DType.INT16, 3,
        (b) -> b.append((short) 1))) {
      assertFalse(shortColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector shortColumnVector = ColumnVector.fromShorts((short) 2, (short) 3,
        (short) 5)) {
      assertFalse(shortColumnVector.hasNulls());
      assertEquals(shortColumnVector.getShort(0), 2);
      assertEquals(shortColumnVector.getShort(1), 3);
      assertEquals(shortColumnVector.getShort(2), 5);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector shortColumnVector = ColumnVector.fromShorts((short) 2, (short) 3,
        (short) 5)) {
      assertThrows(AssertionError.class, () -> shortColumnVector.getShort(3));
      assertFalse(shortColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector shortColumnVector = ColumnVector.fromShorts((short) 2, (short) 3,
        (short) 5)) {
      assertFalse(shortColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> shortColumnVector.getShort(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector cv =
             ColumnVector.fromBoxedShorts(new Short[]{2, 3, 4, 5, 6, 7, null, null})) {
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
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT16, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append((short) 2).appendNull().appendArray(new short[]{5, 4}).build());
    }
  }
}
