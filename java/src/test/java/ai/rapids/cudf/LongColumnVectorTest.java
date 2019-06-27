
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.Mockito.spy;

public class LongColumnVectorTest {

  @Test
  public void testCreateColumnVectorBuilder() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector longColumnVector = ColumnVector.build(DType.INT64, 3, (b) -> b.append(1L))) {
      assertFalse(longColumnVector.hasNulls());
    }
  }

  @Test
  public void testArrayAllocation() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector longColumnVector = ColumnVector.fromLongs(2L, 3L, 5L)) {
      assertFalse(longColumnVector.hasNulls());
      assertEquals(longColumnVector.getLong(0), 2);
      assertEquals(longColumnVector.getLong(1), 3);
      assertEquals(longColumnVector.getLong(2), 5);
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector longColumnVector = ColumnVector.fromLongs(2L, 3L, 5L)) {
      assertThrows(AssertionError.class, () -> longColumnVector.getLong(3));
      assertFalse(longColumnVector.hasNulls());
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector longColumnVector = ColumnVector.fromLongs(2L, 3L, 5L)) {
      assertFalse(longColumnVector.hasNulls());
      assertThrows(AssertionError.class, () -> longColumnVector.getLong(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector cv = ColumnVector.fromBoxedLongs(2L, 3L, 4L, 5L, 6L, 7L, null, null)) {
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
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.INT64, 3)) {
      assertThrows(AssertionError.class,
          () -> builder.append(2L).appendNull().append(5L).append(4L).build());
    }
  }
}
