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

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static ai.rapids.cudf.TableTest.assertPartialColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class ColumnVectorTest {
  @Test
  void testRefCountLeak() throws InterruptedException {
    assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
    long expectedLeakCount = MemoryCleaner.leakCount.get() + 1;
    ColumnVector.fromInts(1, 2, 3);
    long maxTime = System.currentTimeMillis() + 10_000;
    long leakNow;
    do {
      System.gc();
      Thread.sleep(50);
      leakNow = MemoryCleaner.leakCount.get();
    } while (leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
    assertEquals(expectedLeakCount, MemoryCleaner.leakCount.get());
  }

  @Test
  void testConcatTypeError() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromFloats(5.0f, 6.0f)) {
      assertThrows(CudfException.class, () -> ColumnVector.concatenate(v0, v1));
    }
  }

  @Test
  void testConcatNoNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v0 = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromInts(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromInts(8, 9);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertFalse(v.hasNulls());
      assertFalse(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        assertEquals(i + 1, v.getInt(i), "at index " + i);
      }
    }
  }

  @Test
  void testConcatWithNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v0 = ColumnVector.fromDoubles(1, 2, 3, 4);
         ColumnVector v1 = ColumnVector.fromDoubles(5, 6, 7);
         ColumnVector v2 = ColumnVector.fromBoxedDoubles(null, 9.0);
         ColumnVector v = ColumnVector.concatenate(v0, v1, v2)) {
      v.ensureOnHost();
      assertEquals(9, v.getRowCount());
      assertTrue(v.hasNulls());
      assertTrue(v.hasValidityVector());
      for (int i = 0; i < 9; ++i) {
        if (i != 7) {
          assertEquals(i + 1, v.getDouble(i), "at index " + i);
        } else {
          assertTrue(v.isNull(i), "at index " + i);
        }
      }
    }
  }

  public static ColumnVector.Builder randomLongBuilder(int size) {
    return randomLongBuilder(size, size, ThreadLocalRandom.current());
  }

  public static ColumnVector.Builder randomLongBuilder(int size, int numRandom) {
    return randomLongBuilder(size, numRandom, ThreadLocalRandom.current());
  }

  public static ColumnVector.Builder randomLongBuilder(int size, Random random) {
    return randomLongBuilder(size, size, random);
  }

  public static ColumnVector.Builder randomLongBuilder(int size, int numRandom, Random random) {
    ColumnVector.Builder builder = ColumnVector.builder(DType.INT64, size);
    boolean needsCleanup = true;
    try {
      for (int i = 0; i < numRandom; i++) {
        if (random.nextBoolean()) {
          builder.appendNull();
        } else {
          builder.append(random.nextLong());
        }
      }
      needsCleanup = false;
      return builder;
    } finally {
      if (needsCleanup) {
        builder.close();
      }
    }
  }

  @Test
  void testAppendVector() {
    Random random = new Random(192312989128L);
    // validity is the tricky part to get right when shifting the bits around, so make sure we
    // test having the split between the vectors be in different places.
    final int MAX_SIZE = 8 * 3 + 7;
    for (int combinedSize = 1; combinedSize <= MAX_SIZE; combinedSize++) {
      for (int firstPartSize = 0; firstPartSize < combinedSize; firstPartSize++) {
        final int secondPartSize = combinedSize - firstPartSize;
        try (ColumnVector firstPart = randomLongBuilder(firstPartSize, random).buildOnHost();
             ColumnVector secondPart = randomLongBuilder(secondPartSize, random).buildOnHost();
             ColumnVector combined = ColumnVector.build(DType.INT64, combinedSize, (b)-> {
               b.append(firstPart);
               b.append(secondPart);
             })) {
          assertPartialColumnsAreEqual(combined, 0, firstPartSize, firstPart, "firstPart");
          assertPartialColumnsAreEqual(combined, firstPartSize, secondPartSize, secondPart, "secondPart");
          if (combined.hasValidityVector()) {
            long maxIndex =
                BitVectorHelper.getValidityAllocationSizeInBytes(combined.getRowCount()) * 8;
            for (long i = combinedSize; i < maxIndex; i++) {
              assertFalse(combined.isNullExtendedRange(i));
            }
          }
        }
      }
    }
  }
}
