#include <tests/strings/utilities.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(GatherTest, cudf::test::NumericTypes);

// This test exercises using different iterator types as gather map inputs
// to cudf::detail::gather -- device_vector and raw pointers.
TYPED_TEST(GatherTest, GatherDetailDeviceVectorTest)
{
  constexpr cudf::size_type source_size{1000};
  rmm::device_vector<cudf::size_type> gather_map(source_size);
  thrust::sequence(thrust::device, gather_map.begin(), gather_map.end());

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);

  cudf::table_view source_table({source_column});

  // test with device vector iterators
  {
    std::unique_ptr<cudf::table> result =
      cudf::detail::gather(source_table, gather_map.begin(), gather_map.end());

    for (auto i = 0; i < source_table.num_columns(); ++i) {
      cudf::test::expect_columns_equal(source_table.column(i), result->view().column(i));
    }

    cudf::test::expect_tables_equal(source_table, result->view());
  }

  // test with raw pointers
  {
    std::unique_ptr<cudf::table> result = cudf::detail::gather(
      source_table, gather_map.data().get(), gather_map.data().get() + gather_map.size());

    for (auto i = 0; i < source_table.num_columns(); ++i) {
      cudf::test::expect_columns_equal(source_table.column(i), result->view().column(i));
    }

    cudf::test::expect_tables_equal(source_table, result->view());
  }
}

TYPED_TEST(GatherTest, GatherDetailInvalidIndexTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  auto gather_map_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 2) ? -1 : i; });
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(gather_map_data,
                                                             gather_map_data + (source_size * 2));

  cudf::table_view source_table({source_column});
  std::unique_ptr<cudf::table> result =
    cudf::detail::gather(source_table,
                         gather_map,
                         cudf::detail::out_of_bounds_policy::IGNORE,
                         cudf::detail::negative_index_policy::NOT_ALLOWED);

  auto expect_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 2) ? 0 : i; });
  auto expect_valid = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return (i % 2) || (i >= source_size) ? 0 : 1; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + (source_size * 2), expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, IdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(source_table.column(i), result->view().column(i));
  }

  cudf::test::expect_tables_equal(source_table, result->view());
}

TYPED_TEST(GatherTest, ReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto reversed_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullOdds)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather odd-valued indices
  auto map_data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i * 2; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));

  auto expect_data  = cudf::test::make_counting_transform_iterator(0, [](auto i) { return 0; });
  auto expect_valid = cudf::test::make_counting_transform_iterator(0, [](auto i) { return 0; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullEvens)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather even-valued indices
  auto map_data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));

  auto expect_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });
  auto expect_valid = cudf::test::make_counting_transform_iterator(0, [](auto i) { return 1; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, AllNull)
{
  constexpr cudf::size_type source_size{1000};

  // Every element is invalid
  auto data     = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return 0; });

  // Create a gather map that gathers to random locations
  std::vector<cudf::size_type> host_map_data(source_size);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::mt19937 g(0);
  std::shuffle(host_map_data.begin(), host_map_data.end(), g);
  thrust::device_vector<cudf::size_type> map_data(host_map_data);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column{
    data, data + source_size, validity};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data.begin(), map_data.end());

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));

  // Check that the result is also all invalid
  cudf::test::expect_tables_equal(source_table, result->view());
}

TYPED_TEST(GatherTest, MultiColReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  constexpr cudf::size_type n_cols = 3;

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto reversed_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size));
    source_columns.push_back(source_column_wrappers[i]);
  }

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, MultiColNulls)
{
  constexpr cudf::size_type source_size{1000};

  static_assert(0 == source_size % 2, "Size of source data must be a multiple of 2.");

  constexpr cudf::size_type n_cols = 3;

  auto data     = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size, validity));
    source_columns.push_back(source_column_wrappers[i]);
  }

  auto reversed_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = std::move(cudf::gather(source_table, gather_map));

  // Expected data
  auto expect_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return source_size - i - 1; });
  auto expect_valid =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i + 1) % 2; });

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}

class GatherTestStr : public cudf::test::BaseFixture {
};

TEST_F(GatherTestStr, StringColumn)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{1, 2, 3, 4, 5, 6}, {1, 1, 0, 1, 0, 1}};
  cudf::test::strings_column_wrapper col2{{"This", "is", "not", "a", "string", "type"},
                                          {1, 1, 1, 1, 1, 0}};
  cudf::table_view source_table{{col1, col2}};

  cudf::test::fixed_width_column_wrapper<int16_t> gather_map{{0, 1, 3, 4}};

  cudf::test::fixed_width_column_wrapper<int16_t> exp_col1{{1, 2, 4, 5}, {1, 1, 1, 0}};
  cudf::test::strings_column_wrapper exp_col2{{"This", "is", "a", "string"}, {1, 1, 1, 1}};
  cudf::table_view expected{{exp_col1, exp_col2}};

  auto got = cudf::gather(source_table, gather_map);

  cudf::test::expect_tables_equal(expected, got->view());
}

TEST_F(GatherTestStr, Gather)
{
  std::vector<const char*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{4, 1, 5, 2, 7};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end());
  auto results = cudf::detail::gather(source_table,
                                      gather_map,
                                      cudf::detail::out_of_bounds_policy::IGNORE,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED);

  std::vector<const char*> h_expected;
  std::vector<int32_t> expected_validity;
  for (auto itr = h_map.begin(); itr != h_map.end(); ++itr) {
    auto index = *itr;
    if ((0 <= index) && (index < static_cast<decltype(index)>(h_strings.size()))) {
      h_expected.push_back(h_strings[index]);
      expected_validity.push_back(1);
    } else {
      h_expected.push_back("");
      expected_validity.push_back(0);
    }
  }
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), expected_validity.begin());
  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TEST_F(GatherTestStr, GatherIgnoreOutOfBounds)
{
  std::vector<const char*> h_strings{"eee", "bb", "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::table_view source_table({strings});

  std::vector<int32_t> h_map{3, 4, 0, 0};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(h_map.begin(), h_map.end());
  auto results = cudf::detail::gather(source_table,
                                      gather_map,
                                      cudf::detail::out_of_bounds_policy::IGNORE,
                                      cudf::detail::negative_index_policy::NOT_ALLOWED);

  std::vector<const char*> h_expected;
  std::vector<int32_t> expected_validity;
  for (auto itr = h_map.begin(); itr != h_map.end(); ++itr) {
    h_expected.push_back(h_strings[*itr]);
    expected_validity.push_back(1);
  }
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), expected_validity.begin());
  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TEST_F(GatherTestStr, GatherZeroSizeStringsColumn)
{
  cudf::column_view zero_size_strings_column(
    cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0);
  rmm::device_vector<cudf::size_type> gather_map{};
  auto results = cudf::detail::gather(
    cudf::table_view({zero_size_strings_column}), gather_map.begin(), gather_map.end(), true);
  cudf::test::expect_strings_empty(results->get_column(0).view());
}

template <typename T>
class GatherTestList : public cudf::test::BaseFixture {
};
using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_CASE(GatherTestList, FixedWidthTypesNotBool);

TYPED_TEST(GatherTestList, Gather)
{
  using T = TypeParam;

  // List<T>
  cudf::test::lists_column_wrapper<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

  cudf::table_view source_table({list});
  auto results = cudf::gather(source_table, gather_map);

  cudf::test::lists_column_wrapper<T> expected{{1, 2, 3, 4}, {6, 7}};

  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TYPED_TEST(GatherTestList, GatherNulls)
{
  using T = TypeParam;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<T>
  cudf::test::lists_column_wrapper<T> list{
    {{1, 2, 3, 4}, valids}, {5}, {{6, 7}, valids}, {{8, 9, 10}, valids}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

  cudf::table_view source_table({list});
  auto results = cudf::gather(source_table, gather_map);

  cudf::test::lists_column_wrapper<T> expected{{{1, 2, 3, 4}, valids}, {{6, 7}, valids}};

  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TYPED_TEST(GatherTestList, GatherNested)
{
  using T   = TypeParam;
  using LCW = cudf::test::lists_column_wrapper<T>;

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}},
                                             {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{2, 3}, {4, 5}}, {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }

  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
      {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
      {{LCW{0}}},
      {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
       {{0, 1, 3}, {5}},
       {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      {{{10, 20}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{1, 2, 4};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
      {{LCW{0}}},
      {{{10, 20}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestList, GatherNestedForceRecycle)
{
  using T   = TypeParam;
  using LCW = cudf::test::lists_column_wrapper<T>;

  // these cases force the temporary memory-recycling behavior internal
  // to the gather() recursion

  // recycled on both levels
  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{LCW{2}}}, {{LCW{3}}}, {{LCW{5}}}, {{LCW{6}}}, {{LCW{7}}}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 1, 2};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{{{LCW{2}}}, {{LCW{3}}}, {{LCW{5}}}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }

  // recycled on first level but not second
  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{LCW{2}}}, {{LCW{3}, LCW{4}}}, {{LCW{5}}}, {{LCW{6}}}, {{LCW{7}}}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 1, 2};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{{{LCW{2}}}, {{LCW{3}, LCW{4}}}, {{LCW{5}}}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }

  // recycled on both levels
  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{LCW{2}}}, {{LCW{}}}, {{LCW{5}}}, {{LCW{6}}}, {{LCW{7}}}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 1, 2};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{{{LCW{2}}}, {{LCW{}}}, {{LCW{5}}}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestList, GatherOutOfOrder)
{
  using T   = TypeParam;
  using LCW = cudf::test::lists_column_wrapper<T>;

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}},
                                             {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{1, 2, 0};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                                 {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
                                                 {{2, 3}, {4, 5}}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestList, GatherNestedNulls)
{
  using T   = TypeParam;
  using LCW = cudf::test::lists_column_wrapper<T>;

  auto valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{{2, 3}, valids}, {4, 5}},
      {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
      {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
      {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 1, 3};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{{2, 3}, valids}, {4, 5}},
      {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
      {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }

  // List<List<List<T>>>
  {
    cudf::test::lists_column_wrapper<T> list{
      {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
      {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
      {{LCW{0}}},
      {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
       {{0, 1, 3}, {5}},
       {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      {{{{{10, 20}, valids}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}, valids}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{1, 2, 4};

    cudf::table_view source_table({list});
    auto results = cudf::gather(source_table, gather_map);

    cudf::test::lists_column_wrapper<T> expected{
      {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
      {{LCW{0}}},
      {{{{{10, 20}, valids}}, {LCW{30}}, {{40, 50}, {60, 70, 80}}}, valids}};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }
}

TYPED_TEST(GatherTestList, GatherNestedWithEmpties)
{
  using T   = TypeParam;
  using LCW = cudf::test::lists_column_wrapper<T>;

  cudf::test::lists_column_wrapper<T> list{
    {{2, 3}, LCW{}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, {LCW{}}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{0, 2};

  cudf::table_view source_table({list});
  auto results = cudf::gather(source_table, gather_map);

  cudf::test::lists_column_wrapper<T> expected{{{2, 3}, LCW{}}, {LCW{}}};

  cudf::test::expect_columns_equal(results->view().column(0), expected);
}

TYPED_TEST(GatherTestList, GatherDetailInvalidIndex)
{
  using T   = TypeParam;
  using LCW = cudf::test::lists_column_wrapper<T>;

  // List<List<T>>
  {
    cudf::test::lists_column_wrapper<T> list{{{2, 3}, {4, 5}},
                                             {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{0, 15, 16, 2};

    cudf::table_view source_table({list});
    auto results = cudf::detail::gather(source_table,
                                        gather_map,
                                        cudf::detail::out_of_bounds_policy::IGNORE,
                                        cudf::detail::negative_index_policy::NOT_ALLOWED);

    std::vector<int32_t> expected_validity{1, 0, 0, 1};
    cudf::test::lists_column_wrapper<T> expected{
      {{{2, 3}, {4, 5}}, {LCW{}}, {LCW{}}, {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
      expected_validity.begin()};

    cudf::test::expect_columns_equal(results->view().column(0), expected);
  }
}