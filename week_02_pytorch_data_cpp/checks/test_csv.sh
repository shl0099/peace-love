#!/usr/bin/env bash

cd "$(dirname "$0")/.." || exit 1

names=(
    good
    bad_columns
    bad_blank
    bad_nan
    bad_inf
    bad_suffix
    bad_trailing_comma
)

expected=(0 2 3 4 4 5 2)

passed=0
failed=0

for i in "${!names[@]}"; do
    name="${names[$i]}"
    file="data/${name}.csv"
    expected_code="${expected[$i]}"

    python python_validator.py "$file" >/dev/null 2>&1
    py_code=$?

    ./cpp/build/csv_validator "$file" >/dev/null 2>&1
    cpp_code=$?

    if [ "$py_code" -eq "$expected_code" ] &&
       [ "$cpp_code" -eq "$expected_code" ]; then

        echo "PASS: $name (Python=$py_code, C++=$cpp_code)"
        ((passed+=1))
    else
        echo "FAIL: $name (expected=$expected_code, Python=$py_code, C++=$cpp_code)"
        ((failed+=1))
    fi
done

echo "--------------------"
echo "Passed: $passed"
echo "Failed: $failed"

if [ "$failed" -ne 0 ]; then
    exit 1
fi

exit 0
