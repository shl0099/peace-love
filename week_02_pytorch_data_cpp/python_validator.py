import math
import sys


def validate_csv(path):
    try:
        file = open(path, "r", encoding="utf-8")
    except OSError:
        print("failed to open file")
        return 1

    with file:
        for line_count, line in enumerate(file, start=1):

            # 1. 检查空行
            if not line.strip():
                print(f"empty line at line {line_count}")
                return 3

            # 2. 按逗号拆分字段
            fields = line.rstrip("\r\n").split(",")

            # 3. 检查列数
            if len(fields) != 3:
                print(
                    f"invalid column count at line {line_count}: "
                    f"expected 3, got {len(fields)}"
                )
                return 2

            # 第1行是表头
            if line_count == 1:
                continue

            # 4. 检查数值是否合法
            for column, field in enumerate(fields, start=1):
                try:
                    value = float(field)
                except ValueError:
                    print(
                        f"invalid numeric value at line "
                        f"{line_count}, column {column}"
                    )
                    return 5

                if not math.isfinite(value):
                    print(
                        f"non-finite value at line "
                        f"{line_count}, column {column}"
                    )
                    return 4

    print("CSV valid")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python_validator.py <csv_path>")
        sys.exit(1)

    exit_code = validate_csv(sys.argv[1])
    sys.exit(exit_code)
