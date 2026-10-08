#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr
            << "usage: csv_validator <csv_path>"
            << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);

    if (!file.is_open()) {
        std::cerr
            << "failed to open file"
            << std::endl;
        return 1;
    }

    std::string line;
    int line_count = 0;

    while (std::getline(file, line)) {
        ++line_count;

        // 1. 检查空行
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            std::cerr
                << "empty line at line "
                << line_count
                << std::endl;

            return 3;
        }

        // 2. 按逗号拆字段
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string field;

        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }

        // 如果原始行以逗号结束，补上最后一个空字段
        if (!line.empty() && line.back() == ',') {
            fields.push_back("");
        }

        // 3. 检查列数
        if (fields.size() != 3) {
            std::cerr
                << "invalid column count at line "
                << line_count
                << ": expected 3, got "
                << fields.size()
                << std::endl;

            return 2;
        }

        // 第1行是表头，不做数值检查
        if (line_count == 1) {
            continue;
        }

        // 4. 检查每个字段是不是有限数值
        for (std::size_t i = 0; i < fields.size(); ++i) {
            try {
                std::size_t pos = 0;

                double value = std::stod(fields[i], &pos);

                // 新增：检查整个字符串是否都被成功解析
                if (pos != fields[i].size()) {
                    std::cerr
                        << "invalid numeric value at line "
                        << line_count
                        << ", column "
                        << (i + 1)
                        << std::endl;

                    return 5;
                }

                // 原有：检查 NaN / inf
                if (!std::isfinite(value)) {
                    std::cerr
                        << "non-finite value at line "
                        << line_count
                        << ", column "
                        << (i + 1)
                        << std::endl;

                    return 4;
                }
            }
            catch (...) {
                std::cerr
                    << "invalid numeric value at line "
                    << line_count
                    << ", column "
                    << (i + 1)
                    << std::endl;

                return 5;
            }
        }
    }

    std::cout << "CSV valid" << std::endl;

    return 0;
}