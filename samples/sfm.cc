#include <string>
#include <opencv2/opencv.hpp>
#include "System.h"

int main(int argc, const char* argv[]) {
    const std::string argKeys =
        "{imgDir    | path/to/your/SfM/data | dataset path}"
        "{workDir   | path/to/your/SfM/data/sfm_work | work directory}"
        "{config    | path/to/your/SfM/data/Config.yaml | config file}";

    cv::CommandLineParser parser(argc, argv, argKeys);
    const std::string imgDir = parser.get<std::string>("imgDir");
    const std::string workDir = parser.get<std::string>("workDir");
    const std::string configFile = parser.get<std::string>("config");
    sfm::System system(configFile, workDir);
    system.RunSFM(imgDir);
}