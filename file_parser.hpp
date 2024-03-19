#include <filesystem>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "face_preprocessor.hpp"

namespace file_parser {

auto ParseDataset(std::string_view dataset_path) {
  std::vector<cv::Mat> img_set;
  std::map<std::string, std::vector<cv::Mat>> img_map;
  std::unordered_set<std::string> names;

  namespace fs = std::filesystem;
  if (!fs::exists(dataset_path)) {
    std::cout << "Dataset directory does not exist" << std::endl;
    return std::make_optional<decltype(std::make_tuple(img_set, img_map,
                                                       names))>();
  }

  FacePreprocessor fp;
  for (const auto &entry : fs::directory_iterator(dataset_path)) {
    std::string subject_name = entry.path().filename().string();
    if (names.find(subject_name) == names.end()) {
      names.insert(subject_name);
      img_map.insert(std::make_pair(subject_name, std::vector<cv::Mat>()));
    }
    for (const auto &sub_entry : fs::directory_iterator(entry)) {
      if (fs::is_directory(sub_entry.path())) continue;
      std::string img_path = sub_entry.path().string();
      cv::Mat img = cv::imread(img_path);

      // Dectect and preprocessor the image
      fp.Process(img);
      img_map[subject_name].emplace_back(std::move(img));
    }
    // std::cout << std::format("{} is done preprocessed\n", subject_name);
  }

  // Create dataset
  for (std::string name : names)
    img_set.insert(img_set.end(), img_map[name].begin(), img_map[name].end());

  return std::make_optional(std::make_tuple(img_set, img_map, names));
}

auto ParseEigens(std::string_view eigens_path) {
  cv::Mat mean, eigens;

  namespace fs = std::filesystem;
  if (!fs::exists(eigens_path)) {
    std::cout << "Eigenfaces path does not exist" << std::endl;
    return std::make_optional<decltype(std::make_pair(mean, eigens))>();
  }

  cv::FileStorage cv_fs(eigens_path.data(), cv::FileStorage::READ);
  std::string label;
  cv_fs["label"] >> label;
  if (label != "eigens") {
    std::cout << "Wrong eigens file" << std::endl;
    return std::make_optional<decltype(std::make_pair(mean, eigens))>();
  }

  cv_fs["mean"] >> mean;
  cv_fs["face"] >> eigens;

  // Convert to 64-bit floating number
  mean.convertTo(mean, CV_64F);
  eigens.convertTo(eigens, CV_64F);

  return std::make_optional(std::make_pair(mean, eigens));
}

}  // namespace file_parser