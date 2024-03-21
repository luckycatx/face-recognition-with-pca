#include <filesystem>
#include <opencv2/opencv.hpp>

#include "face_preprocessor.hpp"
#include "file_parser.hpp"

cv::Mat ProjectImage(const cv::Mat &img, const cv::Mat &mean,
                     const cv::Mat &eigens) {
  FacePreprocessor fp;
  auto [width, height] = fp.GetStandardSize();
  cv::Mat img_proj = img.reshape(0, width * height);
  img_proj.convertTo(img_proj, CV_64F);
  img_proj -= mean;
  return eigens.t() * img_proj;
}

// Recognize using NCC (Nearest Centroid Classifier)
bool Recognize(std::string_view path, const cv::Mat &mean,
               const cv::Mat &eigens, std::string_view dataset_path) {
  namespace fs = std::filesystem;
  fs::path file_path = path;
  std::string subject_name = file_path.parent_path().filename().string();
  std::string filename = file_path.stem().string();
  std::cout << std::format("Start recognizing ->{}_{}\n", subject_name,
                           filename);

  cv::Mat target = cv::imread(file_path.string());

  // Result image (in grayscale)
  FacePreprocessor fp;
  auto [width, height] = fp.GetStandardSize();
  cv::Mat res = cv::Mat::zeros(height + 8, width * 3 + 16, CV_8UC1);

  // Display source image
  cv::Mat src;
  cv::cvtColor(target, src, cv::COLOR_BGR2GRAY);
  cv::resize(src, src, cv::Size(width, height));
  src.copyTo(res(cv::Rect(4, 4, width, height)));

  // Project the target into face space
  fp.Process(target);
  cv::Mat coord = ProjectImage(target, mean, eigens);

  // Display reconsturcted face
  cv::Mat reconstruction = eigens * coord + mean;
  cv::normalize(reconstruction, reconstruction, 0, 255, cv::NORM_MINMAX,
                CV_8UC1);
  reconstruction.reshape(0, height).copyTo(
      res(cv::Rect(width + 8, 4, width, height)));

  // Get dataset information
  auto dataset_parse_res = file_parser::ParseDataset(dataset_path);
  if (!dataset_parse_res) return false;
  [[maybe_unused]] auto [_, img_map, names] = *dataset_parse_res;

  std::string match_name = *names.begin();
  cv::Mat match_img = img_map[match_name][0];

  // NCC Calculating distances
  double min_d =
      cv::norm(ProjectImage(match_img, mean, eigens), coord, cv::NORM_L2);
  for (std::string name : names) {
    for (cv::Mat img : img_map[name]) {
      double d = cv::norm(ProjectImage(img, mean, eigens), coord, cv::NORM_L2);
      if (d < min_d) {
        min_d = d;
        match_img = img;
        match_name = name;
      }
    }
  }

  // Display match result
  match_img.copyTo(res(cv::Rect(width * 2 + 12, 4, width, height)));
  cv::putText(res, "Matched: " + match_name,
              cv::Point2i(width * 0.17, height * 0.17),
              cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar(255.0));
  cv::imshow("[Target Face] >> [Reconstructed Face] >> [Matched Face]", res);
  cv::waitKey();

  // Save results
  fs::path res_folder = "./results";
  if (!fs::exists(res_folder)) fs::create_directory(res_folder);
  std::string save_path =
      res_folder.string() + '/' + subject_name + '_' + filename + "_result.png";
  cv::imwrite(save_path, res);

  bool ret;
  while (true) {
    int key = cv::waitKey();
    if (key == 'y' || key == 'Y') {
      ret = true;
      break;
    } else if (key == 'n' || key == 'N') {
      ret = false;
      break;
    }
  }
  cv::destroyAllWindows();
  std::cout << std::format("Recognition of {}_{} has done\n", subject_name,
                           filename);
  return ret;
}

void FileRecognition(std::string_view test_path, std::string_view eigens_path,
                     std::string_view dataset_path) {
  namespace fs = std::filesystem;
  if (!fs::exists(test_path)) {
    std::cout << "Test directory does not exist" << std::endl;
    return;
  }

  // Get mean and eigenfaces from file
  auto eigens_parse_res = file_parser::ParseEigens(eigens_path);
  if (!eigens_parse_res) return;
  auto [mean, eigens] = *eigens_parse_res;

  // Start recognizing
  std::cout << "y/Y for correct matching, n/N for wrong matching\n\n";
  int n = 0, hit = 0;
  for (const auto &entry : fs::directory_iterator(test_path)) {
    for (const auto &sub_entry : fs::directory_iterator(entry)) {
      if (fs::is_directory(sub_entry.path())) continue;
      std::string file_path = sub_entry.path().string();
      bool res = Recognize(file_path, mean, eigens, dataset_path);
      ++n;
      if (res) {
        std::cout << "[Y] Correct recognition result\n\n";
        ++hit;
      } else {
        std::cout << "[X] Wrong recognition result\n\n";
      }
    }
  }
  std::cout << std::format(
      "A total of {} images are tested, {} recognitions are hit\n"
      "Hit rate: {}%\n",
      n, hit, static_cast<float>(hit) / n * 100.f);
}

void CamRecognition(std::string_view eigens_path,
                    std::string_view dataset_path) {
  std::cout << "Press <spacebar> to take a photo and start recognize\n";
  namespace fs = std::filesystem;
  fs::path captured_folder = "./captured";
  if (!fs::exists(captured_folder)) fs::create_directory(captured_folder);
  std::string save_path = captured_folder.string() + "/cam_img_test.png";

  cv::VideoCapture vc(0);
  while (true) {
    cv::Mat frame;
    vc >> frame;
    cv::imshow("Camera - [Press <spacebar> to take a photo]", frame);
    if (cv::waitKey(7) == 32) {
      std::cout << "Photo taken\n\n";
      cv::imwrite(save_path, frame);
      cv::waitKey(700);
      cv::destroyAllWindows();
      break;
    }
  }

  // Get mean and eigenfaces from file
  auto eigens_parse_res = file_parser::ParseEigens(eigens_path);
  if (!eigens_parse_res) return;
  auto [mean, eigens] = *eigens_parse_res;

  // Start recognizing
  std::cout << "y/Y for correct matching, n/N for wrong matching\n\n";
  bool res = Recognize(save_path, mean, eigens, dataset_path);
  if (res)
    std::cout << "[Y] Correct recognition result\n";
  else
    std::cout << "[X] Wrong recognition result\n";
}

int main(int argc, char *argv[]) {
  // std::string_view test_path = "./dataset/test";
  // std::string_view eigens_path = "./eigens";
  // std::string_view dataset_path = "./dataset/train";
  if (argc == 4)
    FileRecognition(argv[1], argv[2], argv[3]);
  else if (argc == 3)
    CamRecognition(argv[1], argv[2]);
  else
    std::cout << "Use by: >./recognize <test_path> <eigens_path> <dataset_path>"
                 "(test by files)\n"
                 "or\n>recognize <eigens_path> <dataset_path> "
                 "(test by camera capture)"
              << std::endl;
  system("pause");
  return 0;
}