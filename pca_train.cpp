#include <opencv2/opencv.hpp>

#include "face_preprocessor.hpp"
#include "file_parser.hpp"

auto PCA(std::vector<cv::Mat> &img_arr) {
  std::cout << "Start training...\n";

  // Image size
  int width = img_arr[0].cols;
  int height = img_arr[0].rows;

  // Convert to 64-bit floating number
  for (auto &img : img_arr) img.convertTo(img, CV_64F);

  // Face average
  cv::Mat mean = cv::Mat::zeros(width * height, 1, CV_64F);

  // Convert images to column vectors and sum up
  for (auto &img : img_arr) {
    img = img.reshape(0, width * height);
    mean += img;
  }
  mean /= static_cast<double>(img_arr.size());

  // X matrix i.e. difference matrix of each face from the average
  cv::Mat x = cv::Mat::zeros(width * height, img_arr.size(), CV_64F);
  for (size_t i = 0; i < img_arr.size(); ++i) x.col(i) = img_arr[i] - mean;

  // Calculate covariance by 1 / n - 1  * X.T * X
  /* Invert the transpose position to reduce the number of calculations and will
   * not affect eigenfaces */
  cv::Mat cov = x.t() * x;
  cov /= static_cast<double>(img_arr.size() - 1);

  //  Calculate eigenvalues and eigenvectors
  cv::Mat eigen_vals, eigen_vecs;
  cv::eigen(cov, eigen_vals, eigen_vecs);

  // Get eigenfaces and normalize
  cv::Mat eigen_faces = x * eigen_vecs;
  cv::Mat top_faces = cv::Mat::zeros(height * 3 + 16, width * 3 + 16, CV_8UC1);
  for (int i = 0; i < eigen_faces.cols; ++i) {
    cv::Mat eigen_face = eigen_faces.col(i);
    cv::normalize(eigen_face, eigen_face, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    eigen_face.copyTo(eigen_faces.col(i));
    if (i < 9)
      eigen_face.reshape(0, height).copyTo(top_faces(
          cv::Rect((i % 3) * width + (i % 3 + 1) * 4,
                   (2 - i / 3) * height + (3 - i / 3) * 4, width, height)));
  }

  // Get mean face
  cv::Mat mean_face = cv::Mat::zeros(height, width, CV_8UC1);
  cv::normalize(mean, mean, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  mean.reshape(0, height).copyTo(mean_face(cv::Rect(0, 0, width, height)));

  // Display training results
  cv::imshow("Average face", mean_face);
  cv::imshow("Top eigenfaces", top_faces);
  cv::waitKey(1700);

  return std::make_pair(mean, eigen_faces);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Use by: >./pca_train <dataset_path>" << std::endl;
    return 1;
  };

  namespace fs = std::filesystem;
  // std::string_view dataset_path = "./dataset/train";
  auto dataset_parse_res = file_parser::ParseDataset(argv[1]);
  if (!dataset_parse_res) {
    system("pause");
    return 1;
  }
  auto [img_set, img_map, names] = *dataset_parse_res;

  // Start PCA training
  auto [mean, eigen_faces] = PCA(img_set);

  // Save results
  std::cout << "Eigenfaces saved" << std::endl;
  cv::FileStorage cv_fs("eigens", cv::FileStorage::WRITE);
  cv_fs << "label"
        << "eigens";
  cv_fs << "mean" << mean;
  cv_fs << "face" << eigen_faces;
  cv_fs.release();

  system("pause");
  return 0;
}
