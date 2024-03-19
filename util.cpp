#include <filesystem>
#include <opencv2/opencv.hpp>

// Take photos from camera for training
int main() {
  std::cout << "Press <spacebar> to take photos\nESC or q/Q for quitting\n";
  namespace fs = std::filesystem;
  fs::path captured_folder = "./captured";
  if (!fs::exists(captured_folder)) fs::create_directory(captured_folder);
  std::string save_path = captured_folder.string() + '/';
  int n = 0;
  cv::VideoCapture vc(0);
  while (true) {
    cv::Mat frame;
    vc >> frame;
    cv::imshow("Camera - [Press <spacebar> to take a photo]", frame);
    int key = cv::waitKey(7);
    if (key == 32) {
      std::cout << "Photo taken\n";
      cv::imwrite(save_path + "cam_img_" + std::to_string(++n) + ".png", frame);
    } else if (key == 27 || key == 'q') {
      return 0;
    }
  }
  system("pause");
  return 0;
}