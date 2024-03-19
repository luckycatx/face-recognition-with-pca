#ifndef FACE_DETECTOR_HPP_
#define FACE_DETECTOR_HPP_

#include <opencv2/opencv.hpp>

class FacePreprocessor {
 public:
  FacePreprocessor() : StandardSize(150, 175) {
    std::string cascade_path = "./cascade";
    std::string face_cascade_file = "/haarcascade_frontalface_default.xml";
    std::string eye_cascade_file = "/haarcascade_eye.xml";
    if (!face_clf.load(cascade_path + face_cascade_file) ||
        !eye_clf.load(cascade_path + eye_cascade_file)) {
      std::cout << "Load haar cascade failed" << std::endl;
      return;
    }
  }

  void Process(cv::Mat &img) {
    // Convert image to grayscale
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // Enhance contrast by histogram equalization
    cv::equalizeHist(img, img);

    // Size check
    if (img.size().width < StandardSize.width &&
        img.size().height < StandardSize.height) {
      // std::cout << "Smaller than standard size, scale and skip" << std::endl;
      cv::resize(img, img, StandardSize);
      return;
    }

    // Detect face and eyes by haar cascade classifier
    std::vector<cv::Rect> face_rect;
    face_clf.detectMultiScale(img, face_rect, 1.1, 3, 0, cv::Size(50, 50));
    // for (auto rec : face_rect)
    //   cv::rectangle(img, rec, cv::Scalar(117, 127, 227), 3);
    // cv::imshow("Face detection", img);
    // cv::waitKey(700);

    // Trim the rectangle for cropping
    if (face_rect.size() > 0) {
      cv::Rect &rec = face_rect[0];

      // Expand face area
      rec.x -= rec.width * .125f;
      rec.y -= rec.height * .125f;
      rec.width *= 1.25f;
      rec.height *= 1.25f;

      // Boundary check
      if (rec.x < 0) rec.x = 0;
      if (rec.y < 0) rec.y = 0;
      if (rec.x + rec.width > img.cols) rec.width = img.cols - rec.x;
      if (rec.y + rec.height > img.rows) rec.height = img.rows - rec.y;

      // Crop face area
      img = img(rec);
      // cv::imshow("After cropping", img);
      // cv::waitKey(700);
    } else {
      // std::cout << std::format("No face is detected, scale and skip\n");
      cv::resize(img, img, StandardSize);
      return;
    }

    std::vector<cv::Rect> eye_rect;
    eye_clf.detectMultiScale(img, eye_rect, 1.075, 3, 0, cv::Size(17, 15));
    // for (auto rec : eye_rect)
    //   cv::rectangle(img, rec, cv::Scalar(217, 227, 127), 3);
    // cv::imshow("Eye detection", img);
    // cv::waitKey(700);

    if (eye_rect.size() == 2) {
      cv::Point2f l_eye((float)eye_rect[0].x + eye_rect[0].width / 2.f,
                        (float)eye_rect[0].y + eye_rect[0].height / 2.f);
      cv::Point2f r_eye((float)eye_rect[1].x + eye_rect[1].width / 2.f,
                        (float)eye_rect[1].y + eye_rect[1].height / 2.f);
      if (l_eye.x > r_eye.x) std::swap(l_eye, r_eye);
      cv::Point2f mid = (l_eye + r_eye) / 2.f;
      float slope = (r_eye.y - l_eye.y) / (r_eye.x - l_eye.x);
      float rad = atan(slope);
      float pi = acos(-1.f);
      cv::Mat rot = cv::getRotationMatrix2D(mid, rad / pi * 180.f, 1);

      // Rotate the image
      cv::warpAffine(img, img, rot, img.size());
      // cv::imshow("After rotation", img);
      // cv::waitKey(700);
    } else {
      // std::cout << "Eyes detection is biased, no rotation are performed"
      //           << std::endl;
    }

    // Scale image to standard size
    cv::resize(img, img, StandardSize);
    // cv::imshow("After scale", img);
    // cv::waitKey(700);
  }

  cv::Size GetStandardSize() { return StandardSize; }

 private:
  const cv::Size StandardSize;
  cv::CascadeClassifier face_clf, eye_clf;
};

#endif