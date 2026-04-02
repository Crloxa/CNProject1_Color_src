#pragma once

#include "../include/opencv2/opencv.hpp"

namespace ImgParse {

    bool Main(const cv::Mat& srcImg, cv::Mat& disImg);
    bool processV5Enhanced(const cv::Mat& srcImg, cv::Mat& disImg);

} // namespace ImgParse