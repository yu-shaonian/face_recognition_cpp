#ifndef UTILS_H
#define UTILS_H

#include "ncnn_centerface.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


cv::Mat align(cv::Mat& image, FaceInfo& face);
std::vector<char> get_the_bytes(std::string filename) ;

#endif