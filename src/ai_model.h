#pragma once

#include "cls_image_sharp.hpp"
#include "opencv2/opencv.hpp"

bool InitModel();

int DetectImageClarity(const cv::Mat &img, 
                       int &clarity, 
                       float &confidence);
