#pragma once

#include <opencv2/core.hpp>

// 生成"红色发光区域"掩膜。
// mode = 0：HSV 红色段 与 通道差 取并集（灯条过曝发白时靠通道差兜住）
// mode = 1：两者取交集（更严，噪点更少）
cv::Mat buildRedMask(const cv::Mat& bgr, int mode);
