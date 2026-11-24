#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "types.hpp"

// 画一块装甲板：4 条边 + 4 个彩色角点 + 中心十字。predicted=true 时画虚线并用橙色。
void drawPlate(cv::Mat& img, const cv::Point2f corners[4], bool predicted);

// 左上角 HUD 文本框，逐行写字。
void drawHud(cv::Mat& img, const std::vector<std::string>& lines);

// 调试视图：灯条框 + 候选装甲板 + 掩膜小窗。
void drawDebug(cv::Mat& img, const cv::Mat& mask, const std::vector<LightBar>& bars,
               const std::vector<Armor>& armors);
