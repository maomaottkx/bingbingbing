#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "types.hpp"

struct DetectResult {
  cv::Mat mask;                // 二值掩膜，调试用
  std::vector<LightBar> bars;  // 通过几何与颜色筛选的灯条
  std::vector<Armor> armors;   // 粗筛后的装甲板候选，按 score 升序
  int contours = 0;            // 原始轮廓数
};

// 一帧完整检测：掩膜 -> 轮廓 -> 灯条筛选 -> 两两配对粗筛
DetectResult detect(const cv::Mat& bgr, int maskMode);
