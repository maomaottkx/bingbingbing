#include "color.hpp"

#include <opencv2/imgproc.hpp>
#include <vector>

#include "config.hpp"

cv::Mat buildRedMask(const cv::Mat& bgr, int mode)
{
  // 1. HSV 红色两段
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

  cv::Mat low, high;
  cv::inRange(hsv, cv::Scalar(cfg::kRedH1Low, cfg::g_tune.minS, cfg::g_tune.minV),
              cv::Scalar(cfg::kRedH1High, 255, 255), low);
  cv::inRange(hsv, cv::Scalar(cfg::kRedH2Low, cfg::g_tune.minS, cfg::g_tune.minV),
              cv::Scalar(cfg::kRedH2High, 255, 255), high);
  cv::Mat hueMask = low | high;

  // 2. 通道差 R - max(G,B)：红色灯条过曝发白时 HSV 饱和度会掉，靠这一路兜住
  //    ch[0]=B ch[1]=G ch[2]=R
  std::vector<cv::Mat> ch;
  cv::split(bgr, ch);

  cv::Mat gb = cv::max(ch[1], ch[0]);
  cv::Mat diffMat;
  cv::subtract(ch[2], gb, diffMat);  // uchar 饱和运算，负数截到 0

  cv::Mat diffMask;
  cv::threshold(diffMat, diffMask, cfg::g_tune.redDiff, 255, cv::THRESH_BINARY);

  cv::Mat mask = (mode == 1) ? (hueMask & diffMask) : (hueMask | diffMask);
  if (mode == 2) mask = diffMask;  // 只用通道差，调试用

  if (std::getenv("AIM_DEBUG")) {
    std::printf("  [mask] hue=%d diff=%d out=%d\n", cv::countNonZero(hueMask),
                cv::countNonZero(diffMask), cv::countNonZero(mask));
  }

  // 3. 形态学：开运算去噪点，闭运算把断开的灯条连起来
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 9)));
  return mask;
}
