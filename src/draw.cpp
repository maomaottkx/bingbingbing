#include "draw.hpp"

#include <algorithm>
#include <cstdio>

#include <opencv2/imgproc.hpp>

namespace {

void dashedLine(cv::Mat& img, cv::Point2f a, cv::Point2f b, const cv::Scalar& color, int thick,
                int dash = 14, int gap = 10)
{
  const double total = cv::norm(b - a);
  if (total < 1.0) return;
  const cv::Point2f dir = (b - a) / static_cast<float>(total);
  for (double s = 0; s < total; s += dash + gap) {
    const double e = std::min(s + dash, total);
    cv::line(img, a + dir * static_cast<float>(s), a + dir * static_cast<float>(e), color, thick,
             cv::LINE_AA);
  }
}

}  // namespace

void drawPlate(cv::Mat& img, const cv::Point2f corners[4], bool predicted)
{
  const cv::Scalar edge = predicted ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f a = corners[i];
    const cv::Point2f b = corners[(i + 1) % 4];
    if (predicted) {
      dashedLine(img, a, b, edge, 2);
    } else {
      cv::line(img, a, b, edge, 1, cv::LINE_AA);
    }
  }
  const cv::Scalar cornerColor(0, 255, 0);
  const char* const labels[] = {"p1", "p4", "p3", "p2"};
  const cv::Point offsets[] = {{-28, -8}, {8, -8}, {8, 20}, {-28, 20}};
  constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double fontScale = 0.5;
  int baseline = 0;
  const cv::Size labelSize = cv::getTextSize(labels[0], font, fontScale, 3, &baseline);
  for (int i = 0; i < 4; ++i) {
    cv::circle(img, corners[i], 5, cornerColor, -1, cv::LINE_AA);
    cv::Point origin = cv::Point(corners[i]) + offsets[i];
    origin.x = std::clamp(origin.x, 0, std::max(0, img.cols - labelSize.width));
    origin.y = std::clamp(origin.y, labelSize.height, std::max(labelSize.height, img.rows - baseline));
    cv::putText(img, labels[i], origin, font, fontScale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, labels[i], origin, font, fontScale, cornerColor, 1, cv::LINE_AA);
  }

  // 中心点：白色十字
  const cv::Point2f c = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
  cv::line(img, cv::Point2f(c.x - 12, c.y), cv::Point2f(c.x + 12, c.y), cv::Scalar(255, 255, 255),
           2, cv::LINE_AA);
  cv::line(img, cv::Point2f(c.x, c.y - 12), cv::Point2f(c.x, c.y + 12), cv::Scalar(255, 255, 255),
           2, cv::LINE_AA);
  cv::circle(img, c, 2, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
}

void drawHud(cv::Mat& img, const std::vector<std::string>& lines)
{
  if (lines.empty()) return;

  const int pad = 10;
  const double fontScale = 0.62;
  const int lineH = static_cast<int>(26 * fontScale / 0.62) + 8;
  int width = 0;
  for (const auto& s : lines) {
    int base = 0;
    const int w = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &base).width;
    width = std::max(width, w);
  }

  const cv::Rect box(pad, pad, width + pad * 2, lineH * static_cast<int>(lines.size()) + pad);
  cv::Mat roi = img(box & cv::Rect(0, 0, img.cols, img.rows));
  roi *= 0.25;

  int y = pad + lineH - 8;
  for (const auto& s : lines) {
    const cv::Scalar color = (s.find("PREDICT") != std::string::npos) ? cv::Scalar(0, 165, 255)
                                                                     : cv::Scalar(0, 255, 255);
    cv::putText(img, s, cv::Point(pad + pad / 2, y), cv::FONT_HERSHEY_SIMPLEX, fontScale, color, 1,
                cv::LINE_AA);
    y += lineH;
  }
}

void drawDebug(cv::Mat& img, const cv::Mat& mask, const std::vector<LightBar>& bars,
               const std::vector<Armor>& armors)
{
  // 灯条：黄色框
  for (const auto& b : bars) {
    cv::Point2f pts[4];
    b.rect.points(pts);
    for (int i = 0; i < 4; ++i) {
      cv::line(img, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
  }

  // 候选装甲板：品红细框
  for (const auto& a : armors) {
    for (int i = 0; i < 4; ++i) {
      cv::line(img, a.corners[i], a.corners[(i + 1) % 4], cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
    }
  }

  // 右下角缩略掩膜
  if (!mask.empty()) {
    const int tw = img.cols / 5;
    const int th = mask.rows * tw / mask.cols;
    cv::Mat small;
    cv::resize(mask, small, cv::Size(tw, th));
    cv::Mat smallBgr;
    cv::cvtColor(small, smallBgr, cv::COLOR_GRAY2BGR);
    const cv::Rect dst(img.cols - tw - 12, img.rows - th - 12, tw, th);
    smallBgr.copyTo(img(dst));
    cv::rectangle(img, dst, cv::Scalar(255, 255, 255), 1);
  }
}
