#include "detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "color.hpp"
#include "config.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// 设了 AIM_DEBUG=1 环境变量时，把每个轮廓的指标打出来，用来看谁被哪一条卡掉
bool verbose()
{
  static const bool on = std::getenv("AIM_DEBUG") != nullptr;
  return on;
}

// Rectify between the light-bar axes and measure local contrast in the number region.
int boardNumberContrast(const cv::Mat& bgr, const LightBar& left, const LightBar& right)
{
  constexpr int width = 48, height = 32;
  constexpr int maxChroma = 20;
  const cv::Point2f source[] = {left.axisTop, right.axisTop, right.axisBottom, left.axisBottom};
  const cv::Point2f destination[] = {{0, 0}, {width - 1, 0},
                                     {width - 1, height - 1}, {0, height - 1}};
  cv::Mat patch;
  cv::warpPerspective(bgr, patch, cv::getPerspectiveTransform(source, destination),
                      {width, height});
  const cv::Mat center = patch(cv::Rect(width / 4, 2, width / 2, height - 4));
  int bins[256] = {};
  for (int y = 0; y < center.rows; ++y) {
    const auto* row = center.ptr<cv::Vec3b>(y);
    for (int x = 0; x < center.cols; ++x) {
      const auto& pixel = row[x];
      const int darkest = std::min({pixel[0], pixel[1], pixel[2]});
      const int brightest = std::max({pixel[0], pixel[1], pixel[2]});
      ++bins[brightest - darkest <= maxChroma ? darkest : 0];
    }
  }
  const auto percentile = [&](double fraction) {
    const double end = center.total() * fraction;
    size_t count = 0;
    for (int i = 0; i < 256; ++i) {
      count += bins[i];
      if (count >= end) return i;
    }
    return 255;
  };
  const int background = percentile(0.40);
  const int foreground = percentile(0.65);
  return foreground - background;
}

// 取灯条朝外那一条长边的两个端点（leftBar=true 取向左突出的那条）
void outerEdge(const LightBar& bar, bool leftBar, cv::Point2f* top, cv::Point2f* bottom)
{
  // 4 条边里长度为长边的有 2 条，比较它们的中点 x 决定哪条朝外
  int best = -1;
  double bestLen = 0.0;
  for (int i = 0; i < 4; ++i) {
    double l = cv::norm(bar.box[(i + 1) % 4] - bar.box[i]);
    if (l > bestLen) {
      bestLen = l;
    }
  }
  // 先找出两条长边中的第一条
  std::vector<int> longEdges;
  for (int i = 0; i < 4; ++i) {
    double l = cv::norm(bar.box[(i + 1) % 4] - bar.box[i]);
    if (l > bestLen * 0.9) longEdges.push_back(i);
  }
  if (longEdges.size() < 2) longEdges = {0, 2};
  best = longEdges[0];
  if (longEdges.size() >= 2) {
    const int e0 = longEdges[0], e1 = longEdges[1];
    const double x0 = (bar.box[e0].x + bar.box[(e0 + 1) % 4].x) * 0.5;
    const double x1 = (bar.box[e1].x + bar.box[(e1 + 1) % 4].x) * 0.5;
    best = leftBar ? (x0 < x1 ? e0 : e1) : (x0 > x1 ? e0 : e1);
  }

  cv::Point2f a = bar.box[best];
  cv::Point2f b = bar.box[(best + 1) % 4];
  if (a.y <= b.y) {
    *top = a;
    *bottom = b;
  } else {
    *top = b;
    *bottom = a;
  }
}

// 在轮廓上取平均颜色，得到红纯度
void measurePurity(const cv::Mat& bgr, const std::vector<cv::Point>& contour, LightBar& bar)
{
  long long sR = 0, sG = 0, sB = 0;
  for (const auto& p : contour) {
    const cv::Vec3b& px = bgr.at<cv::Vec3b>(p);
    sB += px[0];
    sG += px[1];
    sR += px[2];
  }
  const double n = contour.empty() ? 1.0 : static_cast<double>(contour.size());
  bar.rg = static_cast<float>(sR / n - sG / n);
  bar.rb = static_cast<float>(sR / n - sB / n);
}

void trimLowerRightGlare(const cv::Mat& bgr, const cv::Mat& mask,
                         const std::vector<cv::Point>& contour, cv::RotatedRect& rect)
{
  const float width = std::min(rect.size.width, rect.size.height);
  const float length = std::max(rect.size.width, rect.size.height);
  const cv::Rect roi = cv::boundingRect(contour);
  if (width < 8.f || length < cfg::kMinBarLen || length > 4.f * width ||
      roi.height < cfg::kMinBarLen) return;

  cv::Mat component = cv::Mat::zeros(roi.size(), CV_8U);
  const cv::Point* polygon = contour.data();
  const int vertices = static_cast<int>(contour.size());
  cv::fillPoly(component, &polygon, &vertices, 1, cv::Scalar(255), cv::LINE_8, 0, -roi.tl());
  cv::bitwise_and(component, mask(roi), component);

  std::vector<cv::Point2i> edges;
  edges.reserve(roi.height);
  for (int y = 0; y < roi.height; ++y) {
    const auto* row = component.ptr<unsigned char>(y);
    const auto* left = std::find_if(row, row + roi.width, [](unsigned char v) { return v != 0; });
    const auto* right = std::find_if(std::make_reverse_iterator(row + roi.width),
                                     std::make_reverse_iterator(left),
                                     [](unsigned char v) { return v != 0; }).base();
    edges.emplace_back(static_cast<int>(left - row), static_cast<int>(right - row) - 1);
  }

  // Fit the strip above the glare and estimate its typical width there.
  std::vector<cv::Point2f> centers;
  std::vector<float> widths;
  const cv::Mat pixels = bgr(roi);
  for (int y = roi.height / 8; y < 2 * roi.height / 3; ++y) {
    const auto* row = pixels.ptr<cv::Vec3b>(y);
    int peak = 0;
    for (int x = 0; x < roi.width; ++x) {
      if (row[x][2] >= 200) peak = std::max(peak, static_cast<int>(row[x][1]));
    }
    if (peak < 75) continue;
    int sum = 0, count = 0;
    for (int x = 0; x < roi.width; ++x) {
      if (row[x][2] >= 200 && row[x][1] * 4 >= peak * 3) {
        sum += x;
        ++count;
      }
    }
    if (count == 0) continue;
    const float center = static_cast<float>(sum) / count;
    centers.emplace_back(center, static_cast<float>(y));
    const auto [left, right] = edges[y];
    if (left <= center && center <= right && right - left >= 4)
      widths.push_back(static_cast<float>(right - left));
  }
  if (centers.size() < 6 || widths.size() < 6) return;
  auto median = widths.begin() + widths.size() / 2;
  std::nth_element(widths.begin(), median, widths.end());
  const float halfWidth = *median * 0.5f;
  cv::Vec4f line;
  cv::fitLine(centers, line, cv::DIST_HUBER, 0, 0.01, 0.01);
  if (std::abs(line[1]) < 0.8f) return;
  const float dxdy = line[0] / line[1];
  const auto centerX = [&](int y) { return line[2] + (y - line[3]) * dxdy; };

  const float margin = std::max(4.f, 0.65f * halfWidth);
  const int coreRadius = std::max(1, cvRound(halfWidth * 0.5f));
  int widenedRows = 0;
  int first = roi.height;
  int last = -1;
  for (int y = 0; y < roi.height; ++y) {
    const auto [left, right] = edges[y];
    const float x = centerX(y);
    const int sampleX = cvRound(x);
    const auto* maskRow = component.ptr<unsigned char>(y);
    const auto* pixelRow = pixels.ptr<cv::Vec3b>(y);
    bool hasCore = false;
    for (int col = std::max(0, sampleX - coreRadius);
         col <= std::min(roi.width - 1, sampleX + coreRadius); ++col) {
      if (maskRow[col] || pixelRow[col][2] >= 180) {
        hasCore = true;
        break;
      }
    }
    if (!hasCore) continue;
    first = std::min(first, y);
    last = y;
    if (y >= 3 * roi.height / 5 && right > x + halfWidth + margin &&
        std::abs(left - (x - halfWidth)) <= margin) ++widenedRows;
  }
  if (widenedRows < std::max(3, roi.height / 15) || last - first < 0.8f * roi.height) return;

  const cv::Point2f top(centerX(first) + roi.x, static_cast<float>(first + roi.y));
  const cv::Point2f bottom(centerX(last) + roi.x, static_cast<float>(last + roi.y));
  const float cleanWidth = 2.f * halfWidth / std::sqrt(1.f + dxdy * dxdy);
  if (cleanWidth >= 0.8f * width) return;
  const cv::Point2f direction = bottom - top;
  const float angle = static_cast<float>(std::atan2(direction.y, direction.x) * 180.0 / kPi);
  rect = cv::RotatedRect((top + bottom) * 0.5f,
                         {static_cast<float>(cv::norm(direction)), cleanWidth}, angle);
  if (verbose()) {
    std::printf("  [glare] center=(%.1f, %.1f) width=%.1f -> %.1f, rows=%d\n",
                rect.center.x, rect.center.y, width, cleanWidth, widenedRows);
  }
}

bool setBarGeometry(LightBar& bar)
{
  bar.rect.points(bar.box);
  bar.center = bar.rect.center;

  // 长边 / 短边由 4 条边的实际长度决定，不依赖 RotatedRect 的宽高约定
  double edge[4];
  for (int i = 0; i < 4; ++i) edge[i] = cv::norm(bar.box[(i + 1) % 4] - bar.box[i]);
  const int longest = static_cast<int>(std::max_element(edge, edge + 4) - edge);
  bar.len = static_cast<float>(edge[longest]);
  bar.wid = static_cast<float>(edge[(longest + 1) % 4]);
  if (bar.wid < 1e-3f) return false;
  bar.ratio = bar.len / bar.wid;

  // 长边与竖直方向的夹角
  const cv::Point2f d = bar.box[(longest + 1) % 4] - bar.box[longest];
  const double angDeg = std::atan2(d.y, d.x) * 180.0 / kPi;
  double m = std::fmod(std::fabs(angDeg), 180.0);
  bar.tiltDeg = static_cast<float>(std::fabs(m - 90.0));

  // 长轴中线的两个端点（按 y 排上下）
  const cv::Point2f unit = d / static_cast<float>(bar.len);
  cv::Point2f e0 = bar.center - unit * (bar.len * 0.5f);
  cv::Point2f e1 = bar.center + unit * (bar.len * 0.5f);
  if (e0.y <= e1.y) {
    bar.axisTop = e0;
    bar.axisBottom = e1;
  } else {
    bar.axisTop = e1;
    bar.axisBottom = e0;
  }

  return true;
}

// Collect short red pieces before applying the normal length and area thresholds.
bool buildBar(const cv::Mat& bgr, const cv::Mat& mask, const std::vector<cv::Point>& contour,
              LightBar& bar, std::vector<LightBar>& fragments)
{
  const double area = cv::contourArea(contour);
  if (area < cfg::kMinFragmentArea) return false;
  bar.rect = cv::minAreaRect(contour);
  if (area >= cfg::kMinBarArea) trimLowerRightGlare(bgr, mask, contour, bar.rect);
  if (!setBarGeometry(bar)) return false;
  measurePurity(bgr, contour, bar);
  const bool ratioOk = bar.ratio >= cfg::kMinBarRatio;
  const bool lenOk = bar.len >= cfg::kMinBarLen;
  const bool tiltOk = bar.tiltDeg <= cfg::kMaxBarTiltDeg;
  const bool rgOk = bar.rg >= cfg::g_tune.minRG;
  const bool rbOk = bar.rb >= cfg::g_tune.minRB;

  if (verbose()) {
    std::printf("  轮廓 area=%.0f 中心(%.0f,%.0f) 长%.1f 宽%.1f 比%.2f 斜%.1f R-G=%.0f R-B=%.0f | %s%s%s%s%s\n",
                area, bar.center.x, bar.center.y, bar.len, bar.wid, bar.ratio, bar.tiltDeg,
                bar.rg, bar.rb, ratioOk ? "" : "比X", lenOk ? "" : "长X", tiltOk ? "" : "斜X",
                rgOk ? "" : "RGX", rbOk ? "" : "RBX");
  }

  const bool barOk = area >= cfg::kMinBarArea && ratioOk && lenOk && tiltOk && rgOk && rbOk;
  if (tiltOk && rgOk && rbOk && bar.len >= cfg::kMinFragmentLen && bar.ratio >= 1.25f)
    fragments.push_back(bar);
  return barOk;
}

bool joinFragments(const LightBar& upper, const LightBar& lower, LightBar& joined)
{
  const float width = std::max(upper.wid, lower.wid);
  if (std::min(upper.wid, lower.wid) < 0.60f * width) return false;

  const cv::Point2f delta = lower.center - upper.center;
  const float distance = static_cast<float>(cv::norm(delta));
  if (delta.y <= 0.f || std::abs(delta.x) > delta.y) return false;
  const cv::Point2f unit = delta / distance;
  const LightBar& longer = upper.len >= lower.len ? upper : lower;
  const cv::Point2f axis = (longer.axisBottom - longer.axisTop) / longer.len;
  const float lateral = std::abs(delta.x * axis.y - delta.y * axis.x);
  if (lateral > 0.65f * width || unit.dot(axis) < 0.94f) return false;

  const auto extent = [&](const LightBar& part) {
    const auto project = [&](cv::Point2f p) { return (p - upper.center).dot(unit); };
    const auto extrema = std::minmax_element(std::begin(part.box), std::end(part.box),
        [&](cv::Point2f a, cv::Point2f b) { return project(a) < project(b); });
    return cv::Vec2f(project(*extrema.first), project(*extrema.second));
  };
  const cv::Vec2f top = extent(upper);
  const cv::Vec2f bottom = extent(lower);
  const float gap = bottom[0] - top[1];
  const float length = bottom[1] - top[0];
  if (gap < 1.f || gap > cfg::kMaxFragmentGapRatio * length || length < cfg::kMinBarLen)
    return false;

  const float meanWidth = (upper.wid * upper.len + lower.wid * lower.len) / (upper.len + lower.len);
  const float angle = static_cast<float>(std::atan2(unit.y, unit.x) * 180.0 / kPi);
  joined.rect = cv::RotatedRect(upper.center + unit * ((top[0] + bottom[1]) * 0.5f),
                                {length, meanWidth}, angle);
  joined.rg = std::min(upper.rg, lower.rg);
  joined.rb = std::min(upper.rb, lower.rb);
  joined.joinedFragments = true;
  return setBarGeometry(joined) && joined.ratio >= cfg::kMinBarRatio;
}

}  // namespace

DetectResult detect(const cv::Mat& bgr, int maskMode)
{
  DetectResult res;
  res.mask = buildRedMask(bgr, maskMode);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(res.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  res.contours = static_cast<int>(contours.size());

  std::vector<LightBar> fragments;
  for (const auto& c : contours) {
    LightBar bar;
    if (buildBar(bgr, res.mask, c, bar, fragments)) res.bars.push_back(bar);
  }
  for (const auto& upper : fragments) {
    for (const auto& lower : fragments) {
      LightBar joined;
      if (joinFragments(upper, lower, joined)) res.bars.push_back(joined);
    }
  }

  // 从左到右排序，装甲板左右灯条必然一左一右
  std::sort(res.bars.begin(), res.bars.end(),
            [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });

  // 两两配对粗筛
  for (size_t i = 0; i < res.bars.size(); ++i) {
    for (size_t j = i + 1; j < res.bars.size(); ++j) {
      const LightBar& l = res.bars[i];
      const LightBar& r = res.bars[j];
      if (r.joinedFragments) continue;

      const double maxLen = std::max(l.len, r.len);
      const double minLen = std::min(l.len, r.len);
      if (minLen < 1e-3) continue;

      Armor a;
      a.lenDiff = static_cast<float>((maxLen - minLen) / maxLen);
      const cv::Point2f dc = r.center - l.center;
      const double gap = cv::norm(dc);
      a.gapRatio = gap > 1e-3 ? static_cast<float>(gap / maxLen) : 0.f;
      a.slope = gap > 1e-3 ? static_cast<float>(std::fabs(dc.y) / gap) : 1.f;

      const bool lenOk = a.lenDiff <= (l.joinedFragments ? cfg::kMaxRecoveredLenDiff : cfg::kMaxLenDiff);
      const bool gapOk = a.gapRatio >= cfg::kMinGapRatio && a.gapRatio <= cfg::kMaxGapRatio;
      const bool slopeOk = a.slope <= cfg::kMaxSlope;
      const bool parOk = std::fabs(l.tiltDeg - r.tiltDeg) <= 20.0f;
      const std::array<cv::Point2f, 4> axisCorners{
          l.axisTop, r.axisTop, r.axisBottom, l.axisBottom};
      const double meanLen = (maxLen + minLen) * 0.5;
      const double areaRatio = cv::contourArea(axisCorners) / (meanLen * meanLen);
      const bool areaOk = areaRatio <= cfg::kMaxPairAreaRatio;
      if (verbose()) {
        std::printf("  配对 %zu-%zu 长差%.2f 距比%.2f 斜率%.2f 倾差%.1f 面积比%.3f | %s%s%s%s%s\n",
                    i, j, a.lenDiff,
                    a.gapRatio, a.slope, std::fabs(l.tiltDeg - r.tiltDeg), areaRatio,
                    lenOk ? "" : "长差X", gapOk ? "" : "距比X", slopeOk ? "" : "斜X",
                    parOk ? "" : "平行X", areaOk ? "" : "面积X");
      }
      if (!lenOk || !gapOk || !slopeOk || !parOk || !areaOk) continue;

      if (l.joinedFragments || a.gapRatio > cfg::kWidePairGapRatio) {
        const int numberContrast = boardNumberContrast(bgr, l, r);
        const bool boardOk = numberContrast >= cfg::kMinBoardNumberContrast;
        if (verbose()) {
          std::printf("  [board] %zu-%zu gap=%.2f contrast=%d %s\n",
                      i, j, a.gapRatio, numberContrast, boardOk ? "OK" : "REJECT");
        }
        if (!boardOk) continue;
      }

      a.left = l;
      a.right = r;
      if (cfg::g_tune.centerCorners) {
        // 用灯条中线端点：灯条侧边粘了亮斑、或一端被遮住时，中线比外缘稳
        a.corners[0] = a.left.axisTop;
        a.corners[1] = a.right.axisTop;
        a.corners[2] = a.right.axisBottom;
        a.corners[3] = a.left.axisBottom;
      } else {
        outerEdge(a.left, true, &a.left.outerTop, &a.left.outerBottom);
        outerEdge(a.right, false, &a.right.outerTop, &a.right.outerBottom);
        a.corners[0] = a.left.outerTop;
        a.corners[1] = a.right.outerTop;
        a.corners[2] = a.right.outerBottom;
        a.corners[3] = a.left.outerBottom;
      }
      a.center = (l.center + r.center) * 0.5f;
      a.score = a.lenDiff + a.slope;
      if (verbose() && l.joinedFragments) {
        std::printf("  [fragment] left=(%.1f, %.1f) span=%.1f, right=(%.1f, %.1f)\n",
                    l.center.x, l.center.y, l.len, r.center.x, r.center.y);
      }

      res.armors.push_back(a);
    }
  }

  std::sort(res.armors.begin(), res.armors.end(),
            [](const Armor& a, const Armor& b) { return a.score < b.score; });
  return res;
}
