#include "solver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "config.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// 装甲板 3D 模型。板子坐标系与相机系同向：x 向右、y 向下、z 指向远离相机。
// 这样正对相机时旋转矩阵接近单位阵，yaw/pitch 都是 0，读数最直观。
// 角点顺序与 Armor::corners 一致：左上 右上 右下 左下。
const std::vector<cv::Point3f>& objectPoints()
{
  const float halfW = static_cast<float>(cfg::g_tune.centerCorners ? cfg::kHalfWCenter : cfg::kHalfW);
  const float halfH = static_cast<float>(cfg::kHalfH);
  static std::vector<cv::Point3f> pts;
  pts = {
    {-halfW, -halfH, 0.f},
    {+halfW, -halfH, 0.f},
    {+halfW, +halfH, 0.f},
    {-halfW, +halfH, 0.f},
  };
  return pts;
}

cv::Mat cameraMatrix()
{
  return (cv::Mat_<double>(3, 3) << cfg::g_cam.fx, 0, cfg::g_cam.cx, 0, cfg::g_cam.fy, cfg::g_cam.cy,
          0, 0, 1);
}

cv::Mat distCoeffs()
{
  return (cv::Mat_<double>(5, 1) << cfg::g_cam.k1, cfg::g_cam.k2, cfg::g_cam.p1, cfg::g_cam.p2,
          cfg::g_cam.k3);
}

}  // namespace

Pose solvePose(const Armor& armor, const cv::Vec3d* rvecHint, PoseTrace* trace)
{
  Pose pose;
  if (trace) *trace = {};
  std::vector<cv::Point2f> imgPoints(armor.corners, armor.corners + 4);

  // 装甲板是共面四点，用 IPPE。IPPE 会给出两个解的候选，靠"板面朝向相机"来挑，
  // 否则近正对时会解出镜像姿态（yaw/pitch 差一截，距离也跟着偏）。
  std::vector<cv::Mat> rvecs, tvecs;
  const int nSol = cv::solvePnPGeneric(objectPoints(), imgPoints, cameraMatrix(), distCoeffs(),
                                      rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
  if (nSol <= 0) return pose;

  // 两个解的旋转差（度）
  auto rotGap = [](const cv::Vec3d& a, const cv::Vec3d& b) {
    cv::Mat Ra, Rb;
    cv::Rodrigues(cv::Mat(a), Ra);
    cv::Rodrigues(cv::Mat(b), Rb);
    const cv::Mat rel = Ra.t() * Rb;
    double c = (cv::trace(rel)[0] - 1.0) * 0.5;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c) * 180.0 / kPi;
  };

  std::vector<double> errs(nSol, 1e9);
  std::vector<double> nzs(nSol, 0.0);
  std::vector<double> pitches(nSol, 0.0);
  double errMin = 1e9;
  for (int i = 0; i < nSol; ++i) {
    std::vector<cv::Point2f> reproj;
    cv::projectPoints(objectPoints(), rvecs[i], tvecs[i], cameraMatrix(), distCoeffs(), reproj);
    double sum = 0.0;
    for (size_t k = 0; k < reproj.size(); ++k) sum += cv::norm(reproj[k] - imgPoints[k]);
    errs[i] = sum / static_cast<double>(reproj.size());

    cv::Mat R;
    cv::Rodrigues(rvecs[i], R);
    nzs[i] = R.at<double>(2, 2);  // 板面法向的 z 分量，正对相机时应为正
    if (trace) trace->branches.push_back({errs[i], nzs[i]});
    pitches[i] = std::asin(-std::max(-1.0, std::min(1.0, R.at<double>(1, 2)))) * 180.0 / kPi;
    if (nzs[i] > 0.0) errMin = std::min(errMin, errs[i]);

    if (std::getenv("AIM_DEBUG")) {
      std::printf("    [pnp] 解%d reproj=%.4f nz=%.3f pitch=%.2f\n", i, errs[i], nzs[i], pitches[i]);
    }
  }

  // 只保留"板面朝向相机"且重投影足够好的解，其余视为不可用
  std::vector<int> usable;
  for (int i = 0; i < nSol; ++i) {
    if (nzs[i] > 0.0 && errs[i] <= errMin + 0.5) usable.push_back(i);
  }

  int bestIdx = -1;
  if (!usable.empty() && rvecHint != nullptr) {
    // 有上一帧姿态：挑与上一帧最接近的解，避免解在两条支路上逐帧来回跳
    double bestGap = 1e9;
    for (int i : usable) {
      const cv::Vec3d r(rvecs[i].at<double>(0), rvecs[i].at<double>(1), rvecs[i].at<double>(2));
      const double gap = rotGap(r, *rvecHint);
      if (gap < bestGap) {
        bestGap = gap;
        bestIdx = i;
      }
    }
  } else if (!usable.empty()) {
    // 没有提示：取倾角最接近正对相机的那支（共面四点的镜像歧义只能这样定，见 README 已知限制）
    double bestAbs = 1e9;
    for (int i : usable) {
      if (std::fabs(pitches[i]) < bestAbs) {
        bestAbs = std::fabs(pitches[i]);
        bestIdx = i;
      }
    }
  }
  if (bestIdx < 0) {
    double bestScore = 1e9;
    for (int i = 0; i < nSol; ++i) {
      const double score = errs[i] + (nzs[i] > 0.0 ? 0.0 : 10.0);
      if (score < bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
  }
  if (bestIdx < 0) return pose;

  if (trace) trace->selected = bestIdx;
  const cv::Mat& rvec = rvecs[bestIdx];
  const cv::Mat& tvec = tvecs[bestIdx];

  // 自检 1：重投影误差
  std::vector<cv::Point2f> reproj;
  cv::projectPoints(objectPoints(), rvec, tvec, cameraMatrix(), distCoeffs(), reproj);
  double sum = 0.0;
  for (size_t i = 0; i < reproj.size(); ++i) sum += cv::norm(reproj[i] - imgPoints[i]);
  pose.reproj = sum / static_cast<double>(reproj.size());

  pose.xyz = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
  pose.dist = cv::norm(pose.xyz);

  // 姿态角。板面法向 = 旋转矩阵第三列（物体 z 轴在相机系下的方向）
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  pose.yaw = std::atan2(R.at<double>(0, 2), R.at<double>(2, 2)) * 180.0 / kPi;
  pose.pitch = std::asin(-std::max(-1.0, std::min(1.0, R.at<double>(1, 2)))) * 180.0 / kPi;

  // 三条门槛，按顺序记下是哪条没过（调参时要看这个）
  if (pose.reproj > cfg::kMaxReprojPx) {
    pose.fail = 1;
    return pose;
  }
  if (pose.dist < cfg::kMinDistMm || pose.dist > cfg::kMaxDistMm) {
    pose.fail = 2;
    return pose;
  }
  if (std::fabs(pose.yaw) > cfg::kMaxYawDeg) {
    pose.fail = 3;
    return pose;
  }
  if (std::fabs(pose.pitch) > cfg::kMaxPitchDeg) {
    pose.fail = 4;
    return pose;
  }

  pose.rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
  pose.ok = true;
  return pose;
}

void projectPose(const Pose& pose, cv::Point2f corners[4], cv::Point2f* center)
{
  const cv::Mat rvec(pose.rvec);
  const cv::Mat tvec(pose.xyz);
  std::vector<cv::Point2f> out;
  cv::projectPoints(objectPoints(), rvec, tvec, cameraMatrix(), distCoeffs(), out);
  for (int i = 0; i < 4 && i < static_cast<int>(out.size()); ++i) corners[i] = out[i];
  if (center) {
    const cv::Point2f c = (out[0] + out[1] + out[2] + out[3]) * 0.25f;
    *center = c;
  }
}
