#pragma once

#include <opencv2/video/tracking.hpp>

#include "types.hpp"

enum class TrackState { Lost, Detecting, Tracking, TempLost };

// 一帧的跟踪输出：这一帧要不要画、画的是实测还是预测
struct TrackOutput {
  bool show = false;
  bool predicted = false;
  bool measurementAccepted = false;
  bool innovationRejected = false;
  double innovationD2 = -1.0;  // -1 表示本帧未做新息检验
  Pose pose;              // 要画的位姿（实测或预测）
  int detectCount = 0;
  int lostFrames = 0;
  TrackState state = TrackState::Lost;
};

// 四状态跟踪器：lost -> detecting -> tracking -> temp_lost -> lost
// 位置用 6 状态匀速卡尔曼（xyz + vxvyvz）外推，供短暂丢帧时保持画面稳定。
class Tracker {
 public:
  void setDt(double dt);

  // 每帧调用一次。found=false 时 pose 参数会被忽略。
  TrackOutput feed(bool found, const Pose& pose);

  TrackState state() const { return state_; }
  int outlierCount() const { return outliers_; }

 private:
  void resetFilter(const cv::Vec3d& xyz);

  double dt_ = 1.0 / 165.0;
  TrackState state_ = TrackState::Lost;
  int detectCount_ = 0;
  int lostFrames_ = 0;
  int outliers_ = 0;   // 被时序检验判为离群、直接丢掉的测量

  Pose last_;

  cv::KalmanFilter kf_{6, 3, 0, CV_64F};
  bool kfReady_ = false;
};
