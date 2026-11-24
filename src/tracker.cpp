#include "tracker.hpp"

#include <cmath>

#include "config.hpp"

void Tracker::setDt(double dt)
{
  if (dt > 1e-4 && dt < 1.0) dt_ = dt;
  if (kfReady_) {
    kf_.transitionMatrix.at<double>(0, 3) = dt_;
    kf_.transitionMatrix.at<double>(1, 4) = dt_;
    kf_.transitionMatrix.at<double>(2, 5) = dt_;
  }
}

void Tracker::resetFilter(const cv::Vec3d& xyz)
{
  kf_.init(6, 3, 0, CV_64F);

  cv::setIdentity(kf_.transitionMatrix);
  kf_.transitionMatrix.at<double>(0, 3) = dt_;
  kf_.transitionMatrix.at<double>(1, 4) = dt_;
  kf_.transitionMatrix.at<double>(2, 5) = dt_;

  cv::setIdentity(kf_.measurementMatrix);
  cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1.0));
  kf_.processNoiseCov.at<double>(3, 3) = 400.0;
  kf_.processNoiseCov.at<double>(4, 4) = 400.0;
  kf_.processNoiseCov.at<double>(5, 5) = 400.0;
  // 测量噪声按 PnP 的实际抖动给：σ≈50 mm。这是用实测的"相邻帧位置跳变"
  // 90% 分位（约 40 mm）反推的，给太紧会把真实测量也判成离群。
  cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(2500.0));
  cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(500.0));

  kf_.statePost = cv::Mat::zeros(6, 1, CV_64F);
  kf_.statePost.at<double>(0) = xyz[0];
  kf_.statePost.at<double>(1) = xyz[1];
  kf_.statePost.at<double>(2) = xyz[2];
  kfReady_ = true;
}

TrackOutput Tracker::feed(bool found, const Pose& pose)
{
  TrackOutput out;
  const cv::Mat* prediction = kfReady_ ? &kf_.predict() : nullptr;

  if (found && pose.ok) {
    if (!prediction) {
      resetFilter(pose.xyz);
      out.measurementAccepted = true;
    } else {
      cv::Mat meas(3, 1, CV_64F);
      meas.at<double>(0) = pose.xyz[0];
      meas.at<double>(1) = pose.xyz[1];
      meas.at<double>(2) = pose.xyz[2];

      // 时序新息检验（马氏距离）：测量离预测太远就当成垃圾解丢掉。
      // 旋转时一根灯条被挡住，解出来的四点会自洽但不能信，这道闸就是拦它的。
      const cv::Mat innovation = meas - prediction->rowRange(0, 3);
      const cv::Mat S = kf_.measurementMatrix * kf_.errorCovPre * kf_.measurementMatrix.t() +
                        kf_.measurementNoiseCov;
      const cv::Mat Sinv = S.inv();
      const cv::Mat d2Mat = innovation.t() * Sinv * innovation;
      const double d2 = d2Mat.at<double>(0);
      out.innovationD2 = d2;
      if (!std::isfinite(d2) || d2 > cfg::kInnovationGate) {
        out.innovationRejected = true;
        ++outliers_;
      } else {
        kf_.correct(meas);
        out.measurementAccepted = true;
      }
    }
  }

  if (out.measurementAccepted) {
    ++detectCount_;
    lostFrames_ = 0;
    if (state_ == TrackState::Lost || state_ == TrackState::Detecting) {
      state_ = (detectCount_ >= cfg::kMinDetectFrames) ? TrackState::Tracking
                                                       : TrackState::Detecting;
    } else {
      state_ = TrackState::Tracking;
    }

    last_ = pose;
    // 只有确认之后（连续检出够帧数）才让调用方画框；确认之前先不画，
    // 否则第一次就是错配解时，会凭空显示一个偏了一两米的框。
    out.show = state_ == TrackState::Tracking;
    out.pose = pose;

  } else {
    // 无候选和新息拒绝共用同一套丢帧处理。
    detectCount_ = 0;
    ++lostFrames_;

    const bool confirmed = state_ == TrackState::Tracking || state_ == TrackState::TempLost;
    state_ = TrackState::Lost;
    if (confirmed && prediction && lostFrames_ <= cfg::kMaxTempLostFrames) {
      Pose p = last_;
      p.xyz = cv::Vec3d(prediction->at<double>(0), prediction->at<double>(1),
                        prediction->at<double>(2));
      p.dist = cv::norm(p.xyz);
      if (p.dist >= cfg::kMinDistMm && p.dist <= cfg::kMaxDistMm) {
        state_ = TrackState::TempLost;
        out.show = true;
        out.predicted = true;
        out.pose = p;
      }
    }
    if (state_ == TrackState::Lost) kfReady_ = false;
  }

  out.state = state_;
  out.detectCount = detectCount_;
  out.lostFrames = lostFrames_;
  return out;
}
