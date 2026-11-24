#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "types.hpp"

struct PoseBranchTrace {
  double reproj;
  double nz;
};

struct PoseTrace {
  std::vector<PoseBranchTrace> branches;
  int selected = -1;
};

// 用 4 个像素角点做 PnP，解出装甲板中心在相机坐标系的位置。
// 内部会自检：重投影误差、距离范围、yaw/pitch 范围，任一不过就不置 ok。
// rvecHint：上一帧的旋转向量。IPPE 对共面四点会给出两个几乎一样好的解，
//           给上一帧姿态就能选到连续的那个，否则姿态角会逐帧来回跳。
Pose solvePose(const Armor& armor, const cv::Vec3d* rvecHint = nullptr,
               PoseTrace* trace = nullptr);

// 把一个位姿反投影回图像：给出板子 4 个角点与中心点的像素坐标（画预测框用）。
void projectPose(const Pose& pose, cv::Point2f corners[4], cv::Point2f* center);
