#pragma once

#include <opencv2/core.hpp>

// 一根灯条（最小外接旋转矩形的几何量 + 颜色纯度）
struct LightBar {
  bool joinedFragments = false;
  cv::RotatedRect rect;
  cv::Point2f box[4]{};       // 旋转矩形 4 顶点
  cv::Point2f center{};       // 中心
  cv::Point2f outerTop{}, outerBottom{};  // 朝板外侧那一条边的上下顶点
  cv::Point2f axisTop{}, axisBottom{};    // 长轴中线的上下端点
  float len = 0.f;            // 长边像素
  float wid = 0.f;            // 短边像素
  float ratio = 0.f;          // 长边 / 短边
  float rg = 0.f;             // 轮廓像素平均 R-G
  float rb = 0.f;             // 轮廓像素平均 R-B
  float tiltDeg = 0.f;        // 长边与竖直方向的夹角
};

// 一块装甲板（左右两根灯条配对）
struct Armor {
  LightBar left, right;
  cv::Point2f corners[4]{};   // 左上 右上 右下 左下
  cv::Point2f center{};
  float lenDiff = 0.f;        // 左右灯条长度差 / 较长者
  float slope = 0.f;          // 两灯条中心连线斜率
  float gapRatio = 0.f;       // 中心距 / 较长灯条
  float score = 0.f;          // 粗筛打分，越小越好
};

// PnP 解算结果（相机坐标系：x 右，y 下，z 前，单位 mm）
struct Pose {
  cv::Vec3d xyz{0, 0, 0};   // 装甲板中心位置
  cv::Vec3d rvec{0, 0, 0};  // 旋转向量
  double dist = 0.0;
  double yaw = 0.0;         // 度
  double pitch = 0.0;       // 度
  double reproj = 0.0;      // 平均重投影误差，像素
  bool ok = false;
  int fail = 0;             // 没通过时的原因：1 重投影 2 距离 3 yaw 4 pitch
};
