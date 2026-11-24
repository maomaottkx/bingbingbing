#pragma once

// 全项目参数集中在这里，.cpp 里不写魔法数字。
namespace cfg {

// ---------- 相机内参（40 张标定板，重投影 RMS 0.1948 px）----------
constexpr double kFx = 2334.3100677892139;
constexpr double kFy = 2335.8999103578203;
constexpr double kCx = 722.62133349286796;
constexpr double kCy = 534.62083305548731;
constexpr double kK1 = -0.080008805450534709;
constexpr double kK2 = 0.085675034453367591;
constexpr double kP1 = 0.0027903039462225331;
constexpr double kP2 = -0.0007315334877910064;
constexpr double kK3 = 0.0;

// ---------- 装甲板物理尺寸（mm）----------
// 角点取在灯条外缘，因此：半宽 = 板外缘到板中心的距离；半高 = 灯条可见发光长度的一半
constexpr double kHalfW = 63.5;
constexpr double kHalfH = 25.0;
// 改用灯条中线端点当角点时，半宽对应"两灯条中心距的一半"（127 mm 外缘减去灯条宽 12 mm）
constexpr double kHalfWCenter = 57.5;

// ---------- 颜色掩膜 ----------
constexpr int kRedH1Low = 0;
constexpr int kRedH1High = 12;
constexpr int kRedH2Low = 165;
constexpr int kRedH2High = 180;
constexpr int kMinS = 60;
constexpr int kMinV = 100;
constexpr int kRedDiff = 40;  // R - max(G,B) 阈值

// ---------- 灯条筛选 ----------
constexpr double kMinBarRatio = 1.5;   // 长边 / 短边
constexpr double kMinBarLen = 25.0;    // 长边像素下限
constexpr double kMinBarArea = 100.0;  // 轮廓面积
constexpr double kMinBarRG = 55.0;     // 平均 R-G，滤掉暖白光源（实测真灯条 64~73，窗户 43）
constexpr double kMinBarRB = 60.0;     // 平均 R-B（实测真灯条 74~128）
constexpr double kMaxBarTiltDeg = 45.0;

// Visible pieces of a light bar split by an occluder.
constexpr double kMinFragmentArea = 55.0;
constexpr double kMinFragmentLen = 12.0;
constexpr double kMaxFragmentGapRatio = 0.30;  // Gap / total span.
constexpr double kMaxRecoveredLenDiff = 0.20;

// ---------- 灯条配对 ----------
constexpr double kMaxLenDiff = 0.35;    // 左右灯条长度差 / 较长者
constexpr double kMaxSlope = 0.35;      // 两灯条中心连线斜率
constexpr double kMinGapRatio = 0.8;    // 中心距 / 较长灯条
constexpr double kMaxGapRatio = 2.8;
constexpr double kMaxPairAreaRatio = 2.5;  // Centerline quadrilateral area / mean bar length squared
constexpr double kWidePairGapRatio = 2.8;  // Wider pairs require visible board detail between bars
constexpr int kMinBoardNumberContrast = 5;  // Difference between central neutral intensity percentiles

// ---------- PnP 裁决 ----------
// 重投影门槛放到 6 px：旋转时一根灯条被挡住，真板的重投影会到 3~6 px，
// 卡 2 px 会把真板自己毙掉（实测 67% -> 92%）。放进来的垃圾解由跟踪器的
// 时序新息检验（kInnovationGate）再筛一道。
constexpr double kMaxReprojPx = 6.0;
constexpr double kMinDistMm = 500.0;
constexpr double kMaxDistMm = 3500.0;
constexpr double kMaxYawDeg = 70.0;
constexpr double kMaxPitchDeg = 45.0;
// 卡尔曼新息马氏距离平方的上限。3 自由度卡方 0.999 分位是 16.27，但实测 PnP
// 位置噪声有 30~50 mm，用 16.27 会把一半的真实测量也拦掉；取 40 是在"拦住跑飞的解"
// 和"不误伤真测量"之间实测出来的折中（框显示 97.85%、闪烁 11 次、跳变 90% 分位 31 mm）。
constexpr double kInnovationGate = 40.0;

// ---------- 跟踪状态机（参数取自考核包 source_demo.yaml）----------
constexpr int kMinDetectFrames = 5;     // 连续检出多少帧才进入跟踪
constexpr int kMaxTempLostFrames = 15;  // 丢失多少帧才判定目标丢失

// 调参用：阈值可以临时用环境变量覆盖（AIM_MINS / AIM_MINV / AIM_DIFF / AIM_RG / AIM_RB），
// 默认取上面的常量
struct Tune {
  int minS = kMinS;
  int minV = kMinV;
  int redDiff = kRedDiff;
  double minRG = kMinBarRG;
  double minRB = kMinBarRB;
  bool centerCorners = false;  // 角点取灯条中线端点（否则取外缘）
};
inline Tune g_tune;

}  // namespace cfg
