#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector.hpp"
#include "draw.hpp"
#include "solver.hpp"
#include "tracker.hpp"

namespace {

struct Options {
  std::string video;
  std::string camera;   // 内参文件（考核包里的 calibration/intrinsics.yaml）
  std::string save;
  std::string snapshot;
  std::string maskOut;   // 单帧模式下把掩膜存成 PNG
  int frame = -1;       // >=0 只跑单帧
  int limit = 0;        // >0 只跑前 N 帧
  int snapFrame = -1;   // 整段模式下在第 N 帧存一张 PNG
  std::vector<int> snapFrames;  // 也可以一次存多帧（--snapshot-frame 1,2,3）
  int maskMode = 0;     // 0 并集 1 交集
  double scale = 1.0;   // 输出缩放
  double outFps = 0;    // >0 时用这个帧率写视频（默认跟原视频一致）
  bool show = true;
  bool debug = false;
  bool quiet = false;
  bool compare = false;
  bool trace = false;   // stdout 输出逐帧 CSV，stderr 输出汇总
  bool tracker = true;  // 关掉就只看逐帧检测结果，用来做"上跟踪前/后"对比
};

// 在候选里挑重投影误差最小的那块板
struct Picked {
  bool found = false;
  Armor armor;
  Pose pose;
  int evaluated = 0;
  int candidateIndex = -1;
  int failCode = 0;      // 全都没通过时，最接近通过的那块卡在哪
  double failReproj = 0;
};

// 在候选里挑最可信的一块板。
// 有上一帧位置时，把"离上一帧多远"计入打分——否则错配的灯条对（两个不相干亮条）
// 只要重投影略小就会被选中，位姿可能偏到一两米外。
Picked pickBest(const std::vector<Armor>& candidates, bool hasAnchor, const cv::Vec3d& anchor,
                std::vector<PoseTrace>* traces = nullptr)
{
  Picked best;
  const size_t n = std::min<size_t>(candidates.size(), 30);
  double bestScore = 1e9;
  double bestFail = 1e9;
  for (size_t i = 0; i < n; ++i) {
    PoseTrace* trace = traces ? &traces->emplace_back() : nullptr;
    const Pose p = solvePose(candidates[i], nullptr, trace);
    ++best.evaluated;
    if (!p.ok) {
      if (p.fail != 0 && p.reproj < bestFail) {
        bestFail = p.reproj;
        best.failCode = p.fail;
        best.failReproj = p.reproj;
      }
      continue;
    }
    double score = p.reproj;
    if (hasAnchor) score += 0.004 * cv::norm(p.xyz - anchor);  // 250 mm 折合 1 px
    if (score < bestScore) {
      bestScore = score;
      best.found = true;
      best.armor = candidates[i];
      best.pose = p;
      best.candidateIndex = static_cast<int>(i);
    }
  }
  return best;
}

const char* stateName(TrackState s)
{
  switch (s) {
    case TrackState::Lost: return "LOST";
    case TrackState::Detecting: return "DETECTING";
    case TrackState::Tracking: return "TRACKING";
    case TrackState::TempLost: return "TEMP_LOST";
  }
  return "?";
}

void printTrace(long long frame, const DetectResult& dr, const Picked& pk, const TrackOutput& tr,
                bool hasPrior, bool upperHalf, int branch, const std::vector<PoseTrace>& traces)
{
  std::printf("%lld,%d,%d,%d,%d,%d,%.6f,%s,%d,%d,%zu,%zu,%d,%d,%d,%d,%d,"
              "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,\"",
              frame, pk.found, tr.measurementAccepted, tr.show, tr.predicted,
              tr.innovationRejected, tr.innovationD2, stateName(tr.state),
              tr.detectCount, tr.lostFrames, dr.bars.size(), dr.armors.size(), pk.evaluated,
              pk.candidateIndex, branch, pk.found ? 0 : pk.failCode, hasPrior,
              pk.pose.reproj, pk.pose.xyz[0], pk.pose.xyz[1], pk.pose.xyz[2],
              pk.pose.yaw, pk.pose.pitch, pk.armor.center.x, pk.armor.center.y,
              tr.pose.xyz[0], tr.pose.xyz[1], tr.pose.xyz[2]);
  // 每个条目由 候选编号:IPPE分支编号:重投影:nz 组成，编号从零开始。
  const char* sep = "";
  for (size_t i = 0; i < traces.size(); ++i) {
    for (size_t j = 0; j < traces[i].branches.size(); ++j) {
      const auto& solution = traces[i].branches[j];
      std::printf("%s%zu:%zu:%.5f:%.7f", sep, i, j, solution.reproj, solution.nz);
      sep = "|";
    }
  }
  std::printf("\",%d\n", upperHalf);
}

// 读考核包里的内参文件（普通 YAML，不是 OpenCV FileStorage 格式），覆盖默认内参
bool loadCamera(const std::string& path)
{
  std::ifstream in(path);
  if (!in) {
    std::printf("打不开内参文件: %s\n", path.c_str());
    return false;
  }
  std::vector<double> k, d;
  std::string line;
  auto grab = [&](const char* key, std::vector<double>& out) {
    if (line.find(key) == std::string::npos) return;
    const size_t lb = line.find('[');
    const size_t rb = line.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb < lb) return;
    std::string body = line.substr(lb + 1, rb - lb - 1);
    for (char& c : body) {
      if (c == ',') c = ' ';
    }
    std::istringstream is(body);
    std::vector<double> v;
    double x = 0;
    while (is >> x) v.push_back(x);
    if (!v.empty()) out = v;
  };
  while (std::getline(in, line)) {
    grab("camera_matrix", k);
    grab("distort_coeffs", d);
  }
  if (k.size() != 9) {
    std::printf("内参文件里没有 9 个数的 camera_matrix: %s\n", path.c_str());
    return false;
  }
  cfg::g_cam.fx = k[0];
  cfg::g_cam.cx = k[2];
  cfg::g_cam.fy = k[4];
  cfg::g_cam.cy = k[5];
  if (d.size() >= 4) {
    cfg::g_cam.k1 = d[0];
    cfg::g_cam.k2 = d[1];
    cfg::g_cam.p1 = d[2];
    cfg::g_cam.p2 = d[3];
  }
  if (d.size() >= 5) cfg::g_cam.k3 = d[4];
  std::printf("内参已载入 %s\n  fx=%.4f fy=%.4f cx=%.4f cy=%.4f\n", path.c_str(), cfg::g_cam.fx,
              cfg::g_cam.fy, cfg::g_cam.cx, cfg::g_cam.cy);
  return true;
}

void usage(const char* exe)
{
  std::printf("用法: %s <视频路径> [选项]\n", exe);
  std::printf("  --all                处理整段视频（默认）\n");
  std::printf("  --limit N            只处理前 N 帧\n");
  std::printf("  --frame N            只处理第 N 帧（0 起）并打印详细结果\n");
  std::printf("  --save 路径          写出结果视频（.mp4 / .avi）\n");
  std::printf("  --snapshot 路径      单帧模式下保存 PNG\n");
  std::printf("  --scale 0.5          输出视频缩放\n");
  std::printf("  --out-fps 30         输出视频帧率（默认跟原视频一致）\n");
  std::printf("  --compare            左右分屏：左=逐帧实测 右=跟踪结果\n");
  std::printf("  --camera 路径        读考核包的内参文件（calibration/intrinsics.yaml）\n");
  std::printf("  --mask and|or        颜色掩膜方式（默认 or）\n");
  std::printf("  --debug              叠加灯条/候选/掩膜\n");
  std::printf("  --no-tracker         关掉跟踪，只看逐帧检测（做对比用）\n");
  std::printf("  --no-show            不弹窗\n");
  std::printf("  --quiet              不打印每帧/进度\n");
  std::printf("  --trace              stdout 输出逐帧 CSV，stderr 输出汇总\n");
}

bool parseArgs(int argc, char** argv, Options& opt)
{
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](double& v) {
      if (i + 1 < argc) v = std::atof(argv[++i]);
    };
    if (a == "--all") {
      continue;
    } else if (a == "--camera") {
      if (i + 1 < argc) opt.camera = argv[++i];
    } else if (a == "--limit") {
      if (i + 1 < argc) opt.limit = std::atoi(argv[++i]);
    } else if (a == "--frame") {
      if (i + 1 < argc) opt.frame = std::atoi(argv[++i]);
    } else if (a == "--save") {
      if (i + 1 < argc) opt.save = argv[++i];
    } else if (a == "--snapshot") {
      if (i + 1 < argc) opt.snapshot = argv[++i];
    } else if (a == "--snapshot-frame") {
      if (i + 1 < argc) {
        // 支持 "300" 或 "300,850,1550"
        std::string list = argv[++i];
        size_t pos = 0;
        while (pos <= list.size()) {
          const size_t comma = list.find(',', pos);
          const std::string one = list.substr(pos, comma == std::string::npos ? std::string::npos
                                                                             : comma - pos);
          if (!one.empty()) opt.snapFrames.push_back(std::atoi(one.c_str()));
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
      }
    } else if (a == "--dump-mask") {
      if (i + 1 < argc) opt.maskOut = argv[++i];
    } else if (a == "--scale") {
      if (i + 1 < argc) opt.scale = std::atof(argv[++i]);
    } else if (a == "--out-fps") {
      if (i + 1 < argc) opt.outFps = std::atof(argv[++i]);
    } else if (a == "--compare") {
      opt.compare = true;
    } else if (a == "--trace") {
      opt.trace = true;
    } else if (a == "--mask") {
      if (i + 1 < argc) {
        const std::string m = argv[++i];
        opt.maskMode = (m == "and") ? 1 : (m == "diff" ? 2 : 0);
      }
    } else if (a == "--debug") {
      opt.debug = true;
    } else if (a == "--no-show") {
      opt.show = false;
    } else if (a == "--no-tracker") {
      opt.tracker = false;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return false;
    } else if (!a.empty() && a[0] == '-') {
      double dummy = 0;
      next(dummy);
      std::printf("未知选项: %s\n", a.c_str());
      return false;
    } else if (opt.video.empty()) {
      opt.video = a;
    }
  }
  if (opt.video.empty()) {
    usage(argv[0]);
    return false;
  }
  return true;
}

bool windowAvailable()
{
  const char* d = std::getenv("DISPLAY");
  return d != nullptr && *d != '\0';
}

}  // namespace

int main(int argc, char** argv)
{
  // 环境变量覆盖颜色阈值，方便不重编译地扫参数
  if (const char* v = std::getenv("AIM_MINS")) cfg::g_tune.minS = std::atoi(v);
  if (const char* v = std::getenv("AIM_MINV")) cfg::g_tune.minV = std::atoi(v);
  if (const char* v = std::getenv("AIM_DIFF")) cfg::g_tune.redDiff = std::atoi(v);
  if (const char* v = std::getenv("AIM_RG")) cfg::g_tune.minRG = std::atof(v);
  if (const char* v = std::getenv("AIM_RB")) cfg::g_tune.minRB = std::atof(v);
  if (const char* v = std::getenv("AIM_CORNERS")) {
    cfg::g_tune.centerCorners = (std::string(v) == "center");
  }

  Options opt;
  if (!parseArgs(argc, argv, opt)) return 1;
  if (!opt.camera.empty() && !loadCamera(opt.camera)) return 1;

  cv::VideoCapture cap(opt.video);
  if (!cap.isOpened()) {
    std::printf("打不开视频: %s\n", opt.video.c_str());
    return 1;
  }

  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps < 1.0 || fps > 1000.0) fps = 30.0;
  const int totalFramesHint = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  // ---------------- 单帧模式 ----------------
  if (opt.frame >= 0) {
    cap.set(cv::CAP_PROP_POS_FRAMES, opt.frame);
    cv::Mat frame;
    if (!cap.read(frame)) {
      std::printf("读不到第 %d 帧\n", opt.frame);
      return 1;
    }

    const int64 t0 = cv::getTickCount();
    const DetectResult dr = detect(frame, opt.maskMode);
    const Picked pk = pickBest(dr.armors, false, cv::Vec3d(0, 0, 0));
    const double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();

    std::printf("=== 单帧 %d ===\n", opt.frame);
    std::printf("原始轮廓 %d 个 -> 合格灯条 %zu 根 -> 配对候选 %zu 块\n", dr.contours,
                dr.bars.size(), dr.armors.size());
    for (size_t i = 0; i < dr.bars.size(); ++i) {
      const auto& b = dr.bars[i];
      std::printf("  灯条%zu 中心(%.1f, %.1f) 长%.1f 宽%.1f 长宽比%.2f 倾斜%.1f° R-G=%.0f R-B=%.0f\n",
                  i, b.center.x, b.center.y, b.len, b.wid, b.ratio, b.tiltDeg, b.rg, b.rb);
    }
    std::printf("选中: %s", pk.found ? "是" : "否（没有候选通过 PnP 裁决）");
    if (pk.found) {
      std::printf("（候选 #%d，共评估 %d 块）\n", pk.candidateIndex, pk.evaluated);
      std::printf("  角点 左上(%.1f, %.1f) 右上(%.1f, %.1f) 右下(%.1f, %.1f) 左下(%.1f, %.1f)\n",
                  pk.armor.corners[0].x, pk.armor.corners[0].y, pk.armor.corners[1].x,
                  pk.armor.corners[1].y, pk.armor.corners[2].x, pk.armor.corners[2].y,
                  pk.armor.corners[3].x, pk.armor.corners[3].y);
      std::printf("  中心 (%.1f, %.1f)\n", pk.armor.center.x, pk.armor.center.y);
      std::printf("  位置 x=%.1f y=%.1f z=%.1f mm  距离=%.1f mm\n", pk.pose.xyz[0], pk.pose.xyz[1],
                  pk.pose.xyz[2], pk.pose.dist);
      std::printf("  yaw=%.3f° pitch=%.3f° 重投影=%.4f px\n", pk.pose.yaw, pk.pose.pitch,
                  pk.pose.reproj);
    } else {
      std::printf("\n");
    }
    std::printf("耗时 %.2f ms\n", ms);

    if (opt.debug) drawDebug(frame, dr.mask, dr.bars, dr.armors);
    if (!opt.maskOut.empty()) {
      cv::imwrite(opt.maskOut, dr.mask);
      std::printf("掩膜已存: %s\n", opt.maskOut.c_str());
    }
    if (pk.found) {
      drawPlate(frame, pk.armor.corners, false);
      drawInferredCorners(frame, pk.armor);
      drawPanelOutline(frame, pk.armor);
    }
    char buf[256];
    std::vector<std::string> hud;
    hud.push_back("SINGLE FRAME " + std::to_string(opt.frame));
    if (pk.found) {
      std::snprintf(buf, sizeof(buf), "XYZ(mm) %.1f %.1f %.1f  DIST %.1f", pk.pose.xyz[0],
                    pk.pose.xyz[1], pk.pose.xyz[2], pk.pose.dist);
      hud.push_back(buf);
      std::snprintf(buf, sizeof(buf), "YAW %.2f  PITCH %.2f  REPROJ %.3f px", pk.pose.yaw,
                    pk.pose.pitch, pk.pose.reproj);
      hud.push_back(buf);
    } else {
      hud.push_back("NO ARMOR");
    }
    std::snprintf(buf, sizeof(buf), "TIME %.2f ms", ms);
    hud.push_back(buf);
    drawHud(frame, hud);

    if (!opt.snapshot.empty()) {
      cv::imwrite(opt.snapshot, frame);
      std::printf("快照已保存: %s\n", opt.snapshot.c_str());
    }
    if (opt.show && windowAvailable()) {
      cv::imshow("aim2027", frame);
      cv::waitKey(0);
    }
    return 0;
  }

  // ---------------- 整段视频 ----------------
  cv::VideoWriter writer;
  bool writerTried = false;

  Tracker tracker;
  tracker.setDt(1.0 / fps);

  long long nFrames = 0, nGeometryPassed = 0, nPredicted = 0, nBlank = 0, nUpperHalf = 0, nShown = 0;
  long long nAccepted = 0, nGeometryUpperHalf = 0;
  long long nBlink = 0;          // 框从有到无的次数
  long long nEmptyBars = 0;
  long long maxGap = 0, curGap = 0;
  long long barHist[10] = {0};
  long long failHist[8] = {0};   // 未检出原因：0 无候选 1 重投影 2 距离 3 yaw 4 pitch
  // 相邻实测帧之间的位置跳变，用来判断放宽门槛后进来的是不是"垃圾位姿"
  std::vector<double> jumps;
  cv::Vec3d prevMeasXyz{0, 0, 0};
  long long prevMeasFrame = -10;
  double distMin = 1e9, distMax = 0;
  std::vector<double> costMs;
  bool prevShow = false;
  Pose lastAccepted;
  bool hasPrior = false;

  if (opt.trace) {
    std::puts("frame,geometry_pass,tracker_accepted,displayed,predicted,innovation_rejected,"
              "innovation_d2,tracker_state,detect_count,lost_frames,bars,candidates,evaluated,"
              "candidate_index,ippe_branch,geometry_fail_code,has_prior,reproj_px,"
              "meas_x_mm,meas_y_mm,meas_z_mm,meas_yaw_deg,meas_pitch_deg,center_x_px,center_y_px,"
              "out_x_mm,out_y_mm,out_z_mm,ippe_candidates,geometry_upper_half");
  }

  cv::Mat frame;
  while (cap.read(frame)) {
    if (opt.limit > 0 && nFrames >= opt.limit) break;
    ++nFrames;

    const int64 t0 = cv::getTickCount();
    const DetectResult dr = detect(frame, opt.maskMode);
    const bool hadPrior = hasPrior;
    std::vector<PoseTrace> traces;
    Picked pk = pickBest(dr.armors, hadPrior, lastAccepted.xyz, opt.trace ? &traces : nullptr);
    int branch = -1;
    if (opt.trace && pk.found) branch = traces[pk.candidateIndex].selected;
    // 用上一帧的姿态作为提示再解一次，避免 IPPE 两个解逐帧来回跳
    if (pk.found && hadPrior) {
      PoseTrace refineTrace;
      const Pose refined = solvePose(pk.armor, &lastAccepted.rvec, opt.trace ? &refineTrace : nullptr);
      if (refined.ok) {
        pk.pose = refined;
        branch = refineTrace.selected;
      }
    }
    TrackOutput tr;
    if (opt.tracker) {
      tr = tracker.feed(pk.found, pk.pose);
    } else {
      // 不做跟踪：只显示本帧实测，缺一帧就掉框
      tr.show = pk.found;
      tr.predicted = false;
      tr.pose = pk.pose;
      tr.state = pk.found ? TrackState::Tracking : TrackState::Lost;
    }
    if (opt.tracker && tr.state == TrackState::Lost) {
      hasPrior = false;
    } else if (opt.tracker ? tr.measurementAccepted : pk.found) {
      lastAccepted = pk.pose;
      hasPrior = true;
    }
    const double ms = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
    costMs.push_back(ms);

    const bool geometryUpperHalf = pk.found && pk.armor.center.y < frame.rows / 2;
    if (opt.trace) {
      printTrace(nFrames, dr, pk, tr, hadPrior, geometryUpperHalf,
                 branch, traces);
    }
    if (pk.found) ++nGeometryPassed;
    if (tr.measurementAccepted) ++nAccepted;
    if (geometryUpperHalf) ++nGeometryUpperHalf;
    if (!pk.found) ++failHist[pk.failCode & 7];
    if (tr.show) {
      ++nShown;
      if (tr.predicted) ++nPredicted;
    } else {
      ++nBlank;
    }
    if (pk.found) {
      distMin = std::min(distMin, pk.pose.dist);
      distMax = std::max(distMax, pk.pose.dist);
    }
    if (tr.show) {
      // 记录"画在屏幕上"的那个位置，相邻帧跳变才是观众看到的抖动
      if (prevMeasFrame == nFrames - 1) jumps.push_back(cv::norm(tr.pose.xyz - prevMeasXyz));
      prevMeasXyz = tr.pose.xyz;
      prevMeasFrame = nFrames;
    }
    if (dr.bars.empty()) ++nEmptyBars;

    const int barsBin = std::min<int>(static_cast<int>(dr.bars.size()), 9);
    barHist[barsBin]++;

    if (tr.show) {
      curGap = 0;
      if (!tr.predicted && pk.found && pk.armor.center.y < frame.rows / 2) ++nUpperHalf;
    } else {
      ++curGap;
      maxGap = std::max(maxGap, curGap);
    }
    if (prevShow && !tr.show) ++nBlink;
    prevShow = tr.show;

    // 绘制
    // 对比模式：左侧显示使用同一先验筛出的候选，右侧显示跟踪器输出。
    char buf[256];
    cv::Mat rawView = opt.compare ? frame.clone() : cv::Mat();
    if (!rawView.empty()) {
      if (pk.found) {
        drawPlate(rawView, pk.armor.corners, false);
        drawInferredCorners(rawView, pk.armor);
        drawPanelOutline(rawView, pk.armor);
      }
      std::vector<std::string> rawHud;
      rawHud.push_back("GEOMETRY PASS (shared selection)");
      std::snprintf(buf, sizeof(buf), "FRAME %lld   BARS %zu   CANDS %zu", nFrames,
                    dr.bars.size(), dr.armors.size());
      rawHud.push_back(buf);
      drawHud(rawView, rawHud);
    }

    if (opt.debug) drawDebug(frame, dr.mask, dr.bars, dr.armors);
    if (tr.show) {
      if (tr.predicted) {
        cv::Point2f corners[4];
        projectPose(tr.pose, corners, nullptr);
        drawPlate(frame, corners, true);
      } else if (pk.found) {
        drawPlate(frame, pk.armor.corners, false);
        drawInferredCorners(frame, pk.armor);
        drawPanelOutline(frame, pk.armor);
      }
    }

    std::vector<std::string> hud;
    std::snprintf(buf, sizeof(buf), "FRAME %lld / %d   STATE %s%s", nFrames, totalFramesHint,
                  stateName(tr.state), tr.predicted ? " (PREDICT)" : "");
    hud.push_back(buf);
    if (tr.show) {
      std::snprintf(buf, sizeof(buf), "XYZ(mm) %.1f %.1f %.1f   DIST %.1f mm", tr.pose.xyz[0],
                    tr.pose.xyz[1], tr.pose.xyz[2], tr.pose.dist);
      hud.push_back(buf);
      std::snprintf(buf, sizeof(buf), "YAW %.2f deg  PITCH %.2f deg  REPROJ %.3f px", tr.pose.yaw,
                    tr.pose.pitch, tr.pose.reproj);
      hud.push_back(buf);
    } else {
      hud.push_back("NO TARGET");
    }
    std::snprintf(buf, sizeof(buf), "TIME %.2f ms   BARS %zu   CANDS %zu", ms, dr.bars.size(),
                  dr.armors.size());
    hud.push_back(buf);
    drawHud(frame, hud);

    // 对比模式把两幅画面并排
    cv::Mat display = frame;
    if (!rawView.empty()) cv::hconcat(rawView, frame, display);

    if (!opt.snapshot.empty()) {
      for (int want : opt.snapFrames) {
        if (nFrames != want) continue;
        // 同一路径存多帧时自动加序号，避免互相覆盖
        std::string path = opt.snapshot;
        if (opt.snapFrames.size() > 1) {
          const size_t dot = path.rfind('.');
          const std::string suffix = "_" + std::to_string(want);
          path = (dot == std::string::npos) ? path + suffix
                                            : path.substr(0, dot) + suffix + path.substr(dot);
        }
        cv::imwrite(path, display);
        std::printf("第 %lld 帧已存为 %s\n", nFrames, path.c_str());
      }
    }

    if (!opt.save.empty()) {
      if (!writerTried) {
        writerTried = true;
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        const cv::Size sz(static_cast<int>(display.cols * opt.scale),
                          static_cast<int>(display.rows * opt.scale));
        writer.open(opt.save, fourcc, opt.outFps > 0 ? opt.outFps : fps, sz);
        if (!writer.isOpened()) {
          std::printf("写不了视频: %s\n", opt.save.c_str());
          opt.save.clear();
        }
      }
      if (writer.isOpened()) {
        cv::Mat out = display;
        if (std::fabs(opt.scale - 1.0) > 1e-6) {
          cv::resize(display, out, cv::Size(), opt.scale, opt.scale, cv::INTER_AREA);
        }
        writer.write(out);
      }
    }

    if (opt.show && windowAvailable()) {
      cv::imshow("aim2027", display);
      if (cv::waitKey(1) == 'q') break;
    }

    if (!opt.quiet && nFrames % 200 == 0) {
      std::fprintf(stderr, "\r处理中 %lld 帧 ...", nFrames);
      std::fflush(stderr);
    }
  }
  if (!opt.quiet) std::fprintf(stderr, "\r");

  if (writer.isOpened()) writer.release();

  // ---------------- 汇总 ----------------
  double sum = 0.0;
  for (double v : costMs) sum += v;
  const double avg = costMs.empty() ? 0.0 : sum / static_cast<double>(costMs.size());
  double median = 0.0;
  if (!costMs.empty()) {
    std::vector<double> s = costMs;
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    median = s[s.size() / 2];
  }

  const auto percent = [nFrames](long long count) { return nFrames ? 100.0 * count / nFrames : 0.0; };
  FILE* summary = opt.trace ? stderr : stdout;
  std::fprintf(summary, "\n===== 汇总 (%s) =====\n", opt.video.c_str());
  std::fprintf(summary, "总帧数              %lld\n", nFrames);
  std::fprintf(summary, "几何门通过帧        %lld  (%.2f%%)\n", nGeometryPassed, percent(nGeometryPassed));
  if (opt.tracker) {
    std::fprintf(summary, "跟踪器接受帧        %lld  (%.2f%%)\n", nAccepted, percent(nAccepted));
  }
  std::fprintf(summary, "上半屏几何通过帧    %lld\n", nGeometryUpperHalf);
  std::fprintf(summary, "预测保持帧          %lld\n", nPredicted);
  std::fprintf(summary, "无目标帧            %lld\n", nBlank);
  std::fprintf(summary, "框显示总帧          %lld  (%.2f%%)\n", nShown, percent(nShown));
  std::fprintf(summary, "框消失次数(闪烁)    %lld\n", nBlink);
  std::fprintf(summary, "最长连续无框        %lld 帧\n", maxGap);
  std::fprintf(summary, "无灯条帧            %lld\n", nEmptyBars);
  std::fprintf(summary, "上半屏实测显示帧    %lld\n", nUpperHalf);
  std::fprintf(summary, "耗时 平均/中位      %.2f / %.2f ms/帧\n", avg, median);
  std::fprintf(summary, "灯条数分布          ");
  for (int i = 0; i < 10; ++i) {
    if (barHist[i]) std::fprintf(summary, "%d根:%lld  ", i, barHist[i]);
  }
  std::fprintf(summary, "\n");
  std::fprintf(summary, "未检出原因          无候选:%lld  重投影:%lld  距离:%lld  yaw:%lld  pitch:%lld\n",
              failHist[0], failHist[1], failHist[2], failHist[3], failHist[4]);
  if (!jumps.empty()) {
    std::sort(jumps.begin(), jumps.end());
    std::fprintf(summary, "相邻帧位置跳变      中位 %.1f mm  90%%分位 %.1f mm  最大 %.1f mm（%zu 个相邻对）\n",
                jumps[jumps.size() / 2], jumps[jumps.size() * 9 / 10], jumps.back(),
                jumps.size());
  }
  std::fprintf(summary, "实测距离范围        %.1f ~ %.1f mm\n",
              distMin > 1e8 ? 0.0 : distMin, distMax);
  if (opt.tracker) {
    std::fprintf(summary, "时序检验拦下测量    %d 帧（离群，未用于修正）\n", tracker.outlierCount());
  }
  return 0;
}
