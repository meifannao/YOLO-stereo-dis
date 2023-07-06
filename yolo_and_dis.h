#pragma once

#include "calibration.h"


// distance  
void stereo_match(int blockSize, int uniquenessRatio, int numDisparities);  //立体匹配计算视差
vector<float> ceju(int x, int y);     // 读取视差图，得到距离信息


// yolo_detection 
void yolo_init();             // yolo 模型初始化部分
vector<Rect> yolo_detection(cv::Mat pic_left);    // yolo 目标检测
cv::Mat yolo_and_dis(cv::Mat rgb_frame_left, cv::Mat rgb_frame_right);          // 检测和测距调用

vector<vector<float>> data_output();
