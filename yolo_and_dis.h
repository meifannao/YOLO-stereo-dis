#pragma once

#include "calibration.h"


// distance  
void stereo_match(int blockSize, int uniquenessRatio, int numDisparities);  //����ƥ������Ӳ�
vector<float> ceju(int x, int y);     // ��ȡ�Ӳ�ͼ���õ�������Ϣ


// yolo_detection 
void yolo_init();             // yolo ģ�ͳ�ʼ������
vector<Rect> yolo_detection(cv::Mat pic_left);    // yolo Ŀ����
cv::Mat yolo_and_dis(cv::Mat rgb_frame_left, cv::Mat rgb_frame_right);          // ���Ͳ�����

vector<vector<float>> data_output();
