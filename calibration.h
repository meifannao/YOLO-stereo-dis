#pragma once
#include<opencv2\opencv.hpp>
#include<opencv2\dnn.hpp>
#include<thread>     //线程
#include<fstream>
#include<iostream> 
#include <math.h> 


using namespace std;
using namespace cv;
using namespace cv::dnn;

void calibration_init();           //标定初始化
cv::Mat show_left(cv::Mat imagedst_L);     //左图像处理函数
cv::Mat show_right(cv::Mat imagedst_R);    //右图像处理函数
cv::Mat show(cv::Mat image_left, cv::Mat image_right);   //调用左右图像处理函数


