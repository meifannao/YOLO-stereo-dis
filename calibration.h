#pragma once
#include<opencv2\opencv.hpp>
#include<opencv2\dnn.hpp>
#include<thread>     //�߳�
#include<fstream>
#include<iostream> 
#include <math.h> 


using namespace std;
using namespace cv;
using namespace cv::dnn;

void calibration_init();           //�궨��ʼ��
cv::Mat show_left(cv::Mat imagedst_L);     //��ͼ������
cv::Mat show_right(cv::Mat imagedst_R);    //��ͼ������
cv::Mat show(cv::Mat image_left, cv::Mat image_right);   //��������ͼ������


