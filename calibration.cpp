#include "calibration.h"

const int imageWidth = 640; // ����ͷ�ķֱ���
const int imageHeight = 360;
double fScale = 0.5;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;
Mat rgbRectifyImageL, rgbRectifyImageR;

Rect validROIL; // ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy; // ӳ���
Mat Rl, Rr, Pl, Pr, Q;			// У����ת����R��ͶӰ����P ��ͶӰ����Q

//
////�»�����-2
// Mat cameraMatrixL = (Mat_<double>(3, 3) << 424.1038, -0.3896, 335.1211,
//	0, 420.1921, 172.5300,
//	0, 0, 1);
// Mat distCoeffL = (Mat_<double>(5, 1) << -0.0418, 0.0528, -0.00099433, 0.0019, -0.3377);
//
//
//// ������궨����  ���ڲκ���Σ�
// Mat cameraMatrixR = (Mat_<double>(3, 3) << 428.5113, -0.3351, 319.9433,
//	0, 424.5980, 176.9508,
//	0, 0, 1);
// Mat distCoeffR = (Mat_<double>(5, 1) << -0.0281, -0.0056, 0.0014, 0.0017, -0.2207);
//
//
//// ƽ����������ת����
// Mat T = (Mat_<double>(3, 1) << -55.0359, 0.2118, -0.0430);//Tƽ������
// Mat rec = (Mat_<double>(3, 3) << 1.0000, -0.00018481, 0.0077,
//	-0.00017787, 0.9989, 0.0474,
//	-0.0077, -0.0474, 0.9988);          //rec��ת������
//
// Mat R;//R ��ת����

// ����˫Ŀ HBV-1780-2 S20
Mat cameraMatrixL = (Mat_<double>(3, 3) << 366.1553, -0.1269, 297.7735,
					 0, 362.8715, 166.3552,
					 0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << 0.0877, 0.0365, -0.00071137, -0.000028615, -0.2657);

// ������궨����  ���ڲκ���Σ�
Mat cameraMatrixR = (Mat_<double>(3, 3) << 367.1318, -0.4996, 303.8951,
					 0, 363.7327, 149.6688,
					 0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << 0.0638, 0.1972, -0.0013, -0.00072949, -0.5715);

// ƽ����������ת����
Mat T = (Mat_<double>(3, 1) << -59.5686, -0.0314, 0.0776); // Tƽ������
Mat rec = (Mat_<double>(3, 3) << 1.0000, -0.00047684, -0.0035,
		   0.00048064, 1.0000, 0.0011,
		   0.0035, -0.0011, 1.0000); // rec��ת������

Mat R; // R ��ת����

// У����ʼ��
void calibration_init()
{
	Rodrigues(rec, R); // Rodrigues�任
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
				  0, imageSize, &validROIL, &validROIR); // ����ƥ��

	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy); // ����У��  ��ͼ
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy); // ����У��  ��ͼ
}

// 640*360
cv::Mat show_left(cv::Mat imagedst_L)
{
	cvtColor(imagedst_L, grayImageL, COLOR_BGR2GRAY);
	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(imagedst_L, rgbRectifyImageL, mapLx, mapLy, INTER_LINEAR);

	return rectifyImageL, rgbRectifyImageL; // ����У������Ҷ�ͼ�����ɫͼ
}

cv::Mat show_right(cv::Mat imagedst_R)
{
	cvtColor(imagedst_R, grayImageR, COLOR_BGR2GRAY);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
	remap(imagedst_R, rgbRectifyImageR, mapRx, mapRy, INTER_LINEAR);

	return rectifyImageR, rgbRectifyImageR; // ����У�����һҶ�ͼ
}

cv::Mat show(cv::Mat image_left, cv::Mat image_right)
{
	Size dsize = Size(image_left.cols * fScale, image_left.rows * fScale);
	Mat frame_L = Mat(dsize, CV_32S);
	Mat frame_R = Mat(dsize, CV_32S);
	resize(image_left, frame_L, dsize);
	resize(image_right, frame_R, dsize);

	/*gray_left, rgb_left = show_left(frame_L);
	gray_right, rgb_right = show_right(frame_R);*/

	thread frame_left(show_left, frame_L);	 // ��ͼ����   �̴߳���
	thread frame_right(show_right, frame_R); // ��ͼ����   �̴߳���
	frame_left.join();						 // �ȴ��߳̽���
	frame_right.join();

	//////��ʾ��ͬһ��ͼ��
	// Mat canvas;
	// double sf;
	// int w, h;
	// sf = 640. / MAX(imageSize.width, imageSize.height);
	// w = cvRound(imageSize.width * sf);
	// h = cvRound(imageSize.height * sf);
	// canvas.create(h, w * 2, CV_8UC3);   //ע��ͨ��

	//////��ͼ�񻭵�������
	// Mat canvasPart = canvas(Rect(0, 0, w, h));                                //�õ�������һ����
	// resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //��ͼ�����ŵ���canvasPartһ����С

	// Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //��ñ���ȡ������
	//	cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));

	//////��ͼ�񻭵�������
	// canvasPart = canvas(Rect(w, 0, w, h));                                      //��û�������һ����
	// resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	// Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
	//	cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	////���϶�Ӧ������

	Mat canvas;
	canvas.create(360, 1280, CV_8UC3); // ע��ͨ��
	rgbRectifyImageL.copyTo(canvas(Rect(0, 0, 640, 360)));
	rgbRectifyImageR.copyTo(canvas(Rect(640, 0, 640, 360)));

	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	return rgbRectifyImageL;
}
