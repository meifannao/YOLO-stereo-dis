#include "calibration.h"

const int imageWidth = 640; // 摄像头的分辨率
const int imageHeight = 360;
double fScale = 0.5;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;
Mat rgbRectifyImageL, rgbRectifyImageR;

Rect validROIL; // 图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy; // 映射表
Mat Rl, Rr, Pl, Pr, Q;			// 校正旋转矩阵R，投影矩阵P 重投影矩阵Q

//
////新机器人-2
// Mat cameraMatrixL = (Mat_<double>(3, 3) << 424.1038, -0.3896, 335.1211,
//	0, 420.1921, 172.5300,
//	0, 0, 1);
// Mat distCoeffL = (Mat_<double>(5, 1) << -0.0418, 0.0528, -0.00099433, 0.0019, -0.3377);
//
//
//// 右相机标定参数  （内参和外参）
// Mat cameraMatrixR = (Mat_<double>(3, 3) << 428.5113, -0.3351, 319.9433,
//	0, 424.5980, 176.9508,
//	0, 0, 1);
// Mat distCoeffR = (Mat_<double>(5, 1) << -0.0281, -0.0056, 0.0014, 0.0017, -0.2207);
//
//
//// 平移向量和旋转向量
// Mat T = (Mat_<double>(3, 1) << -55.0359, 0.2118, -0.0430);//T平移向量
// Mat rec = (Mat_<double>(3, 3) << 1.0000, -0.00018481, 0.0077,
//	-0.00017787, 0.9989, 0.0474,
//	-0.0077, -0.0474, 0.9988);          //rec旋转向量，
//
// Mat R;//R 旋转矩阵

// 集成双目 HBV-1780-2 S20
Mat cameraMatrixL = (Mat_<double>(3, 3) << 366.1553, -0.1269, 297.7735,
					 0, 362.8715, 166.3552,
					 0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << 0.0877, 0.0365, -0.00071137, -0.000028615, -0.2657);

// 右相机标定参数  （内参和外参）
Mat cameraMatrixR = (Mat_<double>(3, 3) << 367.1318, -0.4996, 303.8951,
					 0, 363.7327, 149.6688,
					 0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << 0.0638, 0.1972, -0.0013, -0.00072949, -0.5715);

// 平移向量和旋转向量
Mat T = (Mat_<double>(3, 1) << -59.5686, -0.0314, 0.0776); // T平移向量
Mat rec = (Mat_<double>(3, 3) << 1.0000, -0.00047684, -0.0035,
		   0.00048064, 1.0000, 0.0011,
		   0.0035, -0.0011, 1.0000); // rec旋转向量，

Mat R; // R 旋转矩阵

// 校正初始化
void calibration_init()
{
	Rodrigues(rec, R); // Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
				  0, imageSize, &validROIL, &validROIR); // 立体匹配

	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy); // 畸变校正  左图
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy); // 畸变校正  右图
}

// 640*360
cv::Mat show_left(cv::Mat imagedst_L)
{
	cvtColor(imagedst_L, grayImageL, COLOR_BGR2GRAY);
	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(imagedst_L, rgbRectifyImageL, mapLx, mapLy, INTER_LINEAR);

	return rectifyImageL, rgbRectifyImageL; // 返回校正后左灰度图，左彩色图
}

cv::Mat show_right(cv::Mat imagedst_R)
{
	cvtColor(imagedst_R, grayImageR, COLOR_BGR2GRAY);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
	remap(imagedst_R, rgbRectifyImageR, mapRx, mapRy, INTER_LINEAR);

	return rectifyImageR, rgbRectifyImageR; // 返回校正后右灰度图
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

	thread frame_left(show_left, frame_L);	 // 左图函数   线程处理
	thread frame_right(show_right, frame_R); // 右图函数   线程处理
	frame_left.join();						 // 等待线程结束
	frame_right.join();

	//////显示在同一张图上
	// Mat canvas;
	// double sf;
	// int w, h;
	// sf = 640. / MAX(imageSize.width, imageSize.height);
	// w = cvRound(imageSize.width * sf);
	// h = cvRound(imageSize.height * sf);
	// canvas.create(h, w * 2, CV_8UC3);   //注意通道

	//////左图像画到画布上
	// Mat canvasPart = canvas(Rect(0, 0, w, h));                                //得到画布的一部分
	// resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小

	// Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域
	//	cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));

	//////右图像画到画布上
	// canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分
	// resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	// Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
	//	cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	////画上对应的线条

	Mat canvas;
	canvas.create(360, 1280, CV_8UC3); // 注意通道
	rgbRectifyImageL.copyTo(canvas(Rect(0, 0, 640, 360)));
	rgbRectifyImageR.copyTo(canvas(Rect(640, 0, 640, 360)));

	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);

	return rgbRectifyImageL;
}
