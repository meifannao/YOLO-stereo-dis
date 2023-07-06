/****************************************************/
/*****    程序：GIS腔体异物定位技术研究       *******/
/*****                                        *******/
/*****    yolov4目标检测                      *******/
/*****    BM立体匹配算法                      *******/
/*****    win10      vs2017                   *******/
/*****    opencv4.4  release                  *******/
/*****    cuda10.1                            *******/
/*****    输入:  gis.avi离线视频              *******/
/*****           相机实时视频                 *******/
/*****    @小马哥                             *******/
/*****    2021.2.24                           *******/

// 速度问题同3_1,（显卡问题）
// 该程序和 3_1_Pro_yolo_dis文件中的程序功能一样，均可实时运行；
// 区别在于3_1加载左右独立视频，该程序加载左右合成视频

#include "calibration.h"
#include "yolo_and_dis.h"

/*****主函数*****/
int main()
{

	VideoCapture cap;
	// cap.open(0);
	cap.open("gis.avi");				 // 集成双目视频输入
	cap.set(CAP_PROP_FRAME_WIDTH, 2560); // 设置捕获视频的宽度
	cap.set(CAP_PROP_FRAME_HEIGHT, 720); // 设置捕获视频的高度

	if (!cap.isOpened()) // 判断是否成功打开相机
	{
		cout << "摄像头打开失败!" << endl;
		return -1;
	}

	Mat frame, frame_left, frame_right;

	while (1)
	{

		cap >> frame; // 从相机捕获一帧图像  2560*720

		frame_left = frame(Rect(0, 0, 1280, 720));	   // 1280*720 左图
		frame_right = frame(Rect(1280, 0, 1280, 720)); // 1280*720 右图

		Mat result;
		vector<vector<float>> output_data;
		double start_time = (double)cv::getTickCount(); // 记录开始时间

		result = yolo_and_dis(frame_left, frame_right);

		// 坐标，距离 输出，上位机需要
		output_data = data_output();
		for (size_t i = 0; i < output_data.size(); i++)
		{
			cout << "世界坐标:" << output_data[i][0] << " " << output_data[i][1] << " " << output_data[i][2] << " " << output_data[i][3] << endl;
		}
		if (output_data.size() != 0)
		{
			cout << "\n"
				 << endl;
		}

		double end_time = (double)cv::getTickCount();																		 // 记录处理结束时间
		double fps = cv::getTickFrequency() / (end_time - start_time);														 // 计算帧数
		double spend_time = (end_time - start_time) / cv::getTickFrequency();												 // 计算帧时间
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms"; // 帧信息，字符内容
		putText(result, FPS, Point(0, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, 8);								 // 左上角显示帧具体信息
		imshow("yolo_and_dis_result", result);																				 // 最终结果图  ----- >  原图上框出异物，显示名称和距离

		waitKey(1);
	}
	return 0;
}