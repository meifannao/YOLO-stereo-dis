/****************************************************/
/*****    ����GISǻ�����ﶨλ�����о�       *******/
/*****                                        *******/
/*****    yolov4Ŀ����                      *******/
/*****    BM����ƥ���㷨                      *******/
/*****    win10      vs2017                   *******/
/*****    opencv4.4  release                  *******/
/*****    cuda10.1                            *******/
/*****    ����:  gis.avi������Ƶ              *******/
/*****           ���ʵʱ��Ƶ                 *******/
/*****    @С���                             *******/
/*****    2021.2.24                           *******/

// �ٶ�����ͬ3_1,���Կ����⣩
// �ó���� 3_1_Pro_yolo_dis�ļ��еĳ�����һ��������ʵʱ���У�
// ��������3_1�������Ҷ�����Ƶ���ó���������Һϳ���Ƶ

#include "calibration.h"
#include "yolo_and_dis.h"

/*****������*****/
int main()
{

	VideoCapture cap;
	// cap.open(0);
	cap.open("gis.avi");				 // ����˫Ŀ��Ƶ����
	cap.set(CAP_PROP_FRAME_WIDTH, 2560); // ���ò�����Ƶ�Ŀ��
	cap.set(CAP_PROP_FRAME_HEIGHT, 720); // ���ò�����Ƶ�ĸ߶�

	if (!cap.isOpened()) // �ж��Ƿ�ɹ������
	{
		cout << "����ͷ��ʧ��!" << endl;
		return -1;
	}

	Mat frame, frame_left, frame_right;

	while (1)
	{

		cap >> frame; // ���������һ֡ͼ��  2560*720

		frame_left = frame(Rect(0, 0, 1280, 720));	   // 1280*720 ��ͼ
		frame_right = frame(Rect(1280, 0, 1280, 720)); // 1280*720 ��ͼ

		Mat result;
		vector<vector<float>> output_data;
		double start_time = (double)cv::getTickCount(); // ��¼��ʼʱ��

		result = yolo_and_dis(frame_left, frame_right);

		// ���꣬���� �������λ����Ҫ
		output_data = data_output();
		for (size_t i = 0; i < output_data.size(); i++)
		{
			cout << "��������:" << output_data[i][0] << " " << output_data[i][1] << " " << output_data[i][2] << " " << output_data[i][3] << endl;
		}
		if (output_data.size() != 0)
		{
			cout << "\n"
				 << endl;
		}

		double end_time = (double)cv::getTickCount();																		 // ��¼�������ʱ��
		double fps = cv::getTickFrequency() / (end_time - start_time);														 // ����֡��
		double spend_time = (end_time - start_time) / cv::getTickFrequency();												 // ����֡ʱ��
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms"; // ֡��Ϣ���ַ�����
		putText(result, FPS, Point(0, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, 8);								 // ���Ͻ���ʾ֡������Ϣ
		imshow("yolo_and_dis_result", result);																				 // ���ս��ͼ  ----- >  ԭͼ�Ͽ�������ʾ���ƺ;���

		waitKey(1);
	}
	return 0;
}