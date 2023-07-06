#include "calibration.h"
#include "yolo_and_dis.h"

bool init_reload = true;      //�Ƿ��������궨�������������أ�֮�󲻼��أ�


extern Mat rectifyImageL, rectifyImageR;
extern Mat rgbRectifyImageL, rgbRectifyImageR;

extern Rect validROIL;//ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������  
extern Rect validROIR;

extern Mat Q;


float d;              // ����
Vec3f  point3;        // ������������
Point origin;         // �������Ӳ�ͼ�϶�Ӧ�ĵ�
Mat xyz, disp8;       // ���ͼ���Ӳ�ͼ


//int blockSize = 4, uniquenessRatio = 7, numDisparities = 9;     // ����ƥ����� 
//int blockSize = 4, uniquenessRatio = 8, numDisparities = 9;   // ����ƥ����� 
Ptr<StereoBM> bm = StereoBM::create(16, 9);      // BM �㷨

// yolo ���������� 
Mat scores;                     //���÷�
vector<Rect> boxes;             //����
vector<int> classIds;           //����
vector<float> confidences;      //���Ŷ�����
vector<string> classNamesVec;   //��ǩ����


float confidenceThreshold = 0.2; // ���Ŷ���ֵ����
double confidence;     //���Ŷ�
Point classIdPoint;    //
vector<int> indices;   //�Ǽ���ֵ����ɸѡ��ļ�����������

float axis_x, axis_y, axis_z;
float dis_dis;

vector<float> axis_single;
vector<vector<float>> axis_all;


dnn::Net net;          //yolo���أ����綨��




/*****����ƥ��*****/
void stereo_match(int blockSize, int uniquenessRatio, int numDisparities)
{
	bm->setBlockSize(2 * blockSize + 5);    // SAD���ڴ�С��5~21֮��Ϊ��
	bm->setROI1(validROIL);
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);       //��С�ӲĬ��ֵΪ0, �����Ǹ�ֵ��int��
	bm->setNumDisparities(numDisparities * 16 + 16);  //�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��,���ڴ�С������16����������int��
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);          //uniquenessRatio ��Ҫ���Է�ֹ��ƥ��
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);

	Mat blur1, blur2, h1_result, h2_result;

	/*GaussianBlur(rectifyImageL, blur1, Size(3, 3), 0, 0);
	GaussianBlur(rectifyImageR, blur2, Size(3, 3), 0, 0);
	addWeighted(rectifyImageL, 1.5, blur1, -0.5, 0, rectifyImageL);
	addWeighted(rectifyImageR, 1.5, blur2, -0.5, 0, rectifyImageR);*/

	//��������ͼ���񻯣�͹������  ������Ч����
	//���������ֿ��Լ��������о�����͹������Ϊ����
	bilateralFilter(rectifyImageL, blur1, 15, 100, 5);   //˫���˲�
	bilateralFilter(rectifyImageR, blur2, 15, 100, 5);

	Mat h2_kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(blur1, h1_result, -1, h2_kernel);
	filter2D(blur2, h2_result, -1, h2_kernel);
	convertScaleAbs(h1_result, rectifyImageL);
	convertScaleAbs(h2_result, rectifyImageR);


	Mat disp;
	bm->compute(rectifyImageL, rectifyImageR, disp);   //����ͼ�����Ϊ�Ҷ�ͼ
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));   //��������Ӳ���CV_16S��ʽ
	reprojectImageTo3D(disp, xyz, Q, true);  //��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z / W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
	xyz = xyz * 16;         //����16���õ���ȷ����ά������Ϣ

	Mat kernel_morph = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));   //��ȡ��̬ѧ�����ĽṹԪ��
	morphologyEx(xyz, xyz, MORPH_CLOSE, kernel_morph, Point(-1, -1));    //��̬ѧ���������ն�����ֹ��λ��������ն������¾��벻׼
	//imshow("disparity", disp8);     //�õ��Ӳ�ͼ����ȡ������ֵ���Լ���õ����꣬������Ϣ

}



/***** ѡ�����ص���в�� *****/
vector<float> ceju(int x, int y)
{
	vector<float> zuobiao_all;
	origin = Point(x, y);     //ͼ������  ������Ϊ��λ
	point3 = xyz.at<Vec3f>(origin);
	//  �������� X              �������� Y             �������� Z       ��λ mm 
	d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
	d = sqrt(d);    //mm
	d = d / 10.0;   //cm

	//cout << "����:" << point3[0] / 10 << " " << point3[1] / 10 << " " << point3[2] / 10 << " " << d << endl;

	//zuobiao_all.push_back(x);      //ͼ�����꣬�����Ҫ��ע�������ʱ�����������
	//zuobiao_all.push_back(y);
	zuobiao_all.push_back(point3[0] / 10);
	zuobiao_all.push_back(point3[1] / 10);
	zuobiao_all.push_back(point3[2] / 10);
	zuobiao_all.push_back(d);

	return zuobiao_all;

}


// yolo ��ʼ�����֣�����ģ��
void yolo_init()
{
	//ifstream classNamesFile("./model/yolo-obj.names");  
	ifstream classNamesFile("./model/yolo-screw.names");

	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))    //���������ļ���ȡ
			classNamesVec.push_back(className);
	}
	for (int i = 0; i < classNamesVec.size(); i++) {
		cout << i + 1 << "\t" << classNamesVec[i].c_str() << endl;   //�����������
	}

	//----------------------------ģ������---------------------------------------
	//String cfg = "./model/yolo-obj.cfg";
	//String weight = "./model/yolo-obj.weights";

	
	String cfg = "./model/yolo-screw.cfg";
	String weight = "./model/yolo-screw.weights";

	//ģ�Ͷ���
	net = readNetFromDarknet(cfg, weight);    //����ѵ���õ�ģ��
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);   //CUDA����
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

}


//*********** Ŀ���⣨��ͼ��
vector<Rect> yolo_detection(cv::Mat pic_left)
{

	Mat inputBlob = blobFromImage(pic_left, 1.0 / 255, Size(608, 608), Scalar());
	net.setInput(inputBlob);   //Ԥ�������������

	//��ȡδ���������
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	std::vector<Mat> outs;
	net.forward(outs, outNames);   //�Լ��ص�ͼƬ��������Ԥ��


	float* data;
	int centerX, centerY, width, height, left, top;

	//�ҳ����е�Ŀ�꼰��λ��
	for (int i = 0; i < outs.size(); ++i) {
		data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
			scores = outs[i].row(j).colRange(5, outs[i].cols);
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confidenceThreshold) {
				centerX = (int)(data[0] * pic_left.cols);
				centerY = (int)(data[1] * pic_left.rows);
				width = (int)(data[2] * pic_left.cols);
				height = (int)(data[3] * pic_left.rows);
				left = centerX - width / 2;
				top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);      //��ǩ���
				confidences.push_back((float)confidence);     //Ԥ������ֵ���
				boxes.push_back(Rect(left, top, width, height));      //Ԥ����ţ�boxes  ���ۻ�����Ҫˢ������
			}
		}
	}

	NMSBoxes(boxes, confidences, 0.3, 0.2, indices);   //�Ǽ���ֵ���ƣ�ɾ���ظ����֣����ر������ֵ���� indices

	return boxes;

}




//���Ͳ��    ����1280*720 ͼƬ
cv::Mat yolo_and_dis(cv::Mat rgb_frame_left, cv::Mat rgb_frame_right)
{

	//���س�ʼ���������õ����ֲ������������ó�ʼ����������ֹ����ѭ�������У�����ʱ��
	if (init_reload)
	{
		calibration_init();
		yolo_init();
		init_reload = false;
	}


	Mat rgb_rectify_left;
	rgb_rectify_left = show(rgb_frame_left, rgb_frame_right);


	thread dis(stereo_match, 7, 8, 9);    //����ƥ�䣬�߳�
	thread yolo(yolo_detection, rgb_rectify_left);  //yolo��⣬�߳�
	dis.join();
	yolo.join();                         //�ȴ��߳̽���


	//�����򻭳�����ͬʱ��ʾ��������;�����Ϣ
	Scalar rectColor, textColor; //box �� text ����ɫ
	Rect box, textBox;
	int idx;  //�������
	String className;
	Size labelSize;


	for (int i = 0; i < indices.size(); ++i) {
		idx = indices[i];
		className = classNamesVec[classIds[idx]];

		labelSize = getTextSize(className, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
		box = boxes[idx];


		textBox = Rect(Point(box.x - 1, box.y), Point(box.x + labelSize.width, box.y - labelSize.height));
		rectColor = Scalar(0, 255, 0);  //������ɫ
		textColor = Scalar(255, 0, 0);  //���������ɫ

		axis_single = ceju((int)(box.x + box.width / 2), (int)(box.y + box.height / 2));

		if (axis_single[3] > 700)
			axis_single = ceju((int)(box.x + box.width / 3), (int)(box.y + box.height / 3));
		if (axis_single[3] > 700)
			axis_single = ceju((int)(box.x + 2 * box.width / 3), (int)(box.y + box.height / 3));
		if (axis_single[3] > 700)
			axis_single = ceju((int)(box.x + box.width / 4), (int)(box.y + 3 * box.height / 4));
		if (axis_single[3] > 700)
			ceju((int)(box.x + 3 * box.width / 4), (int)(box.y + 3 * box.height / 4));




		axis_all.push_back(axis_single);

		//circle(rgb_rectify_left, Point(box.x + box.width / 3, box.y + box.height / 3), 2, Scalar(0, 255, 0), -1, 8);  //����
		//circle(rgb_rectify_left, Point(box.x + 2 * box.width / 3, box.y + box.height / 3), 2, Scalar(0, 255, 0), -1, 8);  //����
		//circle(rgb_rectify_left, Point(box.x + box.width / 2, box.y + box.height / 2), 2, Scalar(0, 255, 0), -1, 8);  //����
		//circle(rgb_rectify_left, Point(box.x + box.width / 4, box.y + 3 * box.height / 4), 2, Scalar(0, 255, 0), -1, 8);  //����
		//circle(rgb_rectify_left, Point(box.x + 3 * box.width / 4, box.y + 3 * box.height / 4), 2, Scalar(0, 255, 0), -1, 8);  //����

		rectangle(rgb_rectify_left, box, rectColor, 1, 8, 0);       //������
		rectangle(rgb_rectify_left, textBox, rectColor, -1, 8, 0);  //����������������ʾ��
		putText(rgb_rectify_left, className.c_str(), Point(box.x, box.y - 2), FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, 8);  //��ʾ��������



		if (axis_single[3] < 500)
			putText(rgb_rectify_left, format("%4.1f", axis_single[3]), Point(box.x + labelSize.width + 5, box.y - 2), FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, 8);//��ʾ����ֵ

	}

	classIds.clear();       //�Ǳ��룬�����������չ���    
	confidences.clear();   //������գ���ֹ�ۻ�
	boxes.clear();         //������գ���ֹ�ۻ�


	// 640 X 360   --->>  1280 X 720
	Size dsize = Size(rgb_frame_left.cols, rgb_frame_left.rows);   //���ͼ̫С����ʾ������Э����2���Ŵ�
	Mat imagedst_result = Mat(dsize, CV_32S);
	resize(rgb_rectify_left, imagedst_result, dsize);                // �Ŵ�frame  --- >  imagedst 

	//imshow("src_image", imagedst_result);
	//Mat kern = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);     //resize����ͼƬģ������һ��
	//filter2D(imagedst_result, imagedst_result, imagedst_result.depth(), kern);


	return rgb_rectify_left;    // 640 x 360 �����ͼ 
	//return imagedst_result;       // 1280 x 720 �����ͼ 
}

vector<vector<float>> data_output()
{

	vector<vector<float>> axis_2(axis_all);   //������ʱ����
	axis_single.clear();      //���㣬��ֹ�ۻ�
	axis_all.clear();

	return axis_2;

}


