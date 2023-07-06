#include "calibration.h"
#include "yolo_and_dis.h"

bool init_reload = true;      //是否加载相机标定参数（开机加载，之后不加载）


extern Mat rectifyImageL, rectifyImageR;
extern Mat rgbRectifyImageL, rgbRectifyImageR;

extern Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
extern Rect validROIR;

extern Mat Q;


float d;              // 距离
Vec3f  point3;        // 距离计算坐标点
Point origin;         // 检测框在视差图上对应的点
Mat xyz, disp8;       // 深度图，视差图


//int blockSize = 4, uniquenessRatio = 7, numDisparities = 9;     // 立体匹配参数 
//int blockSize = 4, uniquenessRatio = 8, numDisparities = 9;   // 立体匹配参数 
Ptr<StereoBM> bm = StereoBM::create(16, 9);      // BM 算法

// yolo 检测变量定义 
Mat scores;                     //检测得分
vector<Rect> boxes;             //检测框
vector<int> classIds;           //类别号
vector<float> confidences;      //置信度容器
vector<string> classNamesVec;   //标签名字


float confidenceThreshold = 0.2; // 置信度阈值设置
double confidence;     //置信度
Point classIdPoint;    //
vector<int> indices;   //非极大值抑制筛选后的检测框的索引序号

float axis_x, axis_y, axis_z;
float dis_dis;

vector<float> axis_single;
vector<vector<float>> axis_all;


dnn::Net net;          //yolo加载，网络定义




/*****立体匹配*****/
void stereo_match(int blockSize, int uniquenessRatio, int numDisparities)
{
	bm->setBlockSize(2 * blockSize + 5);    // SAD窗口大小，5~21之间为宜
	bm->setROI1(validROIL);
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);       //最小视差，默认值为0, 可以是负值，int型
	bm->setNumDisparities(numDisparities * 16 + 16);  //视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);          //uniquenessRatio 主要可以防止误匹配
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);

	Mat blur1, blur2, h1_result, h2_result;

	/*GaussianBlur(rectifyImageL, blur1, Size(3, 3), 0, 0);
	GaussianBlur(rectifyImageR, blur2, Size(3, 3), 0, 0);
	addWeighted(rectifyImageL, 1.5, blur1, -0.5, 0, rectifyImageL);
	addWeighted(rectifyImageR, 1.5, blur2, -0.5, 0, rectifyImageR);*/

	//基本处理，图像锐化，凸显纹理  还是有效果的
	//基本处理部分可以继续深入研究，以凸显纹理为方向
	bilateralFilter(rectifyImageL, blur1, 15, 100, 5);   //双边滤波
	bilateralFilter(rectifyImageR, blur2, 15, 100, 5);

	Mat h2_kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(blur1, h1_result, -1, h2_kernel);
	filter2D(blur2, h2_result, -1, h2_kernel);
	convertScaleAbs(h1_result, rectifyImageL);
	convertScaleAbs(h2_result, rectifyImageR);


	Mat disp;
	bm->compute(rectifyImageL, rectifyImageR, disp);   //输入图像必须为灰度图
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));   //计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true);  //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;         //乘以16，得到正确的三维坐标信息

	Mat kernel_morph = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));   //获取形态学操作的结构元素
	morphologyEx(xyz, xyz, MORPH_CLOSE, kernel_morph, Point(-1, -1));    //形态学操作，填充空洞，防止定位像素落入空洞区导致距离不准
	//imshow("disparity", disp8);     //得到视差图，读取其像素值可以计算得到坐标，距离信息

}



/***** 选择像素点进行测距 *****/
vector<float> ceju(int x, int y)
{
	vector<float> zuobiao_all;
	origin = Point(x, y);     //图像坐标  以像素为单位
	point3 = xyz.at<Vec3f>(origin);
	//  世界坐标 X              世界坐标 Y             世界坐标 Z       单位 mm 
	d = point3[0] * point3[0] + point3[1] * point3[1] + point3[2] * point3[2];
	d = sqrt(d);    //mm
	d = d / 10.0;   //cm

	//cout << "距离:" << point3[0] / 10 << " " << point3[1] / 10 << " " << point3[2] / 10 << " " << d << endl;

	//zuobiao_all.push_back(x);      //图像坐标，如果需要，注意在输出时更改索引序号
	//zuobiao_all.push_back(y);
	zuobiao_all.push_back(point3[0] / 10);
	zuobiao_all.push_back(point3[1] / 10);
	zuobiao_all.push_back(point3[2] / 10);
	zuobiao_all.push_back(d);

	return zuobiao_all;

}


// yolo 初始化部分，加载模型
void yolo_init()
{
	//ifstream classNamesFile("./model/yolo-obj.names");  
	ifstream classNamesFile("./model/yolo-screw.names");

	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))    //异物名称文件读取
			classNamesVec.push_back(className);
	}
	for (int i = 0; i < classNamesVec.size(); i++) {
		cout << i + 1 << "\t" << classNamesVec[i].c_str() << endl;   //输出异物名称
	}

	//----------------------------模型设置---------------------------------------
	//String cfg = "./model/yolo-obj.cfg";
	//String weight = "./model/yolo-obj.weights";

	
	String cfg = "./model/yolo-screw.cfg";
	String weight = "./model/yolo-screw.weights";

	//模型读入
	net = readNetFromDarknet(cfg, weight);    //加载训练好的模型
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);   //CUDA加速
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

}


//*********** 目标检测（左图）
vector<Rect> yolo_detection(cv::Mat pic_left)
{

	Mat inputBlob = blobFromImage(pic_left, 1.0 / 255, Size(608, 608), Scalar());
	net.setInput(inputBlob);   //预处理后输入网络

	//获取未连接输出层
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	std::vector<Mat> outs;
	net.forward(outs, outNames);   //对加载的图片进行推理预测


	float* data;
	int centerX, centerY, width, height, left, top;

	//找出所有的目标及其位置
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

				classIds.push_back(classIdPoint.x);      //标签存放
				confidences.push_back((float)confidence);     //预测置信值存放
				boxes.push_back(Rect(left, top, width, height));      //预测框存放，boxes  在累积，需要刷新清零
			}
		}
	}

	NMSBoxes(boxes, confidences, 0.3, 0.2, indices);   //非极大值抑制，删除重复部分，返回保留部分的序号 indices

	return boxes;

}




//检测和测距    传入1280*720 图片
cv::Mat yolo_and_dis(cv::Mat rgb_frame_left, cv::Mat rgb_frame_right)
{

	//加载初始化函数，得到部分参数，而后弃用初始化函数，禁止其在循环中运行，消耗时间
	if (init_reload)
	{
		calibration_init();
		yolo_init();
		init_reload = false;
	}


	Mat rgb_rectify_left;
	rgb_rectify_left = show(rgb_frame_left, rgb_frame_right);


	thread dis(stereo_match, 7, 8, 9);    //立体匹配，线程
	thread yolo(yolo_detection, rgb_rectify_left);  //yolo检测，线程
	dis.join();
	yolo.join();                         //等待线程结束


	//将检测框画出来，同时显示异物种类和距离信息
	Scalar rectColor, textColor; //box 和 text 的颜色
	Rect box, textBox;
	int idx;  //类别索引
	String className;
	Size labelSize;


	for (int i = 0; i < indices.size(); ++i) {
		idx = indices[i];
		className = classNamesVec[classIds[idx]];

		labelSize = getTextSize(className, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
		box = boxes[idx];


		textBox = Rect(Point(box.x - 1, box.y), Point(box.x + labelSize.width, box.y - labelSize.height));
		rectColor = Scalar(0, 255, 0);  //检测框颜色
		textColor = Scalar(255, 0, 0);  //输出文字颜色

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

		//circle(rgb_rectify_left, Point(box.x + box.width / 3, box.y + box.height / 3), 2, Scalar(0, 255, 0), -1, 8);  //左上
		//circle(rgb_rectify_left, Point(box.x + 2 * box.width / 3, box.y + box.height / 3), 2, Scalar(0, 255, 0), -1, 8);  //右上
		//circle(rgb_rectify_left, Point(box.x + box.width / 2, box.y + box.height / 2), 2, Scalar(0, 255, 0), -1, 8);  //中心
		//circle(rgb_rectify_left, Point(box.x + box.width / 4, box.y + 3 * box.height / 4), 2, Scalar(0, 255, 0), -1, 8);  //左下
		//circle(rgb_rectify_left, Point(box.x + 3 * box.width / 4, box.y + 3 * box.height / 4), 2, Scalar(0, 255, 0), -1, 8);  //右下

		rectangle(rgb_rectify_left, box, rectColor, 1, 8, 0);       //画检测框
		rectangle(rgb_rectify_left, textBox, rectColor, -1, 8, 0);  //画异物名称文字显示框
		putText(rgb_rectify_left, className.c_str(), Point(box.x, box.y - 2), FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, 8);  //显示异物名称



		if (axis_single[3] < 500)
			putText(rgb_rectify_left, format("%4.1f", axis_single[3]), Point(box.x + labelSize.width + 5, box.y - 2), FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1, 8);//显示距离值

	}

	classIds.clear();       //非必须，谨慎起见，清空归零    
	confidences.clear();   //必须清空，防止累积
	boxes.clear();         //必须清空，防止累积


	// 640 X 360   --->>  1280 X 720
	Size dsize = Size(rgb_frame_left.cols, rgb_frame_left.rows);   //结果图太小，显示比例不协调，2倍放大
	Mat imagedst_result = Mat(dsize, CV_32S);
	resize(rgb_rectify_left, imagedst_result, dsize);                // 放大，frame  --- >  imagedst 

	//imshow("src_image", imagedst_result);
	//Mat kern = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);     //resize导致图片模糊，锐化一下
	//filter2D(imagedst_result, imagedst_result, imagedst_result.depth(), kern);


	return rgb_rectify_left;    // 640 x 360 检测结果图 
	//return imagedst_result;       // 1280 x 720 检测结果图 
}

vector<vector<float>> data_output()
{

	vector<vector<float>> axis_2(axis_all);   //复制临时变量
	axis_single.clear();      //清零，防止累积
	axis_all.clear();

	return axis_2;

}


