#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <tchar.h>
#include <SDKDDKVer.h>
#include "SGM_lab.h"
#include "auxiliary_fun.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#define M 500

//实现3d点云时调用该库
using namespace Eigen;
using namespace std;
using namespace cv;




int main(int argv, char** argc)
{

    // 初始化路径
	string img_path_left = "../Data/Classic/image1.png";
	string img_path_right = "../Data/Classic/image2.png";

	// 初始化Mat类的元素
	cv::Mat disp_color;

    cv::Mat image_left_color = cv::imread(img_path_left, cv::IMREAD_COLOR);

	//灰度矩阵转化
    cv::Mat image_left = cv::imread(img_path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat image_right = cv::imread(img_path_right, cv::IMREAD_GRAYSCALE);

	// 进行错误处理 判断读取的数据是否符合题目要求
    if (image_left.data == nullptr || image_right.data == nullptr) {
        std::cout << "读取影像失败！" << std::endl;
        return -1;
    }
    if (image_left.rows != image_right.rows || image_left.cols != image_right.cols) {
        std::cout << "左右影像尺寸不一致！" << std::endl;
        return -1;
    }

	//读取图片尺寸
    int32_t width = image_left.cols;
    int32_t height = image_right.rows;
	//存储灰度数据
	uint8_t* bytes_left = new uint8_t[width * height];
	uint8_t* bytes_right = new uint8_t[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = image_left.at<uint8_t>(i, j);
            bytes_right[i * width + j] = image_right.at<uint8_t>(i, j);
        }
    }

    printf("正在储存转化数据...完成!\n");

	// 设置匹配参数
	/********************SGM*************************************/
	// 该SGM实现思想来源于李迎松csdn博客的SGM实现
	// 具体参考包括：对SGM构建类，并采取类函数中match函数对一致性检查、唯一性约束、剔除小联通区集成处理

    Parameter_Set Parameter_Setting;

    // 聚合路径数
    Parameter_Setting.num_paths = 8;
    // 候选视差范围
    Parameter_Setting.min_disparity = argv < 4 ? 0 : atoi(argc[3]);
    Parameter_Setting.max_disparity = argv < 5 ? 64 : atoi(argc[4]);
    // 一致性检查
    Parameter_Setting.is_check_lr = true;
    Parameter_Setting.lrcheck_thres = 1.0f;
    // 唯一性约束
    Parameter_Setting.is_check_unique = true;
    Parameter_Setting.uniqueness_ratio = 0.99;
    // 剔除小连通区
	Parameter_Setting.is_remove_speckles = true;
    Parameter_Setting.min_speckle_aera = 50;
    // 惩罚项P1、P2
    Parameter_Setting.p1 = 10;
    Parameter_Setting.p2_init = 150;
    // 视差图填充
    Parameter_Setting.is_fill_holes = true;

    printf("图片宽度 = %d, 图片高度 = %d\n视差范围 = [%d,%d]\n\n", width, height, Parameter_Setting.min_disparity, Parameter_Setting.max_disparity);

    // 定义SGM匹配类
    match_SGM sgm;

   
    // 初始化
	printf("SGM初始化中...\n");
    if (!sgm.Initialize(width, height, Parameter_Setting)) {
        std::cout << "SGM初始化失败！" << std::endl;
        return -2;
    }
 
    // 匹配
	printf("正在进行匹配...\n");

    // disparity数组保存子像素的视差结果
    auto disparity = new float[uint32_t(width * height)]();
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGM匹配失败！" << std::endl;
        return -2;
    }

	printf("匹配完成...\n");
   
	// 显示视差图
    cv::Mat mat_disparity = cv::Mat(height, width, CV_8UC1);
    float min_disp = width, max_disp = -width;
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            const float disp = disparity[i * width + j];
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            const float disp = disparity[i * width + j];
            if (disp == Invalid_Float) {
                mat_disparity.data[i * width + j] = 0;
            }
            else {
                mat_disparity.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }


	applyColorMap(mat_disparity, disp_color, cv::COLORMAP_JET);

	//结果展示
    cv::imshow("视差图", mat_disparity);
    cv::imshow("视差图-伪彩", disp_color);


	double camera_fx = 718.856, camera_fy = 718.856, camera_cx = 607.1928, camera_cy = 185.2157;// 内参
	double baseline = 0.573;// 基线
	double fx = 4152.073;

	Mat depth(mat_disparity.rows, mat_disparity.cols, CV_16S);  //深度图

	//视差图转深度图
	for (int row = 0; row < depth.rows; row++)
	{
		for (int col = 0; col < depth.cols; col++)
		{
			short d = mat_disparity.ptr<uchar>(row)[col];


			if (d == 0)
				continue;

			depth.ptr<short>(row)[col] = fx * baseline*M / d;
		}
	}

	imshow("depth", depth);
	Mat img_pseudocolor(mat_disparity.rows, mat_disparity.cols, CV_8UC3);//构造RGB图像

	int tmp = 0;
	for (int y = 0; y < mat_disparity.rows; y++)//转为伪彩色图像的具体算法
	{
		for (int x = 0; x < mat_disparity.cols; x++)
		{
			tmp = mat_disparity.at<unsigned char>(y, x);
			img_pseudocolor.at<Vec3b>(y, x)[0] = abs(255 - tmp); //blue
			img_pseudocolor.at<Vec3b>(y, x)[1] = abs(127 - tmp); //green
			img_pseudocolor.at<Vec3b>(y, x)[2] = abs(0 - tmp); //red
		}
	}

	imshow("img_pseudocolor", img_pseudocolor);

	Mat img_color(mat_disparity.rows, mat_disparity.cols, CV_8UC3);//构造RGB图像
#define IMG_B(mat_disparity,y,x) mat_disparity.at<Vec3b>(y,x)[0]
#define IMG_G(mat_disparity,y,x) mat_disparity.at<Vec3b>(y,x)[1]
#define IMG_R(mat_disparity,y,x) mat_disparity.at<Vec3b>(y,x)[2]
	uchar tmp2 = 0;
	for (int y = 0; y < mat_disparity.rows; y++)//转为彩虹图的具体算法，主要思路是把灰度图对应的0～255的数值分别转换成彩虹色：红、橙、黄、绿、青、蓝。
	{
		for (int x = 0; x < mat_disparity.cols; x++)
		{
			tmp2 = mat_disparity.at<uchar>(y, x);
			if (tmp2 <= 51)
			{
				IMG_B(img_color, y, x) = 255;
				IMG_G(img_color, y, x) = tmp2 * 5;
				IMG_R(img_color, y, x) = 0;
			}
			else if (tmp2 <= 102)
			{
				tmp2 -= 51;
				IMG_B(img_color, y, x) = 255 - tmp2 * 5;
				IMG_G(img_color, y, x) = 255;
				IMG_R(img_color, y, x) = 0;
			}
			else if (tmp2 <= 153)
			{
				tmp2 -= 102;
				IMG_B(img_color, y, x) = 0;
				IMG_G(img_color, y, x) = 255;
				IMG_R(img_color, y, x) = tmp2 * 5;
			}
			else if (tmp2 <= 204)
			{
				tmp2 -= 153;
				IMG_B(img_color, y, x) = 0;
				IMG_G(img_color, y, x) = 255 - uchar(128.0 * tmp2 / 51.0 + 0.5);
				IMG_R(img_color, y, x) = 255;
			}
			else
			{
				tmp2 -= 204;
				IMG_B(img_color, y, x) = 0;
				IMG_G(img_color, y, x) = 127 - uchar(127.0 * tmp2 / 51.0 + 0.5);
				IMG_R(img_color, y, x) = 255;
			}
		}
	}
	namedWindow("img_rainbowcolor", 0);
	imshow("img_rainbowcolor", img_color);

	Mat disparity1;
	mat_disparity.convertTo(disparity1, CV_32F, 1.0 / 16.0f);

	vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

	for (int v = 0; v < image_left.rows; v++)
		for (int u = 0; u < image_left.cols; u++) {
			if (disparity1.at<float>(v, u) <= 0.0 )
				continue;
			Vector4d point(0, 0, 0, image_left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色
			// 根据双目模型计算 point 的位置
			double x = (u - camera_cx) / camera_fx;
			double y = (v - camera_cy) / camera_fy;
			short d = disparity1.at<float>(v, u);
			double depth1 = camera_fx * baseline / d;
			point[0] = x * depth1;
			point[1] = y * depth1;
			point[2] = depth1;
			pointcloud.push_back(point);
		}


	// 画出点云
	showPointCloud(pointcloud);


    cv::waitKey(0);

    // 释放内存
    delete[] disparity;
    disparity = nullptr;
    delete[] bytes_left;
    bytes_left = nullptr;
    delete[] bytes_right;
    bytes_right = nullptr;

    system("pause");
    return 0;
}

