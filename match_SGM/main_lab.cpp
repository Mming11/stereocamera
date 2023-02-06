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

//ʵ��3d����ʱ���øÿ�
using namespace Eigen;
using namespace std;
using namespace cv;




int main(int argv, char** argc)
{

    // ��ʼ��·��
	string img_path_left = "../Data/Classic/image1.png";
	string img_path_right = "../Data/Classic/image2.png";

	// ��ʼ��Mat���Ԫ��
	cv::Mat disp_color;

    cv::Mat image_left_color = cv::imread(img_path_left, cv::IMREAD_COLOR);

	//�ҶȾ���ת��
    cv::Mat image_left = cv::imread(img_path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat image_right = cv::imread(img_path_right, cv::IMREAD_GRAYSCALE);

	// ���д����� �ж϶�ȡ�������Ƿ������ĿҪ��
    if (image_left.data == nullptr || image_right.data == nullptr) {
        std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
        return -1;
    }
    if (image_left.rows != image_right.rows || image_left.cols != image_right.cols) {
        std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
        return -1;
    }

	//��ȡͼƬ�ߴ�
    int32_t width = image_left.cols;
    int32_t height = image_right.rows;
	//�洢�Ҷ�����
	uint8_t* bytes_left = new uint8_t[width * height];
	uint8_t* bytes_right = new uint8_t[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = image_left.at<uint8_t>(i, j);
            bytes_right[i * width + j] = image_right.at<uint8_t>(i, j);
        }
    }

    printf("���ڴ���ת������...���!\n");

	// ����ƥ�����
	/********************SGM*************************************/
	// ��SGMʵ��˼����Դ����ӭ��csdn���͵�SGMʵ��
	// ����ο���������SGM�����࣬����ȡ�ຯ����match������һ���Լ�顢Ψһ��Լ�����޳�С��ͨ�����ɴ���

    Parameter_Set Parameter_Setting;

    // �ۺ�·����
    Parameter_Setting.num_paths = 8;
    // ��ѡ�ӲΧ
    Parameter_Setting.min_disparity = argv < 4 ? 0 : atoi(argc[3]);
    Parameter_Setting.max_disparity = argv < 5 ? 64 : atoi(argc[4]);
    // һ���Լ��
    Parameter_Setting.is_check_lr = true;
    Parameter_Setting.lrcheck_thres = 1.0f;
    // Ψһ��Լ��
    Parameter_Setting.is_check_unique = true;
    Parameter_Setting.uniqueness_ratio = 0.99;
    // �޳�С��ͨ��
	Parameter_Setting.is_remove_speckles = true;
    Parameter_Setting.min_speckle_aera = 50;
    // �ͷ���P1��P2
    Parameter_Setting.p1 = 10;
    Parameter_Setting.p2_init = 150;
    // �Ӳ�ͼ���
    Parameter_Setting.is_fill_holes = true;

    printf("ͼƬ��� = %d, ͼƬ�߶� = %d\n�ӲΧ = [%d,%d]\n\n", width, height, Parameter_Setting.min_disparity, Parameter_Setting.max_disparity);

    // ����SGMƥ����
    match_SGM sgm;

   
    // ��ʼ��
	printf("SGM��ʼ����...\n");
    if (!sgm.Initialize(width, height, Parameter_Setting)) {
        std::cout << "SGM��ʼ��ʧ�ܣ�" << std::endl;
        return -2;
    }
 
    // ƥ��
	printf("���ڽ���ƥ��...\n");

    // disparity���鱣�������ص��Ӳ���
    auto disparity = new float[uint32_t(width * height)]();
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGMƥ��ʧ�ܣ�" << std::endl;
        return -2;
    }

	printf("ƥ�����...\n");
   
	// ��ʾ�Ӳ�ͼ
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

	//���չʾ
    cv::imshow("�Ӳ�ͼ", mat_disparity);
    cv::imshow("�Ӳ�ͼ-α��", disp_color);


	double camera_fx = 718.856, camera_fy = 718.856, camera_cx = 607.1928, camera_cy = 185.2157;// �ڲ�
	double baseline = 0.573;// ����
	double fx = 4152.073;

	Mat depth(mat_disparity.rows, mat_disparity.cols, CV_16S);  //���ͼ

	//�Ӳ�ͼת���ͼ
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
	Mat img_pseudocolor(mat_disparity.rows, mat_disparity.cols, CV_8UC3);//����RGBͼ��

	int tmp = 0;
	for (int y = 0; y < mat_disparity.rows; y++)//תΪα��ɫͼ��ľ����㷨
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

	Mat img_color(mat_disparity.rows, mat_disparity.cols, CV_8UC3);//����RGBͼ��
#define IMG_B(mat_disparity,y,x) mat_disparity.at<Vec3b>(y,x)[0]
#define IMG_G(mat_disparity,y,x) mat_disparity.at<Vec3b>(y,x)[1]
#define IMG_R(mat_disparity,y,x) mat_disparity.at<Vec3b>(y,x)[2]
	uchar tmp2 = 0;
	for (int y = 0; y < mat_disparity.rows; y++)//תΪ�ʺ�ͼ�ľ����㷨����Ҫ˼·�ǰѻҶ�ͼ��Ӧ��0��255����ֵ�ֱ�ת���ɲʺ�ɫ���졢�ȡ��ơ��̡��ࡢ����
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
			Vector4d point(0, 0, 0, image_left.at<uchar>(v, u) / 255.0); // ǰ��άΪxyz,����άΪ��ɫ
			// ����˫Ŀģ�ͼ��� point ��λ��
			double x = (u - camera_cx) / camera_fx;
			double y = (v - camera_cy) / camera_fy;
			short d = disparity1.at<float>(v, u);
			double depth1 = camera_fx * baseline / d;
			point[0] = x * depth1;
			point[1] = y * depth1;
			point[2] = depth1;
			pointcloud.push_back(point);
		}


	// ��������
	showPointCloud(pointcloud);


    cv::waitKey(0);

    // �ͷ��ڴ�
    delete[] disparity;
    disparity = nullptr;
    delete[] bytes_left;
    bytes_left = nullptr;
    delete[] bytes_right;
    bytes_right = nullptr;

    system("pause");
    return 0;
}

