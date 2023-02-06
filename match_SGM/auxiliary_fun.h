#pragma once
#include <cstdint>
#include <limits>

constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();
const float pi = 3.1415926f;

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

/********************************************************
*	@ �洢����ͼ�����Ż��Ĺ��ߺ���
********************************************************/

// census�任
void Census_Transform(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height);
// Hamming����
uint8_t compute_hamming(const uint32_t& x, const uint32_t& y);

// ���·���ۺϴ������ݣ����ң�
void horizontalcostaggregate(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
 // ���·���ۺϴ������ݣ����£�
void verticalcostaggregate(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
// ���Խ��߷���·���ۺϴ������ݣ�б���������£�
void dagonalcostaggregate_one(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
// ���Խ��߷���·���ۺϴ������ݣ�б���������ϣ�
void dagonalcostaggregate_two(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

//��ֵ�˲�
void Medianfiltering(const float* in, float* out, const int32_t& width, const int32_t& height, const int32_t wnd_size);

//�޳�С��ͨ��
void RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame, const uint32_t& min_speckle_aera, const float& invalid_val);
