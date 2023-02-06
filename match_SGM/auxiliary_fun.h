#pragma once
#include <cstdint>
#include <limits>

constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();
const float pi = 3.1415926f;

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

/********************************************************
*	@ 存储用于图像处理及优化的工具函数
********************************************************/

// census变换
void Census_Transform(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height);
// Hamming距离
uint8_t compute_hamming(const uint32_t& x, const uint32_t& y);

// 求解路径聚合代价数据（左右）
void horizontalcostaggregate(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
 // 求解路径聚合代价数据（上下）
void verticalcostaggregate(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
// 求解对角线方向路径聚合代价数据（斜向左上右下）
void dagonalcostaggregate_one(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
// 求解对角线方向路径聚合代价数据（斜向左下右上）
void dagonalcostaggregate_two(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

//中值滤波
void Medianfiltering(const float* in, float* out, const int32_t& width, const int32_t& height, const int32_t wnd_size);

//剔除小连通区
void RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame, const uint32_t& min_speckle_aera, const float& invalid_val);
