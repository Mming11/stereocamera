#pragma once
#include <vector>
#include <Eigen/Core>
#include <pangolin/pangolin.h>

//实现3d点云时调用该库
using namespace Eigen;
using namespace std;


//将SGM参数定为结构体，方便后续操作
//其中bool型数据可进行选择是否进行该项操作
struct Parameter_Set {
	uint8_t	num_paths;			// 聚合路径数
	int32_t  min_disparity;		// 最小视差
	int32_t	max_disparity;		// 最大视差

	bool	is_check_unique;	// 是否选择检查唯一性
	float	uniqueness_ratio;	// 唯一性约束阈值

	bool	is_check_lr;		// 是否选择检查左右一致性
	float	lrcheck_thres;		// 左右一致性约束阈值

	bool	is_remove_speckles;	// 是否选择移除小的连通区
	int		min_speckle_aera;	// 最小的连通区面积（像素数）

	bool	is_fill_holes;		// 是否选择填充视差空洞

	int32_t  p1;				// 惩罚项参数P1
	int32_t  p2_init;		    // 惩罚项参数P2

};
// 定义SGM类
class match_SGM
{
private:
	// Census变换
	void CensusTransform() const;

	//代价计算
	void ComputeCost() const;
	// 代价聚合	 
	void CostAggregation() const;
	// 视差计算	 
	void ComputeDisparity() const;
	// 视差计算	 
	void disparitcompute_Right() const;
	// 一致性检查	 
	void Consistency_check();
	// 视差图填充 
	void FillBlankHoles();

	// 内存释放	 
	void Release();

	// 初始化SGM参数实例	 
	Parameter_Set param1;
	// 影像宽	 
	int32_t width_;
	// 影像高	 
	int32_t height_;
	// 左影像数据	 
	const uint8_t* img_left_;
	// 右影像数据	 
	const uint8_t* img_right_;
	// 左影像census值	 
	void* census_left_;
	// 右影像census值	 
	void* census_right_;
	// 初始匹配代价
	uint8_t* cost_init_;
	// 聚合匹配代价	
	uint16_t* cost_aggr_;
	//定义聚合匹配代价-方向 考虑八个方向
	uint8_t* cost_aggr_1_;
	uint8_t* cost_aggr_2_;
	uint8_t* cost_aggr_3_;
	uint8_t* cost_aggr_4_;
	uint8_t* cost_aggr_5_;
	uint8_t* cost_aggr_6_;
	uint8_t* cost_aggr_7_;
	uint8_t* cost_aggr_8_;

	// 左影像视差图	
	float* disp_left_;
	// 右影像视差图	
	float* disp_right_;

	// 是否初始化标志	
	bool is_initialized_;

	// 遮挡区像素集	
	std::vector<std::pair<int, int>> occlusions_;
	// 误匹配区像素集	
	std::vector<std::pair<int, int>> mismatches_;

public:
	match_SGM();
	~match_SGM();

	// 完成类的初始化，SGM参数的设置
	bool Initialize(const int32_t& width, const int32_t& height, const Parameter_Set& option);

    // 主要实现函数，进行census变化，代价计算等操作
	bool Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left);
};

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>>& pointcloud);