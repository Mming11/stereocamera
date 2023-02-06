#pragma once
#include <vector>
#include <Eigen/Core>
#include <pangolin/pangolin.h>

//ʵ��3d����ʱ���øÿ�
using namespace Eigen;
using namespace std;


//��SGM������Ϊ�ṹ�壬�����������
//����bool�����ݿɽ���ѡ���Ƿ���и������
struct Parameter_Set {
	uint8_t	num_paths;			// �ۺ�·����
	int32_t  min_disparity;		// ��С�Ӳ�
	int32_t	max_disparity;		// ����Ӳ�

	bool	is_check_unique;	// �Ƿ�ѡ����Ψһ��
	float	uniqueness_ratio;	// Ψһ��Լ����ֵ

	bool	is_check_lr;		// �Ƿ�ѡ��������һ����
	float	lrcheck_thres;		// ����һ����Լ����ֵ

	bool	is_remove_speckles;	// �Ƿ�ѡ���Ƴ�С����ͨ��
	int		min_speckle_aera;	// ��С����ͨ���������������

	bool	is_fill_holes;		// �Ƿ�ѡ������Ӳ�ն�

	int32_t  p1;				// �ͷ������P1
	int32_t  p2_init;		    // �ͷ������P2

};
// ����SGM��
class match_SGM
{
private:
	// Census�任
	void CensusTransform() const;

	//���ۼ���
	void ComputeCost() const;
	// ���۾ۺ�	 
	void CostAggregation() const;
	// �Ӳ����	 
	void ComputeDisparity() const;
	// �Ӳ����	 
	void disparitcompute_Right() const;
	// һ���Լ��	 
	void Consistency_check();
	// �Ӳ�ͼ��� 
	void FillBlankHoles();

	// �ڴ��ͷ�	 
	void Release();

	// ��ʼ��SGM����ʵ��	 
	Parameter_Set param1;
	// Ӱ���	 
	int32_t width_;
	// Ӱ���	 
	int32_t height_;
	// ��Ӱ������	 
	const uint8_t* img_left_;
	// ��Ӱ������	 
	const uint8_t* img_right_;
	// ��Ӱ��censusֵ	 
	void* census_left_;
	// ��Ӱ��censusֵ	 
	void* census_right_;
	// ��ʼƥ�����
	uint8_t* cost_init_;
	// �ۺ�ƥ�����	
	uint16_t* cost_aggr_;
	//����ۺ�ƥ�����-���� ���ǰ˸�����
	uint8_t* cost_aggr_1_;
	uint8_t* cost_aggr_2_;
	uint8_t* cost_aggr_3_;
	uint8_t* cost_aggr_4_;
	uint8_t* cost_aggr_5_;
	uint8_t* cost_aggr_6_;
	uint8_t* cost_aggr_7_;
	uint8_t* cost_aggr_8_;

	// ��Ӱ���Ӳ�ͼ	
	float* disp_left_;
	// ��Ӱ���Ӳ�ͼ	
	float* disp_right_;

	// �Ƿ��ʼ����־	
	bool is_initialized_;

	// �ڵ������ؼ�	
	std::vector<std::pair<int, int>> occlusions_;
	// ��ƥ�������ؼ�	
	std::vector<std::pair<int, int>> mismatches_;

public:
	match_SGM();
	~match_SGM();

	// �����ĳ�ʼ����SGM����������
	bool Initialize(const int32_t& width, const int32_t& height, const Parameter_Set& option);

    // ��Ҫʵ�ֺ���������census�仯�����ۼ���Ȳ���
	bool Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left);
};

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>>& pointcloud);