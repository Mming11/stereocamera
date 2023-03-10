#include "SGM_lab.h"
#include "auxiliary_fun.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <thread>
#include <omp.h>
#include <pangolin/pangolin.h>
#define USE_THREAD

using namespace std;
using namespace Eigen;

void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>>& pointcloud)
{
	pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
		pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
	);
	pangolin::View& d_cam = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));
	while (pangolin::ShouldQuit() == false)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		d_cam.Activate(s_cam);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glPointSize(2);
		glBegin(GL_POINTS);
		for (auto& p : pointcloud) {
			glColor3f(p[3], p[3], p[3]);
			glVertex3d(p[0], p[1], p[2]);
		}
		glEnd();
		pangolin::FinishFrame();
		// sleep 5 ms
	}

	return;
}

match_SGM::match_SGM() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
census_left_(nullptr), census_right_(nullptr),
cost_init_(nullptr), cost_aggr_(nullptr),
cost_aggr_1_(nullptr), cost_aggr_2_(nullptr),
cost_aggr_3_(nullptr), cost_aggr_4_(nullptr),
cost_aggr_5_(nullptr), cost_aggr_6_(nullptr),
cost_aggr_7_(nullptr), cost_aggr_8_(nullptr),
disp_left_(nullptr), disp_right_(nullptr),
is_initialized_(false)
{
}


match_SGM::~match_SGM()
{
    Release();
    is_initialized_ = false;
}

bool match_SGM::Initialize(const int32_t& width, const int32_t& height, const Parameter_Set& option)
{
	// ··· 赋值

	// 影像尺寸
	width_ = width;
	height_ = height;
	// SGM参数
	param1 = option;

	if (width == 0 || height == 0) {
		return false;
	}

	//··· 开辟内存空间

	// census值（左右影像）
	const int32_t img_size = width * height;

	census_left_ = new uint32_t[img_size]();
	census_right_ = new uint32_t[img_size]();

	// 视差范围
	const int32_t disp_range = option.max_disparity - option.min_disparity;
	if (disp_range <= 0) {
		return false;
	}

	// 匹配代价（初始/聚合）
	const int32_t size = width * height * disp_range;
	cost_init_ = new uint8_t[size]();
	cost_aggr_ = new uint16_t[size]();
	cost_aggr_1_ = new uint8_t[size]();
	cost_aggr_2_ = new uint8_t[size]();
	cost_aggr_3_ = new uint8_t[size]();
	cost_aggr_4_ = new uint8_t[size]();
	cost_aggr_5_ = new uint8_t[size]();
	cost_aggr_6_ = new uint8_t[size]();
	cost_aggr_7_ = new uint8_t[size]();
	cost_aggr_8_ = new uint8_t[size]();

	// 视差图
	disp_left_ = new float[img_size]();
	disp_right_ = new float[img_size]();

	is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

	return is_initialized_;
}


void match_SGM::Release()
{
    // 释放内存
    SAFE_DELETE(census_left_);
    SAFE_DELETE(census_right_);
    SAFE_DELETE(cost_init_);
    SAFE_DELETE(cost_aggr_);
    SAFE_DELETE(cost_aggr_1_);
    SAFE_DELETE(cost_aggr_2_);
    SAFE_DELETE(cost_aggr_3_);
    SAFE_DELETE(cost_aggr_4_);
    SAFE_DELETE(cost_aggr_5_);
    SAFE_DELETE(cost_aggr_6_);
    SAFE_DELETE(cost_aggr_7_);
    SAFE_DELETE(cost_aggr_8_);
    SAFE_DELETE(disp_left_);
    SAFE_DELETE(disp_right_);
}

bool match_SGM::Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left)
{
    if(!is_initialized_) {
        return false;
    }
    if (img_left == nullptr || img_right == nullptr) {
        return false;
    }
    img_left_ = img_left;
    img_right_ = img_right;

    // census变换
    CensusTransform();

    // 代价计算
    ComputeCost();
    // 代价聚合
    CostAggregation();
    // 视差计算
    ComputeDisparity();

    // 左右一致性检查
    if (param1.is_check_lr) {
        // 视差计算（右影像）
        disparitcompute_Right();
        // 一致性检查
        Consistency_check();
    }
    // 移除小连通区
    if (param1.is_remove_speckles) {
        RemoveSpeckles(disp_left_, width_, height_, 1, param1.min_speckle_aera, Invalid_Float);
		printf("完成小联通区的移除!\n");

    }

    // 视差填充
	if(param1.is_fill_holes) {
		FillBlankHoles();
		printf("完成视差填充!\n");
	}

    // 中值滤波
    Medianfiltering(disp_left_, disp_left_, width_, height_, 3);


    // 输出视差图
    memcpy(disp_left, disp_left_, height_ * width_ * sizeof(float));

	return true;
}

void match_SGM::CensusTransform() const
{


#ifdef USE_THREAD
	// 左右影像census变换
	std::thread t1(Census_Transform, img_left_, static_cast<uint32_t*>(census_left_), width_, height_);
	std::thread t2(Census_Transform, img_right_, static_cast<uint32_t*>(census_right_), width_, height_);
	t1.join();
	t2.join();
#else

	Census_Transform(img_left_, static_cast<uint32_t*>(census_left_), width_, height_);
	Census_Transform(img_right_, static_cast<uint32_t*>(census_left_), width_, height_);
#endif

}

void match_SGM::ComputeCost() const
{
	const int32_t& min_disparity = param1.min_disparity;
	const int32_t& max_disparity = param1.max_disparity;
	const int32_t disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}


#ifdef USE_THREAD
	// 计算代价（基于Hamming距离）
#pragma omp parallel for num_threads(2*omp_get_num_procs()-1)
	for (int32_t i = 0; i < height_; i++) {
		for (int32_t j = 0; j < width_; j++) {
			// 逐视差计算代价值
			for (int32_t d = min_disparity; d < max_disparity; d++) {
				auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
				if (j - d < 0 || j - d >= width_) {
					cost = UINT8_MAX / 2;
					continue;
				}

				// 左影像census值
				const auto& census_val_l = static_cast<uint32_t*>(census_left_)[i * width_ + j];
				// 右影像对应像点的census值
				const auto& census_val_r = static_cast<uint32_t*>(census_right_)[i * width_ + j - d];
				// 计算匹配代价
				cost = compute_hamming(census_val_l, census_val_r);

			}
		}
	}
	

#else
	for (int32_t i = 0; i < height_; i++) {
		for (int32_t j = 0; j < width_; j++) {
			// 逐视差计算代价值
			for (int32_t d = min_disparity; d < max_disparity; d++) {
				auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
				if (j - d < 0 || j - d >= width_) {
					cost = UINT8_MAX / 2;
					continue;
				}

				// 左影像census值
				const auto& census_val_l = static_cast<uint32_t*>(census_left_)[i * width_ + j];
				// 右影像对应像点的census值
				const auto& census_val_r = static_cast<uint32_t*>(census_right_)[i * width_ + j - d];
				// 计算匹配代价
				cost = compute_hamming(census_val_l, census_val_r);

			}
		}
	}

#endif

}

void match_SGM::CostAggregation() const
{
	const auto& min_disparity = param1.min_disparity;
	const auto& max_disparity = param1.max_disparity;
	assert(max_disparity > min_disparity);

	const int32_t size = width_ * height_ * (max_disparity - min_disparity);
	if (size <= 0) {
		return;
	}

	const auto& P1 = param1.p1;
	const auto& P2_Int = param1.p2_init;

#ifdef USE_THREAD
	std::vector<std::thread> threads;
	
		// 左右聚合
		std::thread t1(horizontalcostaggregate,img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
		std::thread t2(horizontalcostaggregate, img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);

		// 上下聚合
		std::thread t3(verticalcostaggregate,img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
		std::thread t4(verticalcostaggregate, img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);

		threads.push_back(std::move(t1));
		threads.push_back(std::move(t2));
		threads.push_back(std::move(t3));
		threads.push_back(std::move(t4));

	if (param1.num_paths == 8) {
		// 对角线1聚合
		std::thread t1(dagonalcostaggregate_one,img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_5_, true);
		std::thread t2(dagonalcostaggregate_one, img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_6_, false);
		// 对角线2聚合
		std::thread t3(dagonalcostaggregate_two,img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_7_, true);
		std::thread t4(dagonalcostaggregate_two, img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_8_, false);

		threads.push_back(std::move(t1));
		threads.push_back(std::move(t2));
		threads.push_back(std::move(t3));
		threads.push_back(std::move(t4));
	}

	for (auto& thread : threads) {
		thread.join();
	}

#pragma omp parallel for num_threads(2*omp_get_num_procs()-1)
	// 把4/8个方向加起来
	for (int32_t i = 0; i < size; i++) {
		if (param1.num_paths == 4 || param1.num_paths == 8) {
			cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
		}
		if (param1.num_paths == 8) {
			cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
		}
	}
#else
	// 左右聚合
	horizontalcostaggregate(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
	horizontalcostaggregate(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);
	// 上下聚合
	verticalcostaggregate(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
	verticalcostaggregate(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);

	if (param1.num_paths == 8) {
		// 对角线1聚合
		dagonalcostaggregate_one(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_5_, true);
		dagonalcostaggregate_one(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_6_, false);
		// 对角线2聚合
		dagonalcostaggregate_two(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_7_, true);
		dagonalcostaggregate_two(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_8_, false);
	}

	// 把4/8个方向加起来
	for (int32_t i = 0; i < size; i++) {
		if (param1.num_paths == 4 || param1.num_paths == 8) {
			cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
		}
		if (param1.num_paths == 8) {
			cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
		}
	}
#endif

}

void match_SGM::ComputeDisparity() const
{
    const int32_t& min_disparity = param1.min_disparity;
    const int32_t& max_disparity = param1.max_disparity;
    const int32_t disp_range = max_disparity - min_disparity;
    if(disp_range <= 0) {
        return;
    }

    // 左影像视差图
    const auto disparity = disp_left_;
	// 左影像聚合代价数组
	const auto cost_ptr = cost_aggr_;

    const int32_t width = width_;
    const int32_t height = height_;
    const bool is_check_unique = param1.is_check_unique;
	const float uniqueness_ratio = param1.uniqueness_ratio;

	// 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
    std::vector<uint16_t> cost_local(disp_range);
    
	// ---逐像素计算最优视差
	for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            uint16_t min_cost =UINT16_MAX;
            uint16_t sec_min_cost =UINT16_MAX;
            int32_t best_disparity = 0;

            // ---遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
            for (int32_t d = min_disparity; d < max_disparity; d++) {
	            const int32_t d_idx = d - min_disparity;
                const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + j * disp_range + d_idx];
                if(min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
            }

            if (is_check_unique) {
                // 再遍历一次，输出次最小代价值
                for (int32_t d = min_disparity; d < max_disparity; d++) {
                    if (d == best_disparity) {
                        // 跳过最小代价值
                        continue;
                    }
                    const auto& cost = cost_local[d - min_disparity];
                    sec_min_cost = std::min(sec_min_cost, cost);
                }

                // 判断唯一性约束
                // 若(min-sec)/min < min*(1-uniquness)，则为无效估计
                if (sec_min_cost - min_cost <= static_cast<uint16_t>(min_cost * (1 - uniqueness_ratio))) {
                    disparity[i * width + j] = Invalid_Float;
                    continue;
                }
            }

            // ---子像素拟合
            if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
                disparity[i * width + j] = Invalid_Float;
                continue;
            }
            // 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
            const int32_t idx_1 = best_disparity - 1 - min_disparity;
            const int32_t idx_2 = best_disparity + 1 - min_disparity;
            const uint16_t cost_1 = cost_local[idx_1];
            const uint16_t cost_2 = cost_local[idx_2];
            // 解一元二次曲线极值
            const uint16_t denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
            disparity[i * width + j] = static_cast<float>(best_disparity) + static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
        }
    }
}

void match_SGM::disparitcompute_Right() const
{
    const int32_t& min_disparity = param1.min_disparity;
    const int32_t& max_disparity = param1.max_disparity;
    const int32_t disp_range = max_disparity - min_disparity;
    if (disp_range <= 0) {
        return;
    }

    // 右影像视差图
    const auto disparity = disp_right_;
    // 左影像聚合代价数组
	const auto cost_ptr = cost_aggr_;

    const int32_t width = width_;
    const int32_t height = height_;
    const bool is_check_unique = param1.is_check_unique;
    const float uniqueness_ratio = param1.uniqueness_ratio;

    // 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
    std::vector<uint16_t> cost_local(disp_range);

    // ---逐像素计算最优视差
    // 通过左影像的代价，获取右影像的代价
    // 右cost(xr,yr,d) = 左cost(xr+d,yl,d)
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            uint16_t min_cost =UINT16_MAX;
            uint16_t sec_min_cost =UINT16_MAX;
            int32_t best_disparity = 0;

            // ---统计候选视差下的代价值
        	for (int32_t d = min_disparity; d < max_disparity; d++) {
                const int32_t d_idx = d - min_disparity;
        		const int32_t col_left = j + d;
        		if (col_left >= 0 && col_left < width) {
                    const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + col_left * disp_range + d_idx];
                    if (min_cost > cost) {
                        min_cost = cost;
                        best_disparity = d;
                    }
        		}
                else {
                    cost_local[d_idx] =UINT16_MAX;
                }
            }

            if (is_check_unique) {
                // 再遍历一次，输出次最小代价值
                for (int32_t d = min_disparity; d < max_disparity; d++) {
                    if (d == best_disparity) {
                        // 跳过最小代价值
                        continue;
                    }
                    const auto& cost = cost_local[d - min_disparity];
                    sec_min_cost = std::min(sec_min_cost, cost);
                }

                // 判断唯一性约束
                // 若(min-sec)/min < min*(1-uniquness)，则为无效估计
                if (sec_min_cost - min_cost <= static_cast<uint16_t>(min_cost * (1 - uniqueness_ratio))) {
                    disparity[i * width + j] = Invalid_Float;
                    continue;
                }
            }
            
            // ---子像素拟合
            if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
                disparity[i * width + j] = Invalid_Float;
                continue;
            }

            // 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
            const int32_t idx_1 = best_disparity - 1 - min_disparity;
            const int32_t idx_2 = best_disparity + 1 - min_disparity;
            const uint16_t cost_1 = cost_local[idx_1];
            const uint16_t cost_2 = cost_local[idx_2];
            // 解一元二次曲线极值
            const uint16_t denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
            disparity[i * width + j] = static_cast<float>(best_disparity) + static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
        }
    }
}

void match_SGM::Consistency_check()
{
    const int32_t width = width_;
    const int32_t height = height_;

    const float& threshold = param1.lrcheck_thres;

	// 遮挡区像素和误匹配区像素
	auto& occlusions = occlusions_;
	auto& mismatches = mismatches_;
	occlusions.clear();
	mismatches.clear();

    // ---左右一致性检查
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            // 左影像视差值
        	auto& disp = disp_left_[i * width + j];
			if(disp == Invalid_Float){
				mismatches.emplace_back(i, j);
				continue;
			}

            // 根据视差值找到右影像上对应的同名像素
        	const auto col_right = static_cast<int32_t>(j - disp + 0.5);
            
        	if(col_right >= 0 && col_right < width) {
                // 右影像上同名像素的视差值
                const auto& disp_r = disp_right_[i * width + col_right];
                
        		// 判断两个视差值是否一致（差值在阈值内）
        		if (abs(disp - disp_r) > threshold) {
					// 区分遮挡区和误匹配区
					// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
					// if(disp_rl > disp) 
        			//		pixel in occlusions
					// else 
        			//		pixel in mismatches
					const int32_t col_rl = static_cast<int32_t>(col_right + disp_r + 0.5);
					if(col_rl > 0 && col_rl < width){
						const auto& disp_l = disp_left_[i*width + col_rl];
						if(disp_l > disp) {
							occlusions.emplace_back(i, j);
						}
						else {
							mismatches.emplace_back(i, j);
						}
					}
					else{
						mismatches.emplace_back(i, j);
					}

                    // 让视差值无效
					disp = Invalid_Float;
                }
            }
            else{
                // 通过视差值在右影像上找不到同名像素（超出影像范围）
                disp = Invalid_Float;
				mismatches.emplace_back(i, j);
            }
        }
    }

}

void match_SGM::FillBlankHoles()
{
	const int32_t width = width_;
	const int32_t height = height_;

	std::vector<float> disp_collects;

	// 定义8个方向
	float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float *angle = angle1;
    // 最大搜索行程，没有必要搜索过远的像素
    const int32_t max_search_length = 1.0*std::max(abs(param1.max_disparity), abs(param1.min_disparity));

	float* disp_ptr = disp_left_;
	for (int32_t k = 0; k < 3; k++) {
		// 第一次循环处理遮挡区，第二次循环处理误匹配区
		auto& trg_pixels = (k == 0) ? occlusions_ : mismatches_;
        if (trg_pixels.empty()) {
            continue;
        }
		std::vector<float> fill_disps(trg_pixels.size());
		std::vector<std::pair<int32_t, int32_t>> inv_pixels;
		if (k == 2) {
			//  第三次循环处理前两次没有处理干净的像素
			for (int32_t i = 0; i < height; i++) {
				for (int32_t j = 0; j < width; j++) {
					if (disp_ptr[i * width + j] == Invalid_Float) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// 遍历待处理像素
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto& pix = trg_pixels[n];
            const int32_t y = pix.first;
            const int32_t x = pix.second;

			if (y == height / 2) {
				angle = angle2; 
			}

			// 收集8个方向上遇到的首个有效视差值
			disp_collects.clear();
			for (int32_t s = 0; s < 8; s++) {
				const float ang = angle[s];
				const float sina = float(sin(ang));
				const float cosa = float(cos(ang));
				for (int32_t m = 1; m < max_search_length; m++) {
					const int32_t yy = lround(y + m * sina);
					const int32_t xx = lround(x + m * cosa);
					if (yy<0 || yy >= height || xx<0 || xx >= width) {
						break;
					}
					const auto& disp = *(disp_ptr + yy*width + xx);
					if (disp != Invalid_Float) {
						disp_collects.push_back(disp);
						break;
					}
				}
			}
			if(disp_collects.empty()) {
				continue;
			}

			std::sort(disp_collects.begin(), disp_collects.end());

			// 如果是遮挡区，则选择第二小的视差值
			// 如果是误匹配区，则选择中值
			if (k == 0) {
				if (disp_collects.size() > 1) {
                    fill_disps[n] = disp_collects[1];
				}
				else {
                    fill_disps[n] = disp_collects[0];
				}
			}
			else{
                fill_disps[n] = disp_collects[disp_collects.size() / 2];
			}
		}
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto& pix = trg_pixels[n];
            const int32_t y = pix.first;
            const int32_t x = pix.second;
            disp_ptr[y * width + x] = fill_disps[n];
        }
	}
}
