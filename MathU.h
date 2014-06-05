#pragma once
#include <iostream>
#include <cmath>
#include <cfloat>
#include <math.h>
#include <map>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <RVec.h>

using namespace std;

namespace mc{
	class MathU{
	public:
		enum CovMatOptions{
			SCALE,
			CovMatOptionsNum
		};

		// 重み付の共分散を計算する
		static double calcWCov(const cv::Mat &vec1, const cv::Mat &vec2,
		 double mean1, double mean2, const cv::Mat &vec_weight);

		// 重み付の共分散行列を計算する
		// dataMat 1行に一つの事例を格納
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_mean, const cv::Mat &vec_weight);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight, const bitset<CovMatOptionsNum> flags);

		//L1normを計算する
		static double calcL1norm(const cv::Mat &vec);
		static double calcL2norm(const cv::Mat &vec);
		static double calcTotalSum(const cv::Mat &vec);

		// double rbfKernel(CvMat *X1, CvMat *X2, double rbf_sigma);
		static double rbfKernel(const cv::Mat &vec1, const cv::Mat &vec2, const double rbfSigma);
		// 引数:
		// vec1 -> カーネル関数の引数
		// vec2 -> カーネル関数の引数
		// rbfSigma -> rbfカーネルの計算に用いるσの値
		// 
		static cv::Mat plotEllipse(double x,double y,double a,double b,int rot);

		static cv::Mat plotConfidenceEllipse(const cv::Mat &mean, const cv::Mat &covMatInv, const int sigma);

		static map<string, cv::Mat> normalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd);

		static map<string, cv::Mat> normalize(const cv::Mat &data);

		static map<string, cv::Mat> unnormalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd);

		static cv::Mat calcSteepestDescent(const cv::Mat &vec_initialValue, const cv::Mat &vec_objectiveValiable, const cv::Mat &ASY, const double initStepSize, const double tolerance);

		static cv::Mat calcHillClimbing(const cv::Mat &vec_initialValue, const cv::Mat &vec_objectiveValiable, const cv::Mat &ASY, const double initStepSize, const double tolerance, const int maxIt = 100000, const double alpha = 1);

		static cv::Mat normalizeHistogram(const cv::Mat &src);

		static double calcEuclideanDist(const cv::Mat point1, const cv::Mat point2);
	};
}

