#pragma once
// #include <iostream>
// #include <cmath>
// #include <cfloat>
// #include <math.h>
// #include <map>
// #include <iomanip>
// #include <opencv2/opencv.hpp>
#include <RVec.h>
#include <MatU.h>

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
		static double calcWCov(const RVec &vec1, const RVec &vec2,
			double mean1, double mean2, const RVec &vec_weight);


		// 重み付の共分散行列を計算する
		// dataMat 1行に一つの事例を格納
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_mean, const cv::Mat &vec_weight);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const RVec &vec_mean, const RVec &vec_weight);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const RVec &vec_weight);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight, const bitset<CovMatOptionsNum> flags);
		static cv::Mat calcWCovMat(const cv::Mat &dataMat, const RVec &vec_weight, const bitset<CovMatOptionsNum> flags);

		//L1normを計算する
		static double calcL1norm(const cv::Mat &vec);
		static double calcL1norm(const RVec &vec);
		static double calcL2norm(const cv::Mat &vec);
		static double calcL2norm(const RVec &vec);
		static double calcTotalSum(const cv::Mat &vec);
		static double calcTotalSum(const RVec &vec);

		// double rbfKernel(CvMat *X1, CvMat *X2, double rbf_sigma);
		static double rbfKernel(const cv::Mat &vec1, const cv::Mat &vec2, const double rbfSigma);
		static double rbfKernel(const RVec &vec1, const RVec &vec2, const double rbfSigma);
		// 引数:
		// vec1 -> カーネル関数の引数
		// vec2 -> カーネル関数の引数
		// rbfSigma -> rbfカーネルの計算に用いるσの値

		static cv::Mat plotEllipse(double x,double y,double a,double b,int rot);

		static cv::Mat plotConfidenceEllipse(const cv::Mat &mean, const cv::Mat &covMat, const int sigma);
		static cv::Mat plotConfidenceEllipse(const RVec &mean, const cv::Mat &covMat, const int sigma);

		static RVec toEachColsMean(const cv::Mat &data);
		static RVec toEachColsSquareMean(const cv::Mat &data);
		static RVec toEachColsSD(const cv::Mat &data);
		static RVec toEachColsSD(const cv::Mat &data, const RVec &mean);
		static RVec toEachColsSD(const cv::Mat &data, const RVec &mean, const RVec &squareMean);
		static RVec toEachColsVariance(const RVec sd);

		// 移動平均を計算する。着目要素からwidth-1個を計算に含める。最初の0〜(width-1)個の要素に対しては何もしない。
		static cv::Mat movingAverage(const cv::Mat &mat, const int width);
		// 行列の各列に対して移動平均を計算する
		static cv::Mat movingAverageToEachCol(const cv::Mat &mat, const int arg_width);

		// 時間解像度を変更する。(指定した区間ごとの平均値を計算した結果を返す)
		static map<string, cv::Mat> temporalResolution(const RVec &vec, const int width, const int xdim, const int ydim = 1);

		// １行が(index-dim-1)〜(index)までの値を持つ行列に変換する
		// 引数がRVecとcv::Matの場合で処理が違うので注意
		static cv::Mat toMultiDim(const RVec &vec, const int dim);
		static cv::Mat toMultiDim(const cv::Mat &mat, const int dim);

		static cv::Mat normalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd);
		static cv::Mat normalize(const cv::Mat &data, const RVec &mean, const RVec &sd);

		static map<string, cv::Mat> normalize(const cv::Mat &data);

		static cv::Mat unnormalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd);
		static cv::Mat unnormalize(const cv::Mat &data, const RVec &mean, const RVec &sd);

		static cv::Mat normalizeHistogram(const cv::Mat &src);

		static double calcEuclideanDist(const cv::Mat &point1, const cv::Mat &point2);
		static double calcEuclideanDist(const RVec &point1, const RVec &point2);

		// マハラノビス距離を計算する
		static double calcMahalanobisDist(const RVec &point1, const RVec &point2, const cv::Mat &icover);
	};
}

