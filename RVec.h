#pragma once
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace mc{
	class RVec{
	private:
		bool isValid() const;
		cv::Mat mat_;
	public:
		RVec(){}
		RVec(const cv::Mat& mat);
		RVec(const vector<string>& content);
		// ---------- 演算子オーバーロード ----------
		/** cv::Matを=で代入できるようにするための演算子オーバーロード **/
		RVec& operator=(const cv::Mat& mat);
		double& operator[](unsigned int index);

		cv::Mat& m();

		cv::Mat toMat(const vector<string>& content);
	};
}
