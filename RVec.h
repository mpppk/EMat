#pragma once
// #include <iostream>
// #include <string>
#include <opencv2/opencv.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;

namespace mc{
	class RVec{
	private:
		bool isValid() const;
		string getErrorMessage() const;
		cv::Mat mat_;
	public:
		RVec(){}
		RVec(const unsigned int size);
		RVec(const cv::Mat& mat);
		RVec(const cv::MatExpr& mate);
		// RVec(const cv::Mat& mat) const;
		RVec(const vector<string>& content);
		RVec(const vector<double>& content);

		// ---------- 演算子オーバーロード ----------
		/** cv::Matを=で代入できるようにするための演算子オーバーロード **/
		RVec& operator=(const cv::Mat& mat);
		double& operator[](unsigned int index);
		double operator[](unsigned int index) const;
		RVec& operator+(double n);
		RVec& operator+=(double n);
		RVec& operator-(double n);
		RVec& operator-=(double n);
		RVec& operator*(double n);
		RVec& operator*=(double n);
		RVec& operator/(double n);
		RVec& operator/=(double n);

		int size() const;
		cv::Mat& m();
		const cv::Mat& m() const;
		static vector<RVec> cast(vector<cv::Mat> mats);
		static cv::Mat toMat(const vector<string>& content);
		static cv::Mat toMat(const vector<double>& content);
	};

	// ostream& operator<< (ostream& os, const mc::RVec& rvec) {
	// 	os << rvec.m();
	// 	return os;
	// }
}
