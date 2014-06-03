#pragma once
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include <EMat.h>

namespace mc{
	class RVec : public EMat{
	private:
		bool isValid() const;
		bool isValid(cv::Mat mat) const;

	public:

		// ---------- 演算子オーバーロード ----------
		/**
		 * cv::Matを=で代入できるようにするための演算子オーバーロード
		 */
		RVec& operator=(cv::Mat mat);
		double& operator[](unsigned int index);
		// ---------- コンストラクタ ----------
		RVec();

		// 指定したサイズ，型の行列を作成します．
		// (_type is CV_8UC1, CV_64FC3, CV_32SC(12) など)
		/**
		 * 指定したサイズ，型の行列を作成する
		 */
		RVec(int _rows, int _cols, int _type);

		// 受け取ったmatの値のRVecを作る
		RVec(const cv::Mat mat);

		RVec(const vector< vector<string> > contents);
		// ---------- ここまでコンストラクタ ----------

		static RVec cast(const cv::Mat mat);
	};
}
