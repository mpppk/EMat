// ExtendMat
// cv::Matを拡張するクラス

#pragma once
// #include <string>
// #include <iostream>
// #include <vector>
// #include <map>
// #include <opencv2/opencv.hpp>
#include <MatU.h>
#include <MathU.h>

using namespace std;

namespace mc{

	/**
	 * @class EMat a b cv::Matを継承し、機能を拡張したクラス
	 */

	// cv::Matを拡張したクラス(Extended Mat)
	class EMat : mc::RVec{
	private:
		/**
		 * valueがmin以上max以下ならtrueを返す
		 * @remarks ここで実装する必要ないような気がする
		 * @param min 下限
		 * @param max 上限
		 * @param value 判定対象の値
		 * @return 判定結果
		 */
		template<class T>
		bool isRangeWithin(const T min, const T max, const T value) const{
			return value >= min && value <= max;
		}
		cv::Mat mat_;
	protected:
		// 自身が不正な値を持っていないかチェックする
		// とりあえずはRVecなどで利用する為の実装
		bool isValid() const;

	public:
		// ---------- 演算子オーバーロード ----------
		/**
		 * cv::Matを=で代入できるようにするための演算子オーバーロード
		 */
		EMat& operator=(const cv::Mat& mat);

		/**
		 * 指定行、指定列の値をdoubleで返すためのファンクタ
		 * @param  row 行のインデックス
		 * @param  col 列のインデックス
		 * @return 指定した要素の値
		 */
		double& operator()(unsigned int row, unsigned int col);
		// ---------- ここまで演算子オーバーロード ----------

		// ---------- コンストラクタ ----------
		// 受け取ったmatの値のEMatを作る
		EMat(const cv::Mat& mat);

		EMat(const vector< vector<string> >& contents);
		// ---------- ここまでコンストラクタ ----------

		cv::Mat& m();
		const cv::Mat& m() const;

		RVec toEachColsMean() const;
		RVec toEachColsSD(const RVec &mean) const;
		RVec toEachColsSD() const;
		EMat toNormalizedMat() const;
		EMat toUnnormalizedMat(const RVec &mean, const RVec &sd) const;
		EMat toCovMat() const;
		EMat toWCovMat(const RVec &weight, const bitset<MathU::CovMatOptionsNum> flags) const;

		vector< vector<string> > toVec() const;
		static vector<cv::Mat> cast(const vector<EMat>& emats);

	};
}
