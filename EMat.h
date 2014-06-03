// ExtendMat
// cv::Matを拡張するクラス

#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace mc{

	/**
	 * @class EMat a b cv::Matを継承し、機能を拡張したクラス
	 */

	// cv::Matを拡張したクラス(Extended Mat)
	class EMat : public cv::Mat{
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

	public:

		// ---------- 演算子オーバーロード ----------
		/**
		 * cv::Matを=で代入できるようにするための演算子オーバーロード
		 */
		EMat& operator=(cv::Mat mat);

		/**
		 * 指定行、指定列の値をdoubleで返すためのファンクタ
		 * @param  row 行のインデックス
		 * @param  col 列のインデックス
		 * @return 指定した要素の値
		 */
		double& operator()(unsigned int row, unsigned int col);
		// ---------- ここまで演算子オーバーロード ----------

		// ---------- コンストラクタ ----------
		EMat();

		// 指定したサイズ，型の行列を作成します．
		// (_type is CV_8UC1, CV_64FC3, CV_32SC(12) など)
		/**
		 * 指定したサイズ，型の行列を作成する
		 */
		EMat(int _rows, int _cols, int _type);

		// 受け取ったmatの値のEMatを作る
		EMat(const cv::Mat mat);

		EMat(const vector< vector<string> > contents);
		// ---------- ここまでコンストラクタ ----------

		// ---------- cast系 ----------
		// cv::Matをmc::EMatにキャストする
		static EMat cast(cv::Mat mat);
		static vector<cv::Mat> cast(vector<EMat> emats);
		// ---------- ここまでcast系 ----------

		// ---------- to系 ----------
		/**
		 * 要素をstring型にしてvectorを返す
		 * @return EMatが保持している要素
		 */
		vector< vector<string> > toVec() const;
		static vector< vector<string> > toVec(cv::Mat mat);
		static vector< vector< vector<string> > > toVec(vector<cv::Mat> mats);

		/**
		 * 行列を正規化(平均ゼロ、標準偏差１)する
		 * @remarks 計算結果は保持している行列を変更するのではなく、paramsに格納される
		 */
		void toNormalizedMat();// 未実装

		/**
		 * 行列を正規化(平均ゼロ、標準偏差１)する.平均と分散が分かっている場合にはこちらを利用することで高速化できる
		 * @remarks 計算結果は保持している行列を変更するのではなく、paramsに格納される
		 * @param mean 平均
		 * @param sd 標準偏差		
		 */
		void toNormalizedMat(cv::Mat mean, cv::Mat sd);// 未実装

		/**
		 * 正規化されている行列を正規化前に戻す
		 * @remarks unnormalizeする行列は予め必要なサイズを初期化時に確保しておく必要がある.
		 * @remarks 計算結果は保持している行列を変更するのではなく、paramsに格納される
		 * @param mean 正規化前の平均
		 * @param sd 正規化前の標準偏差
		 */
		void toUnnormalizedMat(cv::Mat mean, cv::Mat sd);// 未実装
		// ---------- ここまでto系 ----------

		// double sum();


	};
}
