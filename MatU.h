// MatUtil
// Matを扱うstaticな関数を持つ

#pragma once
#include <RVec.h>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace mc{

	/**
	 * @class EMat a b cv::Matを継承し、機能を拡張したクラス
	 */

	// cv::Matを拡張したクラス(Extended Mat)
	class MatU{
	public:
		/**
		 * 要素をstring型にしてvectorを返す
		 * @return EMatが保持している要素
		 */
		static vector< vector<string> > toVec(const cv::Mat& mat);
		static vector< vector< vector<string> > > toVec(const vector<cv::Mat>& mats);
		static cv::Mat toMat(const vector<string>& content);
		static RVec toRVec(const vector<string>& content);
		static cv::Mat toMat(const vector< vector<string> >& contents);

		//srcの指定した行からdstの指定した行へをコピーする
		static void copyRow(const cv::Mat &src, cv::Mat &dst, const unsigned int srcRowIndex, unsigned int dstRowIndex);

		//指定した行をsrcからdstへコピーする
		static void copyRow(const cv::Mat &src, cv::Mat &dst, unsigned int row);

		//srcの指定した列からdstの指定した列へをコピーする
		static void copyCol(const cv::Mat &src, cv::Mat &dst, const unsigned int srcColsIndex, const unsigned int dstColIndex);

		static void copyCol(const cv::Mat &src, cv::Mat &dst, const unsigned int row);

		// vec1の後ろにvec2を連結する(vec1,vec2はそれぞれ横ベクトル)
		static cv::Mat mergeRVec(const cv::Mat &vec1, const cv::Mat &vec2);
		static RVec mergeRVec(const RVec &vec1, const RVec &vec2);

		// vector内の横ベクトルをすべて連結する
		static cv::Mat mergeRVec(const vector<cv::Mat> &vecs);
		static RVec mergeRVec(const vector<RVec> &vecs);

		// 横ベクトルにする
		static cv::Mat mergeRVec(const cv::Mat &vecs);

		// mat1の右側にmat2を連結する
		static cv::Mat mergeMatToSide(const cv::Mat &mat1, const cv::Mat &mat2, const int fillBlankNum = 0);

		static cv::Mat mergeMatToBottom(const cv::Mat &mat1, const cv::Mat &mat2, const int fillBlankNum = 0);

		static cv::Mat mergeMatToBottom(const vector<cv::Mat> mats, const int fillBlankNum = 0);
	};
}
