#include "EMat.h"

using namespace std;

namespace mc{

	bool EMat::isValid() const{
		return true;
	}
	// =演算子のオーバーロード
	EMat& EMat::operator=(const cv::Mat& mat){
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
		mat_ = mat;
		return *this;
	}

	// ファンクタ
	// 指定行、指定列の値をdoubleで返す
	double& EMat::operator()(unsigned int row, unsigned int col){
		return mat_.at<double>(row, col);
	}

	// cv::matの値をもとに生成する
	EMat::EMat(const cv::Mat& mat) : mat_(mat){
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
	}

	// ector< vector<string> >を受け取って、それらを要素とする行列を生成する
	EMat::EMat(const vector< vector<string> >& contents) : mat_(toMat(contents)){
		// TODO contentsが行列形式であることが保証されていない
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
	}

	cv::Mat& EMat::m(){return mat_;}
	const cv::Mat& EMat::m() const{
		const cv::Mat& retMat = mat_;
		return retMat;
	}

	cv::Mat EMat::toMat(const vector< vector<string> >& contents){// static
		cv::Mat mat(contents.size(), contents[0].size(), CV_64F);
		// ----------ここからcontentsの要素を代入する処理----------
		for (int i = 0; i < contents.size(); ++i){// 行のループ
			vector<string> row = contents[i];// 現在の行をrowに代入
			for (int j = 0; j < contents[0].size(); ++j){// 列のループ
				mat.at<double>(i, j) = boost::lexical_cast<double>(row[j]);
			}
		}// ----------ここまでcontentsの要素を代入する処理----------
		return mat;
	}

	// 要素がstringのvectorを返す
	vector< vector<string> > EMat::toVec() const{
		return toVec(mat_);
	}

	vector< vector<string> > EMat::toVec(const cv::Mat& mat){// static
		vector< vector<string> > vecs;
		for (int row_i = 0; row_i < mat.rows; ++row_i){
			vector<string> v;
			for(int col_i = 0; col_i < mat.cols; ++col_i){
				string str = boost::lexical_cast<string>(mat.at<double>(row_i, col_i)); 
				v.push_back(str);
			}
			vecs.push_back(v);
		}
		return vecs;
	}

	vector< vector< vector<string> > > EMat::toVec(const vector<cv::Mat>& mats){// static
		if(mats.size() == 0)	throw invalid_argument("mats num is zero");

		vector< vector< vector<string> > > retVec;
		for (int i = 0; i < mats.size(); ++i)	retVec.push_back(toVec(mats[i]));
		return retVec;
	}

	vector<cv::Mat> EMat::cast(const vector<EMat>& emats){// static
		vector<cv::Mat> mats;
		for (int i = 0; i < emats.size(); ++i)	mats.push_back(emats[i].m());
		return mats;
	}
}
