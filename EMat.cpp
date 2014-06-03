#include "EMat.h"

using namespace std;

namespace mc{

	bool EMat::isValid() const{
		return true;
	}

	bool EMat::isValid(cv::Mat mat) const{
		return true;
	}

	// =演算子のオーバーロード
	EMat& EMat::operator=(cv::Mat mat){
		this->create(mat.rows, mat.cols, CV_64F);
		for (int row_i = 0; row_i < mat.rows; ++row_i){
			for (int col_i = 0; col_i < mat.cols; ++col_i){
				this->at<double>(row_i, col_i) = mat.at<double>(row_i, col_i);
			}
		}
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
		return *this;
	}

	// ファンクタ
	// 指定行、指定列の値をdoubleで返す
	double& EMat::operator()(unsigned int row, unsigned int col){
		return this->at<double>(row, col);
	}

	//引数なしのコンストラクタを定義しておく
	EMat::EMat(){}

	EMat::EMat(int _rows, int _cols, int _type) : cv::Mat(_rows, _cols, _type){
	}

	// cv::matの値をもとに生成する
	EMat::EMat(const cv::Mat mat){
		this->create(mat.rows, mat.cols, CV_64F);
		for (int row_i = 0; row_i < mat.rows; ++row_i){
			for (int col_i = 0; col_i < mat.cols; ++col_i){
				this->at<double>(row_i, col_i) = mat.at<double>(row_i, col_i);
			}
		}
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
	}

	// ector< vector<string> >を受け取って、それらを要素とする行列を生成する
	EMat::EMat(const vector< vector<string> > contents){
		// ----------ここから変数宣言----------
		int vecCol = contents[0].size();// 受け取ったvectorの列数
		// ----------ここまで変数宣言----------

		// vectorのサイズに合わせた行列を生成
		this->create(contents.size(), vecCol, CV_64FC1);

		// ----------ここからcontentsの要素を代入する処理----------
		for (int i = 0; i < contents.size(); ++i){// 行のループ
			vector<string> row = contents[i];// 現在の行をrowに代入
			for (int j = 0; j < vecCol; ++j){// 列のループ
				this->at<double>(i, j) = boost::lexical_cast<double>(row[j]);
			}
		}// ----------ここまでcontentsの要素を代入する処理----------
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
	}

	// 要素がstringのvectorを返す
	vector< vector<string> > EMat::toVec() const{
		return toVec(*this);
	}

	// static
	// 要素がstringのvectorを返す
	vector< vector<string> > EMat::toVec(cv::Mat mat){
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

	// static
	vector< vector< vector<string> > > EMat::toVec(vector<cv::Mat> mats){
		if(mats.size() == 0)	throw invalid_argument("mats num is zero");

		vector< vector< vector<string> > > retVec;
		for (int i = 0; i < mats.size(); ++i)	retVec.push_back(toVec(mats[i]));
		return retVec;
	}

	// void EMat::toNormalizedMat(cv::Mat mean, cv::Mat sd){
	// 	cv::Mat tempMat;
	// 	tempMat = this->clone();
	// 	map<string, cv::Mat> tempParams = mc::normalize(tempMat, mean, sd);
	// 	params_["mean"] = tempParams["mean"];
	// 	params_["variance"] = tempParams["variance"];
	// 	params_["sd"] = tempParams["sd"];
	// 	params_["normalizedMat"] = tempParams["normalizedMat"];
	// 	params_["n"] = tempParams["normalizedMat"];
	// 	params_["unnormalizedMat"] = tempMat;
	// 	params_["un"] = tempMat;

	// }

	// void EMat::toNormalizedMat(){
	// 	cv::Mat tempMat;
	// 	tempMat = this->clone();
	// 	map<string, cv::Mat> tempParams = mc::normalize(tempMat);
	// 	params_["mean"] = tempParams["mean"];
	// 	params_["variance"] = tempParams["variance"];
	// 	params_["sd"] = tempParams["sd"];
	// 	params_["normalizedMat"] = tempParams["normalizedMat"];
	// 	params_["n"] = tempParams["normalizedMat"];
	// 	params_["unnormalizedMat"] = tempMat;
	// 	params_["un"] = tempMat;
	// }

	// // 正規化されている行列を正規化前に戻す
	// // かなり実装適当なので、超遅いはず
	// // さらにunnormalizeする行列は予め必要なサイズを初期化時に確保しておく必要がある.
	// void EMat::toUnnormalizedMat(cv::Mat mean, cv::Mat sd){
	// 	cv::Mat tempMat;
	// 	tempMat = this->clone();
	// 	params_["normalizedMat"] = tempMat;
	// 	params_["n"] = tempMat;
	// 	map<string, cv::Mat> tempParams = mc::unnormalize(tempMat, mean, sd);
	// 	params_["mean"] = tempParams["mean"];
	// 	params_["variance"] = tempParams["variance"];
	// 	params_["sd"] = tempParams["sd"];
	// 	params_["unnormalizedMat"] = tempParams["unnormalizedMat"];
	// 	params_["un"] = tempParams["unnormalizedMat"];
	// }

	// double EMat::sum(){
	// 	double sum = 0;
	// 	for (int i = 0; i < this->rows; ++i){
	// 		sum += calcTotalSum(this->row(i));
	// 	}

	// 	return sum;
	// }

	EMat EMat::cast(const cv::Mat mat){
		EMat emat(mat);
		return emat;
	}

	vector<cv::Mat> EMat::cast(vector<EMat> emats){
		vector<cv::Mat> mats;
		for (int i = 0; i < emats.size(); ++i)	mats.push_back(emats[i]);
		return mats;
	}

}
