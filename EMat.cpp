#include <EMat.h>

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
	EMat::EMat(const vector< vector<string> >& contents) : mat_(MatU::toMat(contents)){
		// TODO contentsが行列形式であることが保証されていない
		if(!isValid())	throw invalid_argument("argument can not pass to EMat");
	}

	cv::Mat& EMat::m(){return mat_;}
	const cv::Mat& EMat::m() const{
		const cv::Mat& retMat = mat_;
		return retMat;
	}

	EMat EMat::toNormalizedMat() const{
		return MathU::normalize(mat_)["normalizedMat"];
	}

	EMat EMat::toNormalizedMat(const RVec &mean, const RVec &sd) const{
		return MathU::normalize(mat_, mean, sd);
	}

	EMat EMat::toUnnormalizedMat(const RVec &mean, const RVec &sd) const{
		return MathU::unnormalize(mat_, mean, sd);
	}

	RVec EMat::toEachColsMean() const{
		return MathU::toEachColsMean(mat_);
	}

	RVec EMat::toEachColsSD(const RVec &mean) const{
		return MathU::toEachColsSD(mat_, mean);
	}

	RVec EMat::toEachColsSD() const{
		return MathU::toEachColsSD(mat_);
	}


	EMat EMat::toCovMat() const{
		cv::Mat covMat, mean;
		calcCovarMatrix(mat_, covMat, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
		return covMat;
	}

	EMat EMat::toWCovMat(const RVec &weight, const bitset<MathU::CovMatOptionsNum> flags) const{
		return MathU::calcWCovMat(mat_, weight, flags);
	}

	// 要素がstringのvectorを返す
	vector< vector<string> > EMat::toVec() const{
		return MatU::toVec(mat_);
	}

	vector<cv::Mat> EMat::cast(const vector<EMat>& emats){// static
		vector<cv::Mat> mats;
		for (int i = 0; i < emats.size(); ++i)	mats.push_back(emats[i].m());
		return mats;
	}
}
