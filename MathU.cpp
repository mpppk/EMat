// ユーザー定義
#include <MathU.h>
#include <cmath>

using namespace std;

namespace mc{

	MathU::Gaussian::Gaussian(const double dim, const RVec& mean, const cv::Mat& s)
	: dim_(dim), mean_(mean), s_(s){
		if( !isValid() ){ invalid_argument("invalid argument in MathU::Gaussian::Gaussian"); }
	};

	MathU::Gaussian::Gaussian(const double mean, const double sigma)
	: dim_(1), mean_( cv::Mat::ones(1, 1, CV_64F) * mean ), s_( cv::Mat::ones(1, 1, CV_64F) * sigma ){
		if( !isValid() ){ invalid_argument("invalid argument in MathU::Gaussian::Gaussian"); }
	}
	
	double MathU::Gaussian::calc(const RVec& x) const{
		if ( !isValid(x) ){ throw invalid_argument("x is invalid in MathU::Gaussian::calc"); }
		double e = ( -0.5 * ( x.m() - mean_.m() ).t() * s_.inv() ).dot( x.m() - mean_.m() );
		double u = pow( sqrt(2*CV_PI), dim_ ) * sqrt(cv::norm(s_));
		return (1.0 / u) * exp(e);
	}

	double MathU::Gaussian::calc(const double x) const{
		if( dim_ != 1 ){ throw invalid_argument("dim is not 1. You must use RVec as Gaussian::calc() argument"); }
		return calc( cv::Mat::ones(1, 1, CV_64F) * x );
	}

	bool MathU::Gaussian::isValid() const{
		if( mean_.size() != dim_ ){ return false; }
		if( s_.rows != dim_ ){ return false; }
		if( s_.cols != dim_ ){ return false; }
		return true;
	}
	bool MathU::Gaussian::isValid(const RVec& x) const{
		if( x.size() != mean_.size() ){ return false; }
		return true;
	}

	RVec MathU::Gaussian::convolute(const RVec& x, const int s, const int w){
		int sw = s * w;
		if( x.size() < sw ){ throw invalid_argument("x length is too short. x must have more w * σ size. "); }
		RVec ret = x.m().clone();
		auto table = Gaussian::createTable(s);
		for(int i = sw; i < (x.size() - sw); i++){
			double value = 0;
			for(int j = -sw; j <= sw; j++){ value += x[i + j] * table[j + sw]; }// j + sw => index 0..sw
			ret[i] -= value/calcTotalSum(table);
		}
		return ret;
	}

	RVec MathU::Gaussian::createTable(const int sigma, const int range){
		Gaussian g(0, sigma);
		RVec ret(sigma * range * 2 + 1);
		for(int i = 0; i < ret.size(); i++){
			ret[sigma * range - i] = g.calc(i);
			ret[sigma * range + i] = ret[sigma * range - i];
		}
		return ret;
	}

	// L1normを計算する
	double MathU::calcL1norm(const cv::Mat &vec){
		return calcL1norm(RVec(vec));
	}

	double MathU::calcL1norm(const RVec &vec){
		double L1norm = 0;
		for(int i = 0; i < vec.size(); i++){ L1norm += fabs(vec[i]); }
		return L1norm;
	}

	// L1normは絶対値をとるが、この関数はとらない
	double MathU::calcTotalSum(const cv::Mat &vec){
		return calcTotalSum(RVec(vec));
	}

	double MathU::calcTotalSum(const RVec &vec){
		double sum = 0;
		for(int i = 0; i < vec.size(); i++){ sum += vec[i]; }
		return sum;
	}

	double MathU::calcL2norm(const cv::Mat &vec){
		return calcL2norm(RVec(vec));
	}

	double MathU::calcL2norm(const RVec &vec){
		double L2norm = 0;
		for(int i = 0; i < vec.size(); i++){ L2norm += pow(vec[i], 2); }
		return sqrt(L2norm);
	}

	// 重み付の共分散を計算する
	double MathU::calcWCov(const cv::Mat &vec1, const cv::Mat &vec2,
		double mean1, double mean2, const cv::Mat &vec_weight){
		return calcWCov(RVec(vec1), RVec(vec2), mean1, mean2, RVec(vec_weight));
	}

	double MathU::calcWCov(const RVec &vec1, const RVec &vec2,
		double mean1, double mean2, const RVec &vec_weight){
		if(vec1.size() != vec2.size()){
			ostringstream os;
			os << "[in calcWCov] vec1's col num is " << vec1.size() << "," << endl
			<< "but vec2's col num is " << vec2.size() << "." << endl;
			throw runtime_error(os.str());
		}

		if(vec1.size() != vec_weight.size()){
			ostringstream os;
			os << "[in calcWCov] vec1's col num is " << vec1.size() << "," << endl
			<< "but weight's col num is " << vec2.size() << "." << endl;
			throw runtime_error(os.str());
		}

		RVec vec1SubedMean = vec1.m().clone();
		RVec vec2SubedMean = vec2.m().clone();

		// データから平均を引いた行列を計算
		vec1SubedMean -= mean1;
		vec2SubedMean -= mean2;

		return vec1SubedMean.m().dot( vec2SubedMean.m().mul(vec_weight.m()) );
	}

	// 重み付の共分散行列を計算する
	// dataMat 1行に一つの事例を格納
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_mean, const cv::Mat &vec_weight){
		return calcWCovMat(dataMat, RVec(vec_mean), RVec(vec_weight));
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const RVec &vec_mean, const RVec &vec_weight){
		if(dataMat.cols != vec_mean.size()){
			ostringstream os;
			os << "[in calcWCovMat] dataMat's col num is " << dataMat.cols << "," << endl
			<< "but mean's col num is " << vec_mean.size() << "." << endl;
			throw runtime_error(os.str());
		}

		if(dataMat.rows != vec_weight.size()){
			ostringstream os;
			os << "[in calcWCovMat] dataMat's row num is " << dataMat.rows << "," << endl
			<< "but weight's col num is " << vec_weight.size() << "." << endl;
			throw runtime_error(os.str());
		}
		// ----------エラーチェックここまで----------

		cv::Mat covMat(dataMat.cols, dataMat.cols, CV_64F);
		cv::Mat dataMatSubMean = dataMat.clone();

		// 重みなしの共分散行列が正しく推定できているかの確認用
		RVec vec_noWeight = cv::Mat::ones(1, dataMat.rows, CV_64F);

		//共分散行列の各要素を計算
		for(int covMatRow_i = 0; covMatRow_i < covMat.rows; covMatRow_i++){
			for(int covMatCol_i = 0; covMatCol_i < covMat.cols; covMatCol_i++){
				covMat.at<double>(covMatRow_i, covMatCol_i) = 
				calcWCov(RVec( dataMat.col(covMatRow_i).t() ) ,
					RVec( dataMat.col(covMatCol_i).t() ),
					vec_mean[covMatRow_i],
					vec_mean[covMatCol_i],
					vec_weight);
			}
		}
		return covMat;
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight){
		return MathU::calcWCovMat(dataMat, RVec(vec_weight));
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const RVec &vec_weight){
		RVec vec_dataMean = cv::Mat::Mat::zeros(1, dataMat.cols, CV_64F);
		// 重みを考慮した平均を計算する
		for (int col_i = 0; col_i < dataMat.cols; ++col_i){
			for (int row_i = 0; row_i < dataMat.rows; ++row_i){
				vec_dataMean[col_i] += dataMat.at<double>(row_i, col_i) * vec_weight[row_i];
			}
		}
		vec_dataMean /= calcTotalSum(vec_weight);
		// cout << "dbg in calcWCovMat mean(正しいことを確認済み): " << vec_dataMean << endl;
		return calcWCovMat(dataMat, vec_dataMean, vec_weight);
	}

	// オプションを指定できるcalcWCovMat
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight, const bitset<CovMatOptionsNum> flags){
		return calcWCovMat(dataMat, vec_weight, flags);
	}

	// オプションを指定できるcalcWCovMat
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const RVec &vec_weight, const bitset<CovMatOptionsNum> flags){
		cv::Mat result = calcWCovMat(dataMat, vec_weight);
		if(flags[SCALE])	result /= calcTotalSum(vec_weight);
		return result;
	}

	// opencv2.0用
	double MathU::rbfKernel(const cv::Mat &vec1, const cv::Mat &vec2, const double rbfSigma){
		return rbfKernel(RVec(vec1), RVec(vec2), rbfSigma);
	}
	
	double MathU::rbfKernel(const RVec &vec1, const RVec &vec2, const double rbfSigma){
		RVec tempMat = vec1.m() - vec2.m();
		return exp(-1 * tempMat.m().dot(tempMat.m()) / rbfSigma);
	}

	// 座標(x,y), 長辺と短辺が(a,b),rot分回転した楕円を描画のための1度ごとの座標を返す
	cv::Mat MathU::plotEllipse(double x,double y,double a,double b,int rot){
		FILE *fp;
		double theta;
		unsigned int rotNum = 360;
		cv::Mat retMat = cv::Mat::ones(rotNum, 2, CV_64F) * -999;
		// cv::Mat mat1 = cv::Mat::ones(5, 5, CV_8U)*3;
		cv::Mat tempMat(1, 2, CV_64F);
		for(int i = 0; i < 360; i++){
			theta = 2*M_PI/360;

			retMat.at<double>(i,0) = a * cos((double)theta*i) * cos((double)theta*rot) - b * sin((double)theta*i) * sin((double)theta*rot) + x;
			retMat.at<double>(i,1)  = a * cos((double)theta*i) * sin((double)theta*rot) + b * sin((double)theta*i) * cos((double)theta*rot) + y;
			// retMat.at<double>(i,0) = a * cos((double)theta*i) - b * sin((double)theta*i) + x;
			// retMat.at<double>(i,1)  = a * cos((double)theta*i) + b * sin((double)theta*i) + y;
			// fprintf(fp,"%.3lf",a * cos((double)theta*i) * cos((double)theta*rot) - b * sin((double)theta*i) * sin((double)theta*rot) + x);
			// fprintf(fp,",%.3lf\n",a * cos((double)theta*i) * sin((double)theta*rot) + b * sin((double)theta*i) * cos((double)theta*rot) + y);
		}
		return retMat;
	}

	// 共分散行列から、確立楕円描画のための1度ごとの座標を返す
	// meanは2値でなければいけない
	cv::Mat MathU::plotConfidenceEllipse(const cv::Mat &mean, const cv::Mat &covMat, const int sigma){
		return plotConfidenceEllipse(RVec(mean), covMat, sigma);
	}

	cv::Mat MathU::plotConfidenceEllipse(const RVec &mean, const cv::Mat &covMat, const int sigma){
		cv::Mat eigenValues, eigenVectors;
		cv::eigen(covMat, eigenValues, eigenVectors);
		double degree = atan2(eigenVectors.at<double>(1,0), eigenVectors.at<double>(0,0)) * 180.0 / M_PI;
		// int degree = 45;
		return plotEllipse(
			mean[0],
			mean[1],
			sigma * eigenValues.at<double>(0,0),
			sigma * eigenValues.at<double>(1,0),
			(int)fabs(degree));
		// TODO: degreeがこれでいいのか考える
	}

	RVec MathU::toEachColsMean(const cv::Mat &data){
		RVec sum = cv::Mat::zeros(1, data.cols, CV_64F);
		// 各要素の合計とその二乗を計算
		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				sum[col_i] += data.at<double>(row_i, col_i);
			}
		}
		return sum / data.rows;
	}

	RVec MathU::toEachColsSquareMean(const cv::Mat &data){
		RVec squareSum = cv::Mat::zeros(1, data.cols, CV_64F);
		// 各要素の合計とその二乗を計算
		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				squareSum[col_i] += pow(data.at<double>(row_i, col_i), 2);
			}
		}
		return squareSum / data.rows;
	}

	RVec MathU::toEachColsSD(const cv::Mat &data){
		RVec mean = toEachColsMean(data);
		RVec squareMean = toEachColsSquareMean(data);
		return toEachColsSD(data, mean, squareMean);
	}

	RVec MathU::toEachColsSD(const cv::Mat &data, const RVec &mean){
		RVec squareMean = toEachColsSquareMean(data);
		return toEachColsSD(data, mean, squareMean);
	}

	RVec MathU::toEachColsSD(const cv::Mat &data, const RVec &mean, const RVec &squareMean){
		RVec sd(data.cols);
		for(int i = 0; i < data.cols; i++)	sd[i] = sqrt( squareMean[i] - pow(mean[i], 2) );
		return sd;
	}

	RVec MathU::toEachColsVariance(const RVec sd){
		RVec variance(sd.size());
		for(int i = 0; i < sd.size(); i++){
			variance[i] = sqrt(sd[i]);
		}
		return variance;
	}

	// 移動平均を計算する。着目要素からwidth-1個を計算に含める。最初の0〜(width-1)個の要素に対しては何もしない。
	cv::Mat MathU::movingAverage(const cv::Mat &mat, const int arg_width){
		int width = arg_width - 1;
		cv::Mat retMat(mat.rows, mat.cols, CV_64F);
		auto retIt = retMat.begin<double>();
		for(auto it = mat.begin<double>(); it != (mat.begin<double>()+width); it++){
			*retIt = *it;
			retIt++;
		}
		for(auto it = ( mat.begin<double>()+width ); it != mat.end<double>(); it++){
			double sum = 0;
			int cnt = 0;
			for(auto widthIt = (it-width); widthIt != it+1; widthIt++){
				sum += *widthIt;
				cnt++;
			}
			*retIt = sum/(width+1);
			retIt++;
		}
		return retMat;
	}

	cv::Mat MathU::movingAverageToEachCol(const cv::Mat &mat, const int arg_width){
		cv::Mat retMat = cv::Mat::zeros(mat.rows, mat.cols, CV_64F);
		for(int col_i = 0; col_i < mat.cols; col_i++){
			movingAverage(mat.col(col_i), arg_width).copyTo(retMat.col(col_i));
		}
		return retMat;
	}

	map<string, cv::Mat> MathU::temporalResolution(const RVec &vec, const int width, const int xdim, const int ydim){
		if( xdim < ydim ){ throw invalid_argument("ydim is big than xdim.(in MathU::temporalResolution)"); }

		map<string, cv::Mat> results;
		vector<RVec> xvecs, yvecs;
		cv::Mat smoothMat = movingAverage(vec.m(), width);
		// 平滑化されていない部分は飛ばす
		auto it = smoothMat.begin<double>();
		for(int i = 0; i < width-1; i++){ it++; }

		for(; it != smoothMat.end<double>(); it++){
			RVec xvec(xdim), yvec(ydim);
			if( ( it + (xdim - 1) * width ) == smoothMat.end<double>() ){ break; }
			for(int i = 0; i < xdim; i++){ xvec[i] = *(it + i * width); }
			for(int i = 0; i < ydim; i++){ yvec[i] = xvec[i + xdim - ydim]; }// ydimに応じてxvecをコピー

			xvecs.push_back(xvec);
			yvecs.push_back(yvec);
		}
		results["x"] = MatU::toMat(xvecs);
		results["y"] = MatU::toMat(yvecs);
		return results;
	}

	// １行が(index-dim-1)〜(index)までの値を持つ行列に変換する
	cv::Mat MathU::toMultiDim(const RVec &vec, const int dim){
		cv::Mat mat(vec.size(), dim, CV_64F);
		for(int vec_i = 0; vec_i < vec.size(); vec_i++){
			for(int dim_i = 0; dim_i < dim; dim_i++){
				int matColIndex = (dim - 1) - dim_i;
				int vecIndex = ( (vec_i - dim_i) >= 0 )? (vec_i - dim_i) : 0;
				mat.at<double>(vec_i, matColIndex) = vec[vecIndex];
			}
		}
		return mat;
	}

	// 各列について、toMultiDimを行う
	cv::Mat MathU::toMultiDim(const cv::Mat &mat, const int dim){
		cv::Mat retMat = toMultiDim((RVec)(mat.col(0).t()), dim);// 最初の一回はここで
		for(int col_i = 1; col_i < mat.cols; col_i++){
			RVec vec = mat.col(col_i).t();
			retMat = MatU::mergeMatToSide( retMat, toMultiDim(mat.col(col_i), dim) );
		}
		return retMat;
	}
	
	cv::Mat MathU::normalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd){
		return normalize(data, RVec(mean), RVec(sd));
	}

	cv::Mat MathU::normalize(const cv::Mat &data, const RVec &mean, const RVec &sd){
		// 行列を正規化
		cv::Mat normalizedMat = data.clone();

		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				normalizedMat.at<double>(row_i, col_i) -= mean[col_i];
				normalizedMat.at<double>(row_i, col_i) /= sd[col_i];
			}
		}
		return normalizedMat;
	}

	map<string, cv::Mat> MathU::normalize(const cv::Mat &data){
		RVec mean = toEachColsMean(data);
		RVec sd = toEachColsSD(data, mean);
		RVec variance = toEachColsVariance(sd);

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "variance", variance.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd.m() ) );
		retMats.insert( map<string, cv::Mat>::value_type( "normalizedMat", normalize(data, mean, sd) ) );
		return retMats;
	}

	cv::Mat MathU::unnormalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd){
		return unnormalize(data, RVec(mean), RVec(sd));
	}

	cv::Mat MathU::unnormalize(const cv::Mat &data, const RVec &mean, const RVec &sd){
		cv::Mat unnormalizedMat = data.clone();

		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				unnormalizedMat.at<double>(row_i, col_i) *= sd[col_i];
				unnormalizedMat.at<double>(row_i, col_i) += mean[col_i];				
			}
		}
		return unnormalizedMat;
	}

	cv::Mat MathU::normalizeHistogram(const cv::Mat &src){
		cv::Mat dst;
		// minとmaxを決定する
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

		cv::convertScaleAbs(src, dst, 255.0/(maxVal-minVal), (255.0/(maxVal-minVal))*(-minVal));
		return dst;
	}

	double MathU::calcEuclideanDist(const cv::Mat &point1, const cv::Mat &point2){
		return calcEuclideanDist(RVec(point1), RVec(point2));
	}

	double MathU::calcEuclideanDist(const RVec &point1, const RVec &point2){
		// ベクトルの行数が違う場合は例外
		if(point1.size() != point2.size()){
			throw invalid_argument("in calcEuclideanDist invalid value is passed. vector nums are different.");
		}

		double dist = 0;
		for (int i = 0; i < point1.size(); ++i){
			dist += pow( point1[i] - point2[i], 2 );
		}

		return sqrt(dist);
	}

	double MathU::calcMahalanobisDist(const RVec &point1, const RVec &point2, const cv::Mat &icover){
		RVec diff = point1.m() - point2.m();
		return sqrt( ( diff.m() * icover ).dot( diff.m() ) );
	}

	RVec MathU::correctBaseLine(const RVec &data){
		RVec newData = data.m().clone();
		constexpr int WIDTH = 7;
		constexpr array<int, WIDTH> RATES = {2, 12, 30, 40, 30, 12, 2};
		const int RATES_SUM = accumulate(RATES.begin(), RATES.end(), 0);
		constexpr int MID = WIDTH/2;
		
		for(int i = MID; i < data.size() - MID; i++){
			newData[i] = 0;
			for(int j = -MID; j < MID; j++){ newData[i] += data[i + j] * RATES[j + MID]; }
			newData[i] = data[i] - (newData[i] / RATES_SUM);
		}
		return newData;
	}
}


