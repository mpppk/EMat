// ユーザー定義
#include <MathU.h>

using namespace std;

namespace mc{

	// L1normを計算する
	double MathU::calcL1norm(const cv::Mat &vec){
		// e_isRVec(vec);

		double L1norm = 0;
		for(int i = 0; i < vec.cols; i++){
			L1norm += fabs(vec.at<double>(0, i));
		}
		return L1norm;
	}

	// L1normは絶対値をとるが、この関数はとらない
	double MathU::calcTotalSum(const cv::Mat &vec){
		// e_isRVec(vec);

		double sum = 0;
		for(int i = 0; i < vec.cols; i++){
			sum += vec.at<double>(0, i);
		}
		return sum;
	}

	double MathU::calcL2norm(const cv::Mat &vec){
		// e_isRVec(vec);
		double L2norm = 0;
		for(int i = 0; i < vec.cols; i++){
			L2norm += pow(vec.at<double>(0, i), 2);
		}
		return L2norm;

	}

	// 重み付の共分散を計算する
	double MathU::calcWCov(const cv::Mat &vec1, const cv::Mat &vec2,
		double mean1, double mean2, const cv::Mat &vec_weight){
		
		// ----------エラーチェック----------
		// e_isRVec(vec1, "vec1 [in calcWCov]");
		// e_isRVec(vec2, "vec2 [in calcWCov]");

		if(vec1.cols != vec2.cols){
			ostringstream os;
			os << "[in calcWCov] vec1's col num is " << vec1.cols << "," << endl
			<< "but vec2's col num is " << vec2.cols << "." << endl;
			throw runtime_error(os.str());
		}

		if(vec1.cols != vec_weight.cols){
			ostringstream os;
			os << "[in calcWCov] vec1's col num is " << vec1.cols << "," << endl
			<< "but weight's col num is " << vec2.cols << "." << endl;
			throw runtime_error(os.str());
		}
		// ----------エラーチェックここまで----------

		cv::Mat vec1SubedMean = vec1.clone();
		cv::Mat vec2SubedMean = vec2.clone();

		// データから平均を引いた行列を計算
		vec1SubedMean -= mean1;
		vec2SubedMean -= mean2;

		cv::Mat result = vec1SubedMean * vec2SubedMean.mul(vec_weight).t();
		return result.at<double>(0, 0);
	}

	// 重み付の共分散行列を計算する
	// dataMat 1行に一つの事例を格納
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_mean, const cv::Mat &vec_weight){
		
		// ----------エラーチェック----------
		// e_isRVec(vec_mean, "vec_mean [in calcWCovMat]");
		// e_isRVec(vec_weight, "vec_weight [in calcWCovMat]");		

		if(dataMat.cols != vec_mean.cols){
			ostringstream os;
			os << "[in calcWCovMat] dataMat's col num is " << dataMat.cols << "," << endl
			<< "but mean's col num is " << vec_mean.cols << "." << endl;
			throw runtime_error(os.str());
		}

		if(dataMat.rows != vec_weight.cols){
			ostringstream os;
			os << "[in calcWCovMat] dataMat's row num is " << dataMat.rows << "," << endl
			<< "but weight's col num is " << vec_weight.cols << "." << endl;
			throw runtime_error(os.str());
		}
		// ----------エラーチェックここまで----------


		cv::Mat covMat(dataMat.cols, dataMat.cols, CV_64F);
		cv::Mat dataMatSubMean = dataMat.clone();

		// 重みなしの共分散行列が正しく推定できているかの確認用
		cv::Mat vec_noWeight = cv::Mat::ones(1, dataMat.rows, CV_64F);

		//共分散行列の各要素を計算
		for(int covMatRow_i = 0; covMatRow_i < covMat.rows; covMatRow_i++){
			for(int covMatCol_i = 0; covMatCol_i < covMat.cols; covMatCol_i++){
				covMat.at<double>(covMatRow_i, covMatCol_i) = 
				calcWCov(dataMat.col(covMatRow_i).t(),
					dataMat.col(covMatCol_i).t(),
					vec_mean.at<double>(0, covMatRow_i),
					vec_mean.at<double>(0, covMatCol_i),
					vec_weight);
			}
		}
		return covMat;
	}

	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight){
		// cout << "dbg in calcWCovMat dataMat: " << dataMat << endl;
		// cout << "dbg in calcWCovMat vec_weight: " << vec_weight << endl;
		cv::Mat vec_dataMean = cv::Mat::Mat::zeros(1, dataMat.cols, CV_64F);
		// 重みを考慮した平均を計算する
		for (int col_i = 0; col_i < dataMat.cols; ++col_i){
			for (int row_i = 0; row_i < dataMat.rows; ++row_i){
				vec_dataMean.at<double>(0, col_i) += dataMat.at<double>(row_i, col_i) * vec_weight.at<double>(0, row_i);
			}
		}
		vec_dataMean /= calcTotalSum(vec_weight);
		// cout << "dbg in calcWCovMat mean(正しいことを確認済み): " << vec_dataMean << endl;
		return calcWCovMat(dataMat, vec_dataMean, vec_weight);
	}

	// オプションを指定できるcalcWCovMat
	cv::Mat MathU::calcWCovMat(const cv::Mat &dataMat, const cv::Mat &vec_weight, const bitset<CovMatOptionsNum> flags){
		cv::Mat result = calcWCovMat(dataMat, vec_weight);
		if(flags[SCALE])	result /= calcTotalSum(vec_weight);
		return result;
	}

	// opencv2.0用
	double MathU::rbfKernel(const cv::Mat &vec1, const cv::Mat &vec2, const double rbfSigma){
		// if(!isRVec(vec1) || !isRVec(vec2)){
		// 	throw runtime_error("argument is not vector!! (in rbfKernel)");			
		// }
		cv::Mat vec2_t;
		cv::transpose(vec2, vec2_t);
		cv::Mat tempMat = vec1 - vec2;
		return exp(-1 * tempMat.dot(tempMat) / rbfSigma);
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
		cout << "in plotConfidenceEllipse" << endl;
		// e_isRVec(mean, "in plotConfidenceEllipse. mean is not RVec.");
		cv::Mat eigenValues, eigenVectors;
		// cout << "dbg:plotConfidenceEllipse input mean " << mean << endl;
		// cout << "dbg:plotConfidenceEllipse input covMat" << covMat << endl;
		// cout << "covMat rows: " << covMat.rows << " cols: " << covMat.cols << endl;
		cv::eigen(covMat, eigenValues, eigenVectors);
		// cout << "eigenValues" << eigenValues << endl;
		// cout << "eigenVectors" << eigenVectors << endl;
		double degree = atan2(eigenVectors.at<double>(1,0), eigenVectors.at<double>(0,0)) * 180.0 / M_PI;
		// int degree = 45;
		// cout << "degree: " << degree << endl;
		return plotEllipse(
			mean.at<double>(0,0),
			mean.at<double>(0,1),
			sigma * eigenValues.at<double>(0,0),
			sigma * eigenValues.at<double>(1,0),
			(int)fabs(degree));
		// TODO: degreeがこれでいいのか考える
	}

	map<string, cv::Mat> MathU::normalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd){
		// 行列を正規化
		cv::Mat normalizedMat = data.clone();

		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				normalizedMat.at<double>(row_i, col_i) -= mean.at<double>(0, col_i);
				normalizedMat.at<double>(row_i, col_i) /= sd.at<double>(0, col_i);				
			}
		}

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd ) );
		retMats.insert( map<string, cv::Mat>::value_type( "normalizedMat", normalizedMat ) );
		return retMats;
	}

	map<string, cv::Mat> MathU::normalize(const cv::Mat &data){
		cv::Mat mean(1, data.cols, CV_64F);
		cv::Mat squareMean(1, data.cols, CV_64F);
		cv::Mat sum = cv::Mat::zeros(1, data.cols, CV_64F);
		cv::Mat squareSum = cv::Mat::zeros(1, data.cols, CV_64F);
		cv::Mat variance(1, data.cols, CV_64F);
		cv::Mat sd(1, data.cols, CV_64F);
		cv::Mat normalizedMat = data.clone();

		// 各要素の合計とその二乗を計算
		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				sum.at<double>(0, col_i) += data.at<double>(row_i, col_i);
				squareSum.at<double>(0, col_i) += pow(data.at<double>(row_i, col_i), 2);
			}
		}

		// 平均と標準偏差の計算
		// 分散 = ２乗の平均　－　平均の２乗
		for (int col_i = 0; col_i < data.cols; ++col_i){
			mean.at<double>(0, col_i) = sum.at<double>(0, col_i) / data.rows;
			squareMean.at<double>(0, col_i) = squareSum.at<double>(0, col_i) / data.rows;
			variance.at<double>(0, col_i) = squareMean.at<double>(0, col_i) - pow(mean.at<double>(0, col_i), 2);
			sd.at<double>(0, col_i) = sqrt(variance.at<double>(0, col_i));
		}

		// 行列を正規化
		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				normalizedMat.at<double>(row_i, col_i) -= mean.at<double>(0, col_i);
				normalizedMat.at<double>(row_i, col_i) /= sd.at<double>(0, col_i);				
			}
		}

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean ) );
		retMats.insert( map<string, cv::Mat>::value_type( "variance", variance ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd ) );
		retMats.insert( map<string, cv::Mat>::value_type( "normalizedMat", normalizedMat ) );
		return retMats;
	}

	map<string, cv::Mat> MathU::unnormalize(const cv::Mat &data, const cv::Mat &mean, const cv::Mat &sd){
		cv::Mat unnormalizedMat = data.clone();

		for (int row_i = 0; row_i < data.rows; ++row_i){
			for (int col_i = 0; col_i < data.cols; ++col_i){
				unnormalizedMat.at<double>(row_i, col_i) *= sd.at<double>(0, col_i);
				unnormalizedMat.at<double>(row_i, col_i) += mean.at<double>(0, col_i);				
			}
		}

		// 返り値
		map<string, cv::Mat> retMats;
		retMats.insert( map<string, cv::Mat>::value_type( "mean", mean ) );
		retMats.insert( map<string, cv::Mat>::value_type( "sd", sd ) );
		retMats.insert( map<string, cv::Mat>::value_type( "unnormalizedMat", unnormalizedMat ) );
		// cout << "dbg:Math unnormalize  unnormalizedMat: " << retMats["unnormalizedMat"] << endl; 
		return retMats;
	}

	// // 新たな重みを再急降下法法で計算する関数(GP用に特化しているので、他には使えない。バグが残っている)
	// cv::Mat MathU::calcSteepestDescent(const cv::Mat &vec_initialValue, const cv::Mat &vec_objectiveValiable, const cv::Mat &ASY, const double initStepSize, const double tolerance){
	// 	const int ISNT_MOVE = -1;// どの軸も動かしていないことを示す
	// 	const int PLUS = 0;// 計算結果を保存する行番号
	// 	const int MINUS = 1;// 計算結果を保存する行番号
	// 	const int SIGN_MAX = 2;// 符号は2種類(プラスとマイナス)

	// 	cv::Mat NNInitialValue = vec_initialValue.clone();
	// 	double stepSize = initStepSize;

	// 	// 負の値の重みをゼロにする
	// 	for (int initialValueCol_i = 0; initialValueCol_i < vec_initialValue.cols; ++initialValueCol_i){
	// 		if (vec_initialValue.at<double>(0, initialValueCol_i) < 0 ){
	// 			NNInitialValue.at<double>(0, initialValueCol_i) = 0;
	// 		}
	// 	}

	// 	cv::Mat vec_slack = cv::Mat::zeros(1, ASY.rows, CV_64F);
	// 	cv::Mat currentObjectiveValiable = cv::Mat::zeros(1, ASY.cols, CV_64F);
	// 	// 初期値で計算した時の値を求める
	// 	for (int ASYRow_i = 0; ASYRow_i < ASY.rows; ++ASYRow_i){
	// 		for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 			// cout << "nniv: " << NNInitialValue.at<double>(0, ASYRow_i) << endl;
	// 			// cout << "asy: " << ASY.at<double>(ASYRow_i, ASYCol_i) << endl;
	// 			currentObjectiveValiable.at<double>(0, ASYCol_i) += NNInitialValue.at<double>(0, ASYRow_i) * ASY.at<double>(ASYRow_i, ASYCol_i);
	// 		}
	// 	}
	// 	// cout << "currentObjectiveValiable: " << currentObjectiveValiable << endl;
	// 	// 目的変数との差を計算
	// 	double currentDist = 0;
	// 	for (int cOV_i = 0; cOV_i < currentObjectiveValiable.cols; ++cOV_i){
	// 		currentDist += fabs(currentObjectiveValiable.at<double>(0, cOV_i) - vec_objectiveValiable.at<double>(0, cOV_i));
	// 	}
	// 	// cout << "currentDist" << currentDist << endl;
	// 	// 各軸について動かしてみる
	// 	const int MAX_IT = 1000;// 最大反復回数
	// 	unsigned int it = 0;// 反復回数

	// 	// ステップサイズが許容誤差以下になるまでループ
	// 	while(stepSize < tolerance || it < MAX_IT){
	// 		int moveSign = -999;// 軸をどちら側に動かすか
	// 		int candidateIndex = ISNT_MOVE;

	// 		// 動かすべき軸を決定して動かす処理
	// 		for (int moveRow_i = 0; moveRow_i < ASY.rows; ++moveRow_i){
	// 			cv::Mat nextObjectiveValiable = cv::Mat::zeros(SIGN_MAX, ASY.cols, CV_64F);

	// 			// スラーク変数と元の重みの合計がゼロ以下かどうか
	// 			vector<bool> isLessThanZero;
	// 			isLessThanZero.push_back(false);
	// 			isLessThanZero.push_back(false);
	// 			if ( 0 <= (NNInitialValue.at<double>(0, moveRow_i) + vec_slack.at<double>(0, moveRow_i) + stepSize) ){
	// 				// cout << "is less than zero (PLUS)" << endl;
	// 				isLessThanZero[PLUS] = true;
	// 			}
	// 			if ( 0 <= (NNInitialValue.at<double>(0, moveRow_i) + vec_slack.at<double>(0, moveRow_i) - stepSize) ){
	// 				// cout << "is less than zero (MINUS)" << endl;
	// 				isLessThanZero[MINUS] = true;
	// 			}
				
	// 			// スラーク変数のL2ノルムを計算
	// 			double norm = 0;
	// 			for (int slack_i = 0; slack_i < vec_slack.cols; ++slack_i){
	// 				norm += vec_slack.at<double>(0, slack_i) * vec_slack.at<double>(0, slack_i);
	// 			}
	// 			// ペナルティを計算
	// 			double penalty = norm + stepSize;

	// 			// 軸を動かしたときの結果を計算
	// 			for (int ASYRow_i = 0; ASYRow_i < ASY.rows; ++ASYRow_i){
	// 				for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 					double moveValue = (ASYRow_i == moveRow_i)? stepSize : 0;
	// 					// 各軸に対して、プラス方向とマイナス方向にそれぞれ動いてみる
	// 					nextObjectiveValiable.at<double>(PLUS, ASYCol_i) += (NNInitialValue.at<double>(0, ASYRow_i) + vec_slack.at<double>(0, ASYRow_i) + moveValue) * ASY.at<double>(ASYRow_i, ASYCol_i) + penalty;
	// 					nextObjectiveValiable.at<double>(MINUS, ASYCol_i) += (NNInitialValue.at<double>(0, ASYRow_i) + vec_slack.at<double>(0, ASYRow_i) - moveValue) * ASY.at<double>(ASYRow_i, ASYCol_i) + penalty;
	// 				}
	// 			}

	// 			// cout << "nextObjectiveValiable: " << nextObjectiveValiable << endl;
	// 			// プラス方向とマイナス方向について
	// 			for (int sign_i = 0; sign_i < SIGN_MAX; ++sign_i){
	// 				double nextDist = 0;

	// 				// スラーク変数と元の重みの合計がゼロ以下になる場合は距離を計算しない
	// 				if (isLessThanZero[sign_i] == true)	continue; 

	// 				// 目的変数との距離を計算
	// 				for (int nOVCol_i = 0; nOVCol_i < nextObjectiveValiable.cols; ++nOVCol_i){
	// 					nextDist += fabs(nextObjectiveValiable.at<double>(sign_i, nOVCol_i) - vec_objectiveValiable.at<double>(0, nOVCol_i));
	// 				}
	// 				// 今までの計算結果と比べて、目的変数により近い結果が得られていればそちらを残す
	// 				if(currentDist > nextDist){
	// 					cout << "plus: " << isLessThanZero[PLUS] << endl;
	// 					cout << "minus: " << isLessThanZero[MINUS] << endl;
	// 					cout <<"update: currentDist-> " << currentDist << " newDist[" << moveRow_i << "]-> " << nextDist << endl;
	// 					candidateIndex = moveRow_i;// 動く軸
	// 					moveSign = sign_i;// 動く方向
	// 					currentDist = nextDist;
	// 				}
	// 			}
	// 		}
	// 		// もしどこにも移動しなかったらstepSizeを小さくする
	// 		if (candidateIndex == ISNT_MOVE){
	// 			if(tolerance > stepSize)	break;// ステップサイズが許容誤差より小さくなったら終了
	// 			else stepSize /= 2;
	// 		}else{
	// 			// 一番 目的変数に近くなる方向へ移動
	// 			vec_slack.at<double>(0, candidateIndex) += (moveSign == PLUS)?	stepSize : -stepSize;
	// 		}

	// 		cout << "it: " << it << " stepSize: " << stepSize << "	tolerance: " << tolerance << endl;
	// 		it++;
	// 	}
	// 	cout << "slack: " << vec_slack << endl;

	// 	// 新しい重みを計算
	// 	cv::Mat newWeight = NNInitialValue.clone();
	// 	for (int slack_i = 0; slack_i < vec_slack.cols; ++slack_i){
	// 		newWeight.at<double>(0, slack_i) += vec_slack.at<double>(0, slack_i);
	// 	}
	// 	cout << "new weight: " << newWeight << endl;

	// 	return vec_slack;
	// }

	// // 新たな重みを山登り法で計算する関数(GP用に特化しているので、他には使えない)
	// cv::Mat MathU::calcHillClimbing(const cv::Mat &arg_vec_weight, const cv::Mat &arg_vec_GPMean, const cv::Mat &arg_ASY, const double initStepSize, const double tolerance, const int maxIt, const double alpha){
	// 	// const unsigned int maxIt = 1000;
	// 	double stepSize = initStepSize;
	// 	// 引数のエラーチェック
	// 	// mc::e_isRVec(arg_vec_weight, "vec_weight is not vector!! (in calcHillClimbing)");
	// 	// mc::e_isRVec(arg_vec_GPMean, "vec_GPMean is not vector!! (in calcHillClimbing)");

	// 	// 小数点以下の有効桁数3桁
	// 	// cout << fixed << setprecision(3);

	// 	//RVecに代入していく
	// 	RVec vec_weight;	vec_weight = arg_vec_weight;
	// 	RVec vec_GPMean;	vec_GPMean = arg_vec_GPMean;
	// 	RVec ASY;			ASY = arg_ASY;

	// 	// 負の値の重みがゼロになるようなスラック変数を設定
	// 	RVec vec_slack(1, vec_weight.size(), CV_64F);
	// 	for (int weight_i = 0; weight_i < vec_weight.size(); ++weight_i){
	// 		vec_slack[weight_i] = (vec_weight[weight_i] < 0)?	fabs(vec_weight[weight_i]): 0;
	// 	}

	// 	cout << endl << "---------- HillClimbing start ----------" << endl;
	// 	// cout << "weight: " << vec_weight << endl;
	// 	// cout << "slack: " << vec_slack << endl;
	// 	// cout << "asy: " << ASY << endl;

	// 	// 初期値で計算した時の真値との距離を求める
	// 	double currentDist = 0;
	// 	RVec vec_tempOV(1, ASY.cols, CV_64F);
	// 	cv::Mat vec_initMat = cv::Mat::zeros(1, ASY.cols, CV_64F);
	// 	vec_tempOV = vec_initMat;

	// 	// tempOVを計算
	// 	for (int weight_i = 0; weight_i < vec_weight.size(); ++weight_i){
	// 		for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 			// cout << "weight/ " << vec_weight[weight_i] << "  slack/ " << vec_slack[weight_i] << "  asy/ " << ASY(weight_i, ASYCol_i) << endl;
	// 			vec_tempOV[ASYCol_i] += (vec_weight[weight_i] + vec_slack[weight_i]) * ASY(weight_i, ASYCol_i);
	// 		}
	// 	}
	// 	// cout << "vec_tempOV: " << vec_tempOV << endl;
	// 	// slack変数のL2ノルムを計算
	// 	double slackNorm = calcL2norm(vec_slack);
	// 	// currentDistを計算
	// 	for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 		currentDist += pow(vec_tempOV[ASYCol_i] - vec_GPMean[ASYCol_i], 2) + slackNorm;
	// 	}
	// 	cout << "init dist: " << currentDist << endl;
	// 	// 誤差が閾値以下になるまでループ
	// 	unsigned int it = 0;
	// 	double shiftValue = 0;
	// 	while(tolerance <= stepSize && it < maxIt){
	// 		bool nFlag = false;// 移動した結果が負の値になるかどうか
	// 		// プラスとマイナス方向に移動した時の真値との距離を求める

	// 		RVec vec_nextPlusWeight(vec_weight.size());
	// 		RVec vec_nextMinusWeight(vec_weight.size());

	// 		// slack変数に足しこむ値を保持する
	// 		RVec vec_addToSlack(1, vec_weight.size(), CV_64F);
			
	// 		bool moveFlagOfAll = false;

	// 		// 現在の重み
	// 		RVec vec_currentWeight;
	// 		vec_currentWeight = vec_weight + vec_slack;

	// 		// 移動量を決める			
	// 		double shiftValue = stepSize;
	// 		for (int weight_i = 0; weight_i < vec_currentWeight.size(); ++weight_i){
	// 			if (vec_currentWeight[weight_i] != 0 && vec_currentWeight[weight_i] <= shiftValue){
	// 				shiftValue = vec_currentWeight[weight_i];
	// 			}
	// 		}

	// 		cout << "it: " << it << " stepSize: " << stepSize << " tolerance: " << tolerance << " shiftValue: " << shiftValue << "dist: " << currentDist << endl;
	// 		// cout << "GPmean: " << vec_GPMean;

	// 		// addToSlackの初期化
	// 		for (int weight_i = 0; weight_i < vec_weight.size(); ++weight_i){
	// 			vec_addToSlack[weight_i] = 0;
	// 		}

	// 		// ここから各要素ごとに移動方向を決める処理
	// 		for (int changeWeightValue_i = 0; changeWeightValue_i < vec_weight.size(); ++changeWeightValue_i){


	// 			RVec vec_plusOV(ASY.cols);
	// 			RVec vec_minusOV(ASY.cols);
	// 			// 初期化
	// 			for (int OV_i = 0; OV_i < vec_plusOV.cols; ++OV_i){
	// 				vec_plusOV[OV_i] = 0;
	// 				vec_minusOV[OV_i] = 0;
	// 			}

	// 			// 指定の要素をstepSizeぶん動かす
	// 			vec_nextPlusWeight = vec_currentWeight.m().clone();
	// 			vec_nextMinusWeight = vec_currentWeight.m().clone();
	// 			vec_nextPlusWeight[changeWeightValue_i] += shiftValue;
	// 			vec_nextMinusWeight[changeWeightValue_i] -= shiftValue;
	// 			// stepSizeを引いた結果ゼロになる場合は、ゼロにする
	// 			if(vec_nextMinusWeight[changeWeightValue_i] <= 0){
	// 				vec_nextMinusWeight[changeWeightValue_i] = 0;
	// 			}
	// 			// cout << "currentWeight: " << vec_currentWeight << endl;
	// 			// cout << "plusWeight: " << vec_nextPlusWeight << endl;
	// 			// cout << "minusWeight: " << vec_nextMinusWeight << endl;

	// 			// プラス方向に移動した場合とマイナス方向に移動した場合のOVを計算
	// 			for (int weight_i = 0; weight_i < vec_weight.size(); ++weight_i){
	// 				for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 					vec_plusOV[ASYCol_i] += vec_nextPlusWeight[weight_i] * ASY(weight_i, ASYCol_i);
	// 					vec_minusOV[ASYCol_i] += vec_nextMinusWeight[weight_i] * ASY(weight_i, ASYCol_i);
	// 				}
	// 			}

	// 			// 新しい重みのL2ノルムを計算
	// 			double plusWeightNorm = calcL2norm(vec_nextPlusWeight.m());
	// 			double minusWeightNorm = calcL2norm(vec_nextMinusWeight.m());

	// 			// プラスとマイナスそれぞれでの目的関数の結果との距離を計算
	// 			double plusDist = 0;
	// 			double minusDist = 0;
	// 			for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 				// plusDist += pow(vec_plusOV[ASYCol_i] - vec_GPMean[ASYCol_i], 2);
	// 				// minusDist += pow(vec_minusOV[ASYCol_i] - vec_GPMean[ASYCol_i], 2);
 // 					plusDist += pow(vec_plusOV[ASYCol_i] - vec_GPMean[ASYCol_i], 2) + alpha * plusWeightNorm;
	// 				minusDist += pow(vec_minusOV[ASYCol_i] - vec_GPMean[ASYCol_i], 2) + alpha * minusWeightNorm;
	// 			}
	// 			// cout << "currentDist: " << currentDist << " plusDist: " << plusDist << " minusDist: " << minusDist << endl;

	// 			// 移動方向と距離に応じてスラーク変数を更新
	// 			// プラス方向移動時に距離が短くなった場合
	// 			if (plusDist < currentDist){
	// 				// cout << "col:" << changeWeightValue_i << " update: " << currentDist << " -> " << plusDist << endl;
	// 				moveFlagOfAll = true;
	// 				vec_addToSlack[changeWeightValue_i] = shiftValue;
	// 			}
	// 			// マイナス方向移動時に距離が短くなった場合
	// 			if (minusDist < currentDist && vec_currentWeight[changeWeightValue_i] != 0){
	// 				// cout << "col:" << changeWeightValue_i << " update: " << currentDist << " -> " << minusDist << endl;
	// 				moveFlagOfAll = true;
	// 				vec_addToSlack[changeWeightValue_i] = -shiftValue;
	// 			}

	// 		}// weightの各要素についての処理ここまで

	// 		//動いた軸が存在しなければ、stepSizeを小さくする
	// 		if (!moveFlagOfAll){
	// 			stepSize /= 2;
	// 			it++;
	// 			cout << endl;
	// 			continue;
	// 		}

	// 		RVec vec_newOV(1, ASY.cols, CV_64F);
	// 		// 初期化
	// 		for (int OV_i = 0; OV_i < vec_newOV.cols; ++OV_i){
	// 			vec_newOV[OV_i] = 0;
	// 		}

	// 		// slackを更新
	// 		RVec vec_nextWeight(vec_weight.size());
	// 		vec_nextWeight = vec_weight + vec_slack + vec_addToSlack;

	// 		// OVを計算
	// 		for (int weight_i = 0; weight_i < vec_weight.size(); ++weight_i){
	// 			for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 				vec_newOV[ASYCol_i] += vec_nextWeight[weight_i] * ASY(weight_i, ASYCol_i);
	// 			}
	// 		}

	// 		double nextWeightNorm = calcL2norm(vec_nextWeight);

	// 		double newDist = 0;
	// 		for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 			newDist += pow(vec_newOV[ASYCol_i] - vec_GPMean[ASYCol_i], 2) + alpha * nextWeightNorm;
	// 		}
	// 		// cout << "nextWeight: " << vec_nextWeight << endl;
	// 		// cout << "newOV: " << vec_newOV << endl;
	// 		// cout << "constraint: " << alpha * nextWeightNorm << " newDist: " << newDist << endl;

	// 		if (newDist <= currentDist){
	// 			currentDist = newDist;
	// 			vec_slack += vec_addToSlack;
	// 		}else{
	// 			stepSize /= 2;
	// 		}
	// 		it++;
	// 		// cout << endl;
	// 	}// stepSizeがtoleranceを下回るまでループする処理ここまで

	// 	// 新たな重みを返す
	// 	RVec vec_newWeight(1, vec_weight.size(), CV_64F);
	// 	for (int weight_i = 0; weight_i < vec_weight.size(); ++weight_i){
	// 		vec_newWeight[weight_i] = vec_weight[weight_i] + vec_slack[weight_i];
	// 	}
	// 	// cout << "weight: " << vec_weight << endl;
	// 	// cout << "new weight: " << vec_newWeight << endl;

	// 	double dist = 0;
	// 	RVec OV(1, ASY.cols, CV_64F);
	// 	for (int OVCol_i = 0; OVCol_i < OV.cols; ++OVCol_i){
	// 		OV[OVCol_i] = 0;
	// 	}
	// 	for (int weight_i = 0; weight_i < vec_newWeight.cols; ++weight_i){
	// 		for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 			OV[ASYCol_i] += vec_newWeight[weight_i] * ASY(weight_i, ASYCol_i);
	// 		}
	// 	}
	// 	for (int ASYCol_i = 0; ASYCol_i < ASY.cols; ++ASYCol_i){
	// 		dist += pow(OV[ASYCol_i] - vec_GPMean[ASYCol_i], 2);
	// 	}
	// 	cout << "dist: " << dist << endl;
	// 	return vec_newWeight;
	// }

	cv::Mat MathU::normalizeHistogram(const cv::Mat &src){
		cv::Mat dst;
		// minとmaxを決定する
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);

		cv::convertScaleAbs(src, dst, 255.0/(maxVal-minVal), (255.0/(maxVal-minVal))*(-minVal));
		return dst;
	}

	double MathU::calcEuclideanDist(const cv::Mat point1, const cv::Mat point2){
		// e_isRVec(point1);
		// e_isRVec(point2);

		// ベクトルの行数が違う場合は例外
		if(point1.cols != point2.cols){
			throw invalid_argument("in calcEuclideanDist invalid value is passed. vector nums are different.");
		}

		double dist = 0;
		for (int i = 0; i < point1.cols; ++i){
			dist += pow( point1.at<double>(0, i) - point2.at<double>(0, i), 2 );
		}

		return sqrt(dist);
	}
}


