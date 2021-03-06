#include <gtest/gtest.h>
#include <EMat.h>

using namespace std;

vector< vector<string> > getData(){
	vector< vector<string> > retVec;
	retVec.push_back(vector<string>{"1","9"});
	retVec.push_back(vector<string>{"2","8"});
	retVec.push_back(vector<string>{"3","7"});
	retVec.push_back(vector<string>{"4","6"});
	retVec.push_back(vector<string>{"5","5"});
	retVec.push_back(vector<string>{"6","4"});
	retVec.push_back(vector<string>{"7","3"});
	retVec.push_back(vector<string>{"8","2"});
	retVec.push_back(vector<string>{"9","1"});
	return retVec;
}

vector< vector<string> > getWeight(){
	vector< vector<string> > retVec;
	vector<string> v{"1"};
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	retVec.push_back(v);
	return retVec;
}

inline int toInt(std::string s) {int v; std::istringstream sin(s);sin>>v;return v;}
template<class T> inline std::string toString(T x) {std::ostringstream sout;sout<<x;return sout.str();}

// startValueから連番の値を振った行列を返す
vector< vector<string> > getTempVec(int startValue = 1, int rowsNum = 3, int colsNum = 3){
	int value = startValue;
	vector< vector<string> > contents;

	for (int row_i = 0; row_i < rowsNum; ++row_i){
		vector<string> content;
		for (int i = 0; i < colsNum; ++i){
			content.push_back(toString<int>(value));
			value++;
		}
		contents.push_back(content);
	}
	return contents;
}

cv::Mat getTempMat(int cnt = 1, const int row = 3, const int col = 3){
	cv::Mat mat(row, col, CV_64F);
	for (int row_i = 0; row_i < mat.rows; ++row_i){
		for (int col_i = 0; col_i < mat.cols; ++col_i){
			mat.at<double>(row_i, col_i) = cnt;
			cnt++;
		}
	}
	return mat;
}

class RVecTest : public ::testing::Test{
protected:
	string fileDirPass;

	virtual void SetUp(){
		// ファイルの書きだし先
		fileDirPass = "../test/TestRVec/";
	}
};

TEST_F(RVecTest, constructorTest){
	ASSERT_THROW( mc::RVec rvec(getTempMat() ), invalid_argument );

	// vector<double>からRVecを生成する
	vector<double> vec{1, 2, 3};
	mc::RVec rvec(vec);
	EXPECT_EQ(1, rvec[0]);

	// vector<string>からRVecを生成する
	vector<string> vec2{"1", "2", "3"};
	mc::RVec rvec2(vec2);
	EXPECT_EQ(1, rvec2[0]);
}

TEST_F(RVecTest, operatorTest){
	cv::Mat mat = ( cv::Mat_<double>(1, 3) << 1,2,3 );
	mc::RVec rvec = mat;
	EXPECT_EQ(3, rvec[2]);
	rvec[0] = 4;
	EXPECT_EQ(4, rvec[0]);
}

class EMatTest : public ::testing::Test{
protected:
	string fileDirPass;

	virtual void SetUp(){
		// ファイルの書きだし先
		fileDirPass = "../test/TestEMat/";
	}
};

TEST_F(EMatTest, constructorTest){
	EXPECT_EQ( mc::EMat( getTempMat() )(0, 0), 1 );
	EXPECT_EQ( mc::EMat( getTempVec() )(0, 0), 1 );
}

TEST_F(EMatTest, toVecTest){
	// toVec(), toVec(cv::Mat)
	mc::EMat emat(getTempVec());
	vector< vector<string> > vec = emat.toVec();
	
	EXPECT_EQ(1, toInt(vec[0][0]));
	EXPECT_EQ(9, toInt(vec[2][2]));

	// toVec(vector<cv::Mat>)
	vector<mc::EMat> emats;
	mc::EMat emat2(getTempVec(10));
	mc::EMat emat3(getTempVec(20));
	emats.push_back(emat);
	emats.push_back(emat2);
	emats.push_back(emat3);

	vector< vector< vector<string> > > ematsVec = mc::MatU::toVec(mc::EMat::cast(emats));
	EXPECT_EQ(1, toInt(ematsVec[0][0][0]));
	EXPECT_EQ(10, toInt(ematsVec[1][0][0]));
	EXPECT_EQ(28, toInt(ematsVec[2][2][2]));
}

TEST_F(EMatTest, castTest){
	vector<mc::EMat> emats;
	emats.push_back(mc::EMat(getTempVec()));
	emats.push_back(mc::EMat(getTempVec(10)));
	emats.push_back(mc::EMat(getTempVec(20)));
	vector<cv::Mat> mats = mc::EMat::cast(emats);
	EXPECT_EQ(20, mats[2].at<double>(0, 0));
}

TEST_F(EMatTest, normalizeTest){
	mc::EMat emat(getData());
	mc::EMat emat2 = emat.toNormalizedMat();
	mc::EMat emat3 = emat2.toUnnormalizedMat(emat.toEachColsMean(), emat.toEachColsSD());
	EXPECT_EQ(1, emat3.m().at<double>(0, 0));
	EXPECT_EQ(5, emat3.m().at<double>(4, 0));
	EXPECT_EQ(9, emat3.m().at<double>(0, 1));
}

class MatUTest : public ::testing::Test{
	protected:
		string fileDirPass;

		virtual void SetUp(){
			// ファイルの書きだし先
			fileDirPass = "../test/TestEMat/";
		}
};

TEST_F(MatUTest, toVecTest){
	// vector< vector<string> > toVec(const cv::Mat&)
	auto vec = mc::MatU::toVec( getTempMat() );
	EXPECT_EQ( 1, toInt(vec[0][0]) );
	EXPECT_EQ( 5, toInt(vec[1][1]) );

	// vector< vector< vector<string> > > MatU::toVec(const vector<cv::Mat>& mats){
	vector<cv::Mat> mats;
	mats.push_back(getTempMat());
	mats.push_back(getTempMat(10));
	mats.push_back(getTempMat(20));
	auto vec2 = mc::MatU::toVec(mats);
	EXPECT_EQ( 20, toInt(vec2[2][0][0]) );
}

TEST_F(MatUTest, toMatTest){
	// cv::Mat toMat(const vector<string>& content);
	cv::Mat mat = mc::MatU::toMat(vector<string>{"1", "2", "3"});
	EXPECT_EQ( 3, mat.at<double>(0, 2) );

	// cv::Mat toMat(const vector< vector<string> >& contents);
	cv::Mat mat2 = mc::MatU::toMat(getTempVec());
	EXPECT_EQ( 3, mat.at<double>(0, 2) );

	// cv::Mat toMat(const vector<RVec>& vecs)
	vector<mc::RVec> vecs;
	vecs.push_back( cv::Mat::ones(1, 3, CV_64F) );
	vecs.push_back( cv::Mat::ones(1, 3, CV_64F) * 2 );
	vecs.push_back( cv::Mat::ones(1, 3, CV_64F) * 3 );
	cv::Mat mat3 = mc::MatU::toMat( vecs );
	EXPECT_EQ( 1, mat3.at<double>(0, 0) );
	EXPECT_EQ( 2, mat3.at<double>(1, 1) );
	EXPECT_EQ( 3, mat3.at<double>(2, 2) );

	// 列数が違う行があったときに例外を吐くかチェック
	vecs.push_back(cv::Mat::ones(1, 2, CV_64F) * 4);
	EXPECT_THROW( mc::MatU::toMat( vecs ), invalid_argument );
}

TEST_F(MatUTest, copyRowTest){
	cv::Mat mat1 = getTempMat();
	cv::Mat mat2 = getTempMat(10);
	mc::MatU::copyRow(mat1, mat2, 0, 0);
	// expected mat
	//   1  2  3
	//  13 14 15
	//  16 17 18 

	EXPECT_EQ( 1, mat2.at<double>(0, 0));
	EXPECT_EQ( 3, mat2.at<double>(0, 2));
	EXPECT_EQ(16, mat2.at<double>(2, 0));
}

TEST_F(MatUTest, copyColTest){
	cv::Mat mat1 = getTempMat();
	cv::Mat mat2 = getTempMat(10);
	mc::MatU::copyCol(mat1, mat2, 0, 0);
	// expected mat
	//  1 11 12
	//  4 14 15
	//  7 17 18 

	EXPECT_EQ( 1, mat2.at<double>(0, 0));
	EXPECT_EQ( 7, mat2.at<double>(2, 0));
	EXPECT_EQ(11, mat2.at<double>(0, 1));
}

TEST_F(MatUTest, mergeRVecTest){
	cv::Mat mat = (cv::Mat_<double>(1,3) << 1, 2, 3);
	cv::Mat mat2 = (cv::Mat_<double>(1,3) << 4, 5, 6);
	cv::Mat mat3 = mc::MatU::mergeRVec(mat, mat2);
	EXPECT_EQ(6, mat3.at<double>(0, 5));

	vector<cv::Mat> mats;
	mats.push_back(mat);
	mats.push_back(mat2);
	mat3 = mc::MatU::mergeRVec(mats);
	EXPECT_EQ(6, mat3.at<double>(0, 5));

	cv::Mat mat4(2, 3, CV_64F);
	for(int i = 0; i < mat.cols; i++)	mat4.at<double>(0, i) = mat.at<double>(0, i);
	for(int i = 0; i < mat2.cols; i++)	mat4.at<double>(1, i) = mat2.at<double>(0, i);
	mc::RVec rvec = mc::MatU::mergeRVec(mat4);
	EXPECT_EQ(6, rvec[5]);
}

TEST_F(MatUTest, mergeMatToSideTest){
	cv::Mat mat1 = getTempMat();
	cv::Mat mat2 = getTempMat(10, 2, 4);

	cv::Mat mat3 = mc::MatU::mergeMatToSide(mat1, mat2, -99);
	// expected mat
	// 1   2   3  10  11  12  13
	// 4   5   6  14  15  16  17
	// 7   8   9 -99 -99 -99 -99

	EXPECT_EQ(3, mat3.rows);
	EXPECT_EQ(7, mat3.cols);
	EXPECT_EQ(9, mat3.at<double>(2, 2));
	EXPECT_EQ(10, mat3.at<double>(0, 3));
	EXPECT_EQ(17, mat3.at<double>(1, 6));
	EXPECT_EQ(-99, mat3.at<double>(2, 6));

	mat3 = mc::MatU::mergeMatToSide(mat2, mat1, -99);
	// expected mat
	//  10  11  12  13  1   2   3
	//  14  15  16  17  4   5   6
	// -99 -99 -99 -99  7   8   9

	EXPECT_EQ(3, mat3.rows);
	EXPECT_EQ(7, mat3.cols);
	EXPECT_EQ(10, mat3.at<double>(0, 0));
	EXPECT_EQ(3, mat3.at<double>(0, 6));
	EXPECT_EQ(7, mat3.at<double>(2, 4));
}

TEST_F(MatUTest, mergeMatToBottomTest){
	cv::Mat mat1 = getTempMat();
	cv::Mat mat2 = getTempMat(10, 2, 4);

	cv::Mat mat3 = mc::MatU::mergeMatToBottom(mat1, mat2, -99);
	// expected mat
	//    1   2   3 -99 
	//    4   5   6 -99
	//    7   8   9 -99
	//   10  11  12  13
	//   14  15  16  17

	EXPECT_EQ(5, mat3.rows);
	EXPECT_EQ(4, mat3.cols);
	EXPECT_EQ(-99, mat3.at<double>(0, 3));
	EXPECT_EQ(10, mat3.at<double>(3, 0));
	EXPECT_EQ(17, mat3.at<double>(4, 3));

	mat3 = mc::MatU::mergeMatToBottom(mat2, mat1, -99);
	// expected mat
	//   10  11  12  13
	//   14  15  16  17
	//    1   2   3 -99 
	//    4   5   6 -99
	//    7   8   9 -99

	EXPECT_EQ(5, mat3.rows);
	EXPECT_EQ(4, mat3.cols);
	EXPECT_EQ(-99, mat3.at<double>(2, 3));
	EXPECT_EQ(1, mat3.at<double>(2, 0));
	EXPECT_EQ(17, mat3.at<double>(1, 3));

	vector<cv::Mat> mats;
	mats.push_back(getTempMat());
	mats.push_back(getTempMat(10, 2, 4));
	mats.push_back(getTempMat(18, 3, 1));

	mat3 = mc::MatU::mergeMatToBottom(mats, -99);
	// expected mat
	//   1   2   3 -99
	//   4   5   6 -99
	//   7   8   9 -99
	//  10  11  12  13
	//  14  15  16  17
	//  18 -99 -99 -99
	//  19 -99 -99 -99
	//  20 -99 -99 -99

	EXPECT_EQ(8, mat3.rows);
	EXPECT_EQ(4, mat3.cols);
	EXPECT_EQ(-99, mat3.at<double>(2, 3));
	EXPECT_EQ(17, mat3.at<double>(4, 3));
	EXPECT_EQ(18, mat3.at<double>(5, 0));
	EXPECT_EQ(-99, mat3.at<double>(5, 1));
}

TEST_F(MatUTest, writeCSVTest){
	mc::MatU::writeCSV( mc::RVec( getTempMat(1, 1, 10) ), "./writeCSVTest.csv" );
}

class MathUTest : public ::testing::Test{
protected:
	string fileDirPass;

	virtual void SetUp(){
		// ファイルの書きだし先
		fileDirPass = "../test/TestMathFuncs/";
	}
};

TEST(MathUTest, normTest){
	cv::Mat mat = (cv::Mat_<double>(1,3) << 1, -2, 3);
	double l1norm = mc::MathU::calcL1norm(mat);
	EXPECT_EQ(6, l1norm);
	
	cv::Mat mat2 = (cv::Mat_<double>(1,4) << 2, -2, 2, -2);
	double l2norm = mc::MathU::calcL2norm(mat2);
	EXPECT_EQ(4, l2norm);

	cv::Mat mat3 = (cv::Mat_<double>(1,3) << 1, -2, 3);
	double sum = mc::MathU::calcTotalSum(mat3);
	EXPECT_EQ(2, sum);
}

// calcWCovMatが正しく動作しているかのテスト
TEST(MathUTest, calcWCovTest){
	mc::EMat data(getData());
	mc::EMat weight(getWeight());
	// data.createByVec(getData());
	// weight.createByVec(getWeight());

	bitset<mc::MathU::CovMatOptionsNum> flags;

	// 最後に重みで割らない場合
	mc::EMat covarMat = mc::MathU::calcWCovMat(data.m(), mc::RVec(weight.m().t()), flags);
	EXPECT_EQ(60, covarMat(0, 0));
	EXPECT_EQ(-60, covarMat(1, 0));
	EXPECT_EQ(60, covarMat(1, 1));

	// 最後に重みで割る場合
	flags.set(mc::MathU::SCALE);
	covarMat = mc::MathU::calcWCovMat(data.m(), mc::RVec(weight.m().t()), flags);
	EXPECT_DOUBLE_EQ(60/9.0, covarMat(0, 0));
	EXPECT_DOUBLE_EQ(-60/9.0, covarMat(1, 0));
	EXPECT_DOUBLE_EQ(60/9.0, covarMat(1, 1));
}

// ユークリッド距離の計算が正しく動作しているかのテスト
TEST(MathUTest, calcEuclideanDistTest){
	cv::Mat point1 = (cv::Mat_<double>(1,3) << 1, 2, 3);
	cv::Mat point2 = (cv::Mat_<double>(1,3) << 3, 4, 4);
	double dist = mc::MathU::calcEuclideanDist(point1, point2);
	EXPECT_DOUBLE_EQ(dist, 3);
}

// マハラノビス距離の計算が正しく動作しているかのテスト
TEST(MathUTest, calcMahalanobisDistTest){
	cv::Mat point1 = (cv::Mat_<double>(1,2) << 1, 2);
	cv::Mat point2 = (cv::Mat_<double>(1,2) << 3, 4);
	cv::Mat icover = (cv::Mat_<double>(2,2) << 1, 0, 0, 1);
	double dist = mc::MathU::calcMahalanobisDist(point1, point2, icover);
	double dist2 = cv::Mahalanobis(point1, point2, icover);
	EXPECT_DOUBLE_EQ(dist, dist2);
}

// 移動平均の計算が正しく行われているか
TEST(MathUTest, movingAverageTest){
	cv::Mat mat = getTempMat();
	cv::Mat MAMat = mc::MathU::movingAverage(mat, 3);
	EXPECT_DOUBLE_EQ(2, MAMat.at<double>(0, 2));
	EXPECT_DOUBLE_EQ(8, MAMat.at<double>(2, 2));

	cv::Mat mat2 = getTempMat(1, 10, 2);
	cv::Mat MAMat2 = mc::MathU::movingAverageToEachCol(mat2, 2);
	EXPECT_DOUBLE_EQ(2, MAMat2.at<double>(1, 0));
	EXPECT_DOUBLE_EQ(19, MAMat2.at<double>(9, 1));
}

// 時間解像度の変更が正しく行われるか
TEST(MathUTest, temporalResolutionTest){
	const int WIDTH = 3;
	const int XDIM = 2;
	const int YDIM = 2;
	cv::Mat mat = getTempMat(1, 1, 10);
	auto map = mc::MathU::temporalResolution(mat, WIDTH, XDIM);
	EXPECT_EQ(2, map.at("x").at<double>(0, 0));
	EXPECT_EQ(5, map.at("x").at<double>(0, 1));
	EXPECT_EQ(5, map.at("y").at<double>(0, 0));
	EXPECT_EQ(6, map.at("x").at<double>(4, 0));
	EXPECT_EQ(9, map.at("x").at<double>(4, 1));
	EXPECT_EQ(9, map.at("y").at<double>(4, 0));
	EXPECT_EQ(5, map.at("x").rows);
	EXPECT_EQ(XDIM, map.at("x").cols);
	EXPECT_EQ(5, map.at("y").rows);

	// yがxと同じ次元のとき
	map = mc::MathU::temporalResolution(mat, WIDTH, XDIM, YDIM);
	EXPECT_EQ(2, map.at("x").at<double>(0, 0));
	EXPECT_EQ(2, map.at("y").at<double>(0, 0));
	EXPECT_EQ(9, map.at("x").at<double>(4, 1));
	EXPECT_EQ(9, map.at("y").at<double>(4, 1));
	EXPECT_EQ( map.at("x").rows , map.at("y").rows);
	EXPECT_EQ( map.at("x").cols , map.at("y").cols);

	// 不正な引数を与えたとき
	ASSERT_THROW( mc::MathU::temporalResolution(mat, WIDTH, XDIM, XDIM+1), invalid_argument );
}

// 次元を増やす処理が正しく行われているか
TEST(MathUTest, toMultiDimTest){
	cv::Mat mat = getTempMat();
	mc::RVec rvec = mat.reshape(1, 1);// 1チャンネル１行の行列に変換
	// cout << rvec.m() << endl;
	int dim = 3;
	cv::Mat newMat = mc::MathU::toMultiDim(rvec, dim);
	// cout << newMat << endl;
	EXPECT_DOUBLE_EQ(rvec.size(), newMat.rows);
	EXPECT_DOUBLE_EQ(dim, newMat.cols);
	EXPECT_DOUBLE_EQ(1, newMat.at<double>(0, 0));
	EXPECT_DOUBLE_EQ(2, newMat.at<double>(3, 0));
	EXPECT_DOUBLE_EQ(7, newMat.at<double>(8, 0));
	EXPECT_DOUBLE_EQ(9, newMat.at<double>(8, 2));
}

// 次元を増やす処理が正しく行われているか
TEST(MathUTest, toMultiDimFromMatTest){
	cv::Mat mat = getTempMat();
	int dim = 3;
	cv::Mat newMat = mc::MathU::toMultiDim(mat, dim);
	EXPECT_DOUBLE_EQ(mat.rows, newMat.rows);
	EXPECT_DOUBLE_EQ(mat.cols * dim, newMat.cols);
	EXPECT_DOUBLE_EQ(7, newMat.at<double>(2, 2));
	EXPECT_DOUBLE_EQ(8, newMat.at<double>(2, 5));
	EXPECT_DOUBLE_EQ(9, newMat.at<double>(2, 8));
}

TEST(MathUTest, correctBaseLineTest){
	mc::RVec vec = mc::MathU::correctBaseLine( getTempMat(1, 1, 10) );
	EXPECT_DOUBLE_EQ(0.109375, vec[3]);// これが正しいのか自信ない
}

class GaussianTest : public ::testing::Test{
protected:
	virtual void SetUp(){
		// ファイルの書きだし先
	}
};

TEST(GaussianTest, calcTest){
	const int sigma  = 1;
	const int width = 2;
	mc::RVec rvec = mc::MathU::Gaussian::convolute( getTempMat(1, 1, 10), sigma, width);
	// cout << rvec.m() << endl;
}

int main( int argc, char* argv[] ){
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}

