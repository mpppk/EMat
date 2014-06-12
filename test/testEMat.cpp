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

TEST_F(EMatTest, RVecTest){
	ASSERT_THROW( mc::RVec rvec(getTempMat() ), invalid_argument );
	cv::Mat mat = ( cv::Mat_<double>(1, 3) << 1,2,3 );
	mc::RVec rvec = mat;
	EXPECT_EQ(3, rvec[2]);
	rvec[0] = 4;
	EXPECT_EQ(4, rvec[0]);
	// cout << rvec << endl;
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

// calcWCovMatが正しく動作しているかのテスト
TEST(MathUTest, calcEuclideanDistTest){
	cv::Mat point1 = (cv::Mat_<double>(1,3) << 1, 2, 3);
	cv::Mat point2 = (cv::Mat_<double>(1,3) << 3, 4, 4);
	double dist = mc::MathU::calcEuclideanDist(point1, point2);
	EXPECT_DOUBLE_EQ(dist, 3);
}

int main( int argc, char* argv[] ){
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}

