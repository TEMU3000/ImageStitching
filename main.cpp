#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <utility>
#include <cmath>
#include <ctime>

#define TOTAL_IMAGES 18
#define SHRINK_RATIO 0.5
#define FOCAL_LENGTH 704.5 //1145.87  //765.96

using namespace cv;
using namespace std;

typedef struct feature_point{
    int i;
    int j;
    vector<Vec3b> descriptor;
} Feature_Point;

vector<Mat> src(TOTAL_IMAGES);
vector<Mat> I(TOTAL_IMAGES);
vector<Mat> Ix(TOTAL_IMAGES);
vector<Mat> Iy(TOTAL_IMAGES);
vector<Mat> IxSq(TOTAL_IMAGES);
vector<Mat> IySq(TOTAL_IMAGES);
vector<Mat> Ixy(TOTAL_IMAGES);
vector<Mat> SxSq(TOTAL_IMAGES);
vector<Mat> SySq(TOTAL_IMAGES);
vector<Mat> Sxy(TOTAL_IMAGES);
vector<Mat> R;
vector<vector<Feature_Point>> features(TOTAL_IMAGES);
vector<vector<pair<Feature_Point,Feature_Point>>> good_matches(TOTAL_IMAGES);
vector<Mat> cyl;
vector<Mat> transformation;
double f[] = {580.601, 579.297, 580.766, 579.218, 577.435, 577.44,
                    578.606, 580.336, 580.681, 585.638, 587.866, 584.383,
                    583.408, 581.838, 579.621, 578.245, 579.491};

int readImages();
void shrinkImages(const double m);
void intensity();
void gradient();
void products();
void gaussian();
void computeR(const double k);
void collect_fp();
void match_fp();
int calc_dist(const Feature_Point *a, const Feature_Point *b);
void cylindrical(bool do_cylindrical);
void ransac();
void align();

int main(int argc, char *argv[])
{
    if(readImages() != 0){
        printf("Error reading image files!\n");
        return -1;
    };
    //shrinkImages(SHRINK_RATIO);

    intensity();
    gradient();
    products();
    gaussian();

    computeR(0.04);

    collect_fp();

    // OPENCV cornerHarris
    /*Mat dst, dst_norm;
    cornerHarris( I[0], dst, 2, 3, 0.04, BORDER_DEFAULT );
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    for(int i = 0; i < I[0].rows ; i++ ) {
        for(int j = 0; j < I[0].cols; j++ ) {
            if( (int) dst_norm.at<float>(i,j) > 90 ) {
                circle( src[0], Point( j, i ), 3, Scalar(0,255,255), 1, 8, 0 );
            }
        }
    }
    namedWindow( "OPENCV Corner", CV_WINDOW_AUTOSIZE );
    imshow( "OPENCV Corner", src[0] );*/

    match_fp();

    cylindrical(true);
    srand (time(NULL));
    ransac();

    align();

    waitKey(0);
    return 0;
}

int readImages(){
    printf("Reading image files ...\n");
    /*src[0] = imread("set4/DSC_0014.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("set4/DSC_0015.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("set4/DSC_0016.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("set4/DSC_0017.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("set4/DSC_0018.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("set4/DSC_0019.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("set4/DSC_0020.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("set4/DSC_0021.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("set4/DSC_0022.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("set4/DSC_0023.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;
    src[10] = imread("set4/DSC_0024.jpg", CV_LOAD_IMAGE_COLOR); if(src[10].empty()) return -1;
    src[11] = imread("set4/DSC_0025.jpg", CV_LOAD_IMAGE_COLOR); if(src[11].empty()) return -1;*/

    src[0] = imread("parrington/prtn17.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("parrington/prtn16.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("parrington/prtn15.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("parrington/prtn14.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("parrington/prtn13.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("parrington/prtn12.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("parrington/prtn11.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("parrington/prtn10.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("parrington/prtn09.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("parrington/prtn08.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;
    src[10] = imread("parrington/prtn07.jpg", CV_LOAD_IMAGE_COLOR); if(src[10].empty()) return -1;
    src[11] = imread("parrington/prtn06.jpg", CV_LOAD_IMAGE_COLOR); if(src[11].empty()) return -1;
    src[12] = imread("parrington/prtn05.jpg", CV_LOAD_IMAGE_COLOR); if(src[12].empty()) return -1;
    src[13] = imread("parrington/prtn04.jpg", CV_LOAD_IMAGE_COLOR); if(src[13].empty()) return -1;
    src[14] = imread("parrington/prtn03.jpg", CV_LOAD_IMAGE_COLOR); if(src[14].empty()) return -1;
    src[15] = imread("parrington/prtn02.jpg", CV_LOAD_IMAGE_COLOR); if(src[15].empty()) return -1;
    src[16] = imread("parrington/prtn01.jpg", CV_LOAD_IMAGE_COLOR); if(src[16].empty()) return -1;
    src[17] = imread("parrington/prtn00.jpg", CV_LOAD_IMAGE_COLOR); if(src[17].empty()) return -1;

    /*src[0] = imread("set1/DSC_0256.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("set1/DSC_0257.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("set1/DSC_0258.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("set1/DSC_0259.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("set1/DSC_0260.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("set1/DSC_0261.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("set1/DSC_0262.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("set1/DSC_0263.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("set1/DSC_0264.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("set1/DSC_0265.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;*/

    /*src[0] = imread("set2/DSC_0172.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("set2/DSC_0173.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("set2/DSC_0174.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("set2/DSC_0175.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("set2/DSC_0176.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("set2/DSC_0177.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("set2/DSC_0178.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("set2/DSC_0179.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("set2/DSC_0180.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("set2/DSC_0181.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;
    src[10] = imread("set2/DSC_0182.jpg", CV_LOAD_IMAGE_COLOR); if(src[10].empty()) return -1;*/

    /*src[0] = imread("set5/DSC_0035.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("set5/DSC_0036.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("set5/DSC_0037.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("set5/DSC_0038.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("set5/DSC_0039.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("set5/DSC_0040.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("set5/DSC_0041.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("set5/DSC_0042.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("set5/DSC_0026.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("set5/DSC_0027.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;
    src[10] = imread("set5/DSC_0028.jpg", CV_LOAD_IMAGE_COLOR); if(src[10].empty()) return -1;
    src[11] = imread("set5/DSC_0029.jpg", CV_LOAD_IMAGE_COLOR); if(src[11].empty()) return -1;
    src[12] = imread("set5/DSC_0030.jpg", CV_LOAD_IMAGE_COLOR); if(src[12].empty()) return -1;
    src[13] = imread("set5/DSC_0031.jpg", CV_LOAD_IMAGE_COLOR); if(src[13].empty()) return -1;
    src[14] = imread("set5/DSC_0032.jpg", CV_LOAD_IMAGE_COLOR); if(src[14].empty()) return -1;
    src[15] = imread("set5/DSC_0033.jpg", CV_LOAD_IMAGE_COLOR); if(src[15].empty()) return -1;
    src[16] = imread("set5/DSC_0034.jpg", CV_LOAD_IMAGE_COLOR); if(src[16].empty()) return -1;*/
    /*src[0] = imread("set5/DSC_0038.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("set5/DSC_0039.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("set5/DSC_0040.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("set5/DSC_0041.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("set5/DSC_0042.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("set5/DSC_0026.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("set5/DSC_0027.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("set5/DSC_0028.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("set5/DSC_0029.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("set5/DSC_0030.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;
    src[10] = imread("set5/DSC_0031.jpg", CV_LOAD_IMAGE_COLOR); if(src[10].empty()) return -1;
    src[11] = imread("set5/DSC_0032.jpg", CV_LOAD_IMAGE_COLOR); if(src[11].empty()) return -1;
    src[12] = imread("set5/DSC_0033.jpg", CV_LOAD_IMAGE_COLOR); if(src[12].empty()) return -1;
    src[13] = imread("set5/DSC_0034.jpg", CV_LOAD_IMAGE_COLOR); if(src[13].empty()) return -1;
    src[14] = imread("set5/DSC_0035.jpg", CV_LOAD_IMAGE_COLOR); if(src[14].empty()) return -1;
    src[15] = imread("set5/DSC_0036.jpg", CV_LOAD_IMAGE_COLOR); if(src[15].empty()) return -1;
    src[16] = imread("set5/DSC_0037.jpg", CV_LOAD_IMAGE_COLOR); if(src[16].empty()) return -1;*/

    return 0;
}

void shrinkImages(const double m){
    printf("Resizing images to %fx ...\n", m);
    for(int i=0; i < TOTAL_IMAGES; i++){
        resize(src[i], src[i], Size((src[i]).cols * m, (src[i]).rows * m),0,0,INTER_LINEAR);
    }
}

void intensity(){
    printf("Convert to intensity ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        cvtColor( src[i], I[i], CV_BGR2GRAY );
    }
}

void gradient(){
    printf("Calculate x and y gradient ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        Sobel(I[i], Ix[i], CV_32F , 1, 0, 3, BORDER_DEFAULT);
        Sobel(I[i], Iy[i], CV_32F , 0, 1, 3, BORDER_DEFAULT);
    }
}

void products(){
    printf("Compute products of derivatives ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        pow(Ix[i], 2.0, IxSq[i]);
        pow(Iy[i], 2.0, IySq[i]);
        multiply(Ix[i], Iy[i], Ixy[i]);
    }
}

void gaussian(){
    printf("Compute gaussian sums ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        GaussianBlur(IxSq[i], SxSq[i], Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
        GaussianBlur(IySq[i], SySq[i], Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
        GaussianBlur(Ixy[i], Sxy[i], Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    }
}

void computeR(const double k){
    printf("Compute det, trace, R using k = %f ...\n", k);

    for(int i=0; i < TOTAL_IMAGES; i++){
        Mat tmp1, tmp2;
        multiply(SxSq[i], SySq[i], tmp1);
        multiply(Sxy[i], Sxy[i], tmp2);

        Mat traceM = (Mat)SxSq[i] + (Mat)SySq[i];
        pow(traceM, 2.0, traceM);

        Mat r = (tmp1 - tmp2) - k * traceM;
        R.push_back(r);
        //normalize( R[i], R[i], 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    }
}

void collect_fp(){
    printf("collect feature points ...\n");
    for(int z=0; z < TOTAL_IMAGES; z++){
        vector<Feature_Point> image_fp;
        for(int i = 2; i < I[z].rows-2 ; i++ ) {
            for(int j = 2; j < I[z].cols-2 ; j++ ) {
                if( R[z].at<float>(i,j) > 400000000000.0 ) {
                    if(R[z].at<float>(i,j) > R[z].at<float>(i-1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j+1) ){
                        //circle( src[z], Point( j, i ), 5, Scalar(255,190,30), 2, 8, 0 );

                        Feature_Point fp;
                        fp.i = i;
                        fp.j = j;
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j+2));

                        image_fp.push_back(fp);
                    }
                }
            }
        }

        features[z] = image_fp;

        /*char windowname[50];
        sprintf(windowname, "img%d corner", z);
        namedWindow( windowname, CV_WINDOW_NORMAL );
        imshow( windowname, src[z] );*/
    }
    //waitKey(0);
}

void match_fp(){
    printf("matching feature points ...\n");

    for(int z=0; z < TOTAL_IMAGES-1; z++){
        printf("features in %d: %d\n", z, features[z].size());
        printf("features in %d: %d\n", z+1, features[z+1].size());

        Mat window1 = src[z].clone();
        Mat window2 = src[z+1].clone();
        for(int i=0; i < features[z].size(); i++){
            circle( window1, Point( features[z][i].j, features[z][i].i ), 5, Scalar(255,190,30), 2, 8, 0 );
        }
        for(int i=0; i < features[z+1].size(); i++){
            circle( window2, Point( features[z+1][i].j, features[z+1][i].i ), 5, Scalar(255,190,30), 2, 8, 0 );
        }

        vector<pair<int,int>> matchingpoints;
        int min_dist = 2147483647;
        for(int i=0; i < features[z].size(); i++){
            int dist = 2147483647;
            int point_b = -1;
            for(int j=0; j < features[z+1].size(); j++){
                if(features[z+1][j].j > src[z+1].cols*2/3 || features[z+1][j].j > features[z][i].j){
                    continue;
                }
                int d = calc_dist(&features[z][i], &features[z+1][j]);
                if(d < dist){
                    dist = d;
                    point_b = j;
                }
            }
            if( dist < min_dist ) min_dist = dist;
            matchingpoints.push_back(make_pair(point_b, dist));

        }
        printf("min_dist: %d\n", min_dist);

        vector<int> point_b_min_dist(features[z+1].size() , 2147483647);
        for(int i=0; i < features[z].size(); i++){
            if( matchingpoints[i].second <= max(2*min_dist, 100000)){
                if(matchingpoints[i].second < point_b_min_dist[matchingpoints[i].first]){
                    point_b_min_dist[matchingpoints[i].first] = matchingpoints[i].second;
                }
            }
        }

        for(int i=0; i < features[z].size(); i++){
            if( matchingpoints[i].second <= max(2*min_dist, 100000)){
                if(matchingpoints[i].second > point_b_min_dist[matchingpoints[i].first]){
                    continue;
                }
                if(features[z][i].j < src[z].cols*1/3){ continue; }
                good_matches[z].push_back(make_pair(features[z][i], features[z+1][matchingpoints[i].first]));
                circle( window1, Point( features[z][i].j, features[z][i].i ), 5, Scalar(20,20,255), 2, 8, 0 );
                circle( window2, Point( features[z+1][matchingpoints[i].first].j, features[z+1][matchingpoints[i].first].i ), 5, Scalar(20,20,255), 2, 8, 0 );
            }
        }

        /*char windowname[10]; char windowname2[10];
        sprintf(windowname, "img%d", z);
        namedWindow( windowname, CV_WINDOW_NORMAL );
        imshow( windowname, window1 );
        sprintf(windowname2, "img%d", z+1);
        namedWindow( windowname2, CV_WINDOW_NORMAL );
        imshow( windowname2, window2 );
        waitKey(0);
        destroyWindow(windowname);
        destroyWindow(windowname2);*/

        Mat matching_img(window1.rows*2, window1.cols, window1.type() );
        Mat part;
        part = matching_img(Rect(0,0,window1.cols,window1.rows));
        window1.copyTo(part);
        part =  matching_img(Rect(0,window1.rows,window1.cols,window1.rows));
        window2.copyTo(part);

        for(int i=0; i < good_matches[z].size(); i++){
            line(matching_img, Point(good_matches[z][i].first.j,good_matches[z][i].first.i),
                 Point(good_matches[z][i].second.j,good_matches[z][i].second.i+window1.rows), Scalar(255,0,0),2);
        }

        namedWindow( "matching img", CV_WINDOW_NORMAL );
        imshow( "matching img", matching_img );
        waitKey(0);
        destroyWindow("matching img");
    }
}

int calc_dist(const Feature_Point *a, const Feature_Point *b){
    int d = 0;
    for(int i=0; i < 25; i++){
        int tmp = (a -> descriptor[i])[0] - (b -> descriptor[i])[0];
        d += tmp * tmp;
        tmp = (a -> descriptor[i])[1] - (b -> descriptor[i])[1];
        d += tmp * tmp;
        tmp = (a -> descriptor[i])[2] - (b -> descriptor[i])[2];
        d += tmp * tmp;
    }
    return d;
}

void cylindrical(bool do_cylindrical){
    printf("cylindrical projection ...\n");
    if(!do_cylindrical){
        for(int z=0; z < TOTAL_IMAGES; z++){
            cyl.push_back(src[z]);
        }
        return;
    }

    for(int z=0; z < TOTAL_IMAGES; z++){
        int rows = src[z].rows;
        int cols = src[z].cols;
        Mat m_cyl(rows, cols, CV_8UC3, Scalar(0, 0, 0));

        for(int i=0; i < rows; i++){
            for(int j=0; j < cols; j++){
                int y = i - (rows/2);
                int x = j - (cols/2);
                int x_cyl = FOCAL_LENGTH * atan( x / FOCAL_LENGTH );
                int y_cyl = FOCAL_LENGTH * y / sqrt( x*x + FOCAL_LENGTH*FOCAL_LENGTH );
                m_cyl.at<Vec3b>(y_cyl + rows/2, x_cyl + cols/2) = src[z].at<Vec3b>(i,j);
            }
        }
        cyl.push_back(m_cyl);

        if(z < TOTAL_IMAGES-1){
            // project matching feature points
            for(int i=0; i < good_matches[z].size(); i++){
                int y = good_matches[z][i].first.i - (rows/2);
                int x = good_matches[z][i].first.j - (cols/2);
                good_matches[z][i].first.j = FOCAL_LENGTH * atan( x / FOCAL_LENGTH ) + cols/2;
                good_matches[z][i].first.i = FOCAL_LENGTH * y / sqrt( x*x + FOCAL_LENGTH*FOCAL_LENGTH ) + rows/2;

                y = good_matches[z][i].second.i - (rows/2);
                x = good_matches[z][i].second.j - (cols/2);
                good_matches[z][i].second.j = FOCAL_LENGTH * atan( x / FOCAL_LENGTH ) + cols/2;
                good_matches[z][i].second.i = FOCAL_LENGTH * y / sqrt( x*x + FOCAL_LENGTH*FOCAL_LENGTH ) + rows/2;

            }
        }

        /*char windowname[10];
        sprintf(windowname, "cyl%d", z);
        namedWindow( windowname, CV_WINDOW_FULLSCREEN );
        imshow( windowname, m_cyl );
        waitKey(0);*/
    }
}

void ransac(){
    printf("ransac ...\n");
    for(int z=0; z < TOTAL_IMAGES-1; z++){
        int max_inliers = -1;
        Mat bestM;
        for(int k=0; k < 500; k++){
            int match_size = good_matches[z].size();
            int a,b,c;
            a = rand() % match_size;
            do {b = rand() % match_size;} while (a == b);
            do {c = rand() % match_size;} while (a == c || b == c);
            Mat M1(4, 4, CV_32F, Scalar(0.0));
            //Mat M1(6, 4, CV_32F, Scalar(0.0));
            Mat M2(4, 1, CV_32F, Scalar(0.0));
            //Mat M2(6, 1, CV_32F, Scalar(0.0));
            M1.at<float>(0,0) = (float) good_matches[z][a].first.i;  M1.at<float>(0,1) = (float) good_matches[z][a].first.j;
            M1.at<float>(1,2) = (float) good_matches[z][a].first.i;  M1.at<float>(1,3) = (float) good_matches[z][a].first.j;
            M1.at<float>(2,0) = (float) good_matches[z][b].first.i;  M1.at<float>(2,1) = (float) good_matches[z][b].first.j;
            M1.at<float>(3,2) = (float) good_matches[z][b].first.i;  M1.at<float>(3,3) = (float) good_matches[z][b].first.j;
            //M1.at<float>(4,0) = (float) good_matches[z][c].first.i;  M1.at<float>(4,1) = (float) good_matches[z][c].first.j;
            //M1.at<float>(5,2) = (float) good_matches[z][c].first.i;  M1.at<float>(5,3) = (float) good_matches[z][c].first.j;
            M2.at<float>(0,0) = (float) good_matches[z][a].second.i;
            M2.at<float>(1,0) = (float) good_matches[z][a].second.j;
            M2.at<float>(2,0) = (float) good_matches[z][b].second.i;
            M2.at<float>(3,0) = (float) good_matches[z][b].second.j;
            //M2.at<float>(4,0) = (float) good_matches[z][c].second.i;
            //M2.at<float>(5,0) = (float) good_matches[z][c].second.j;
            Mat dst(4, 1, CV_32F, Scalar(0.0));

            //printf("  (%f,%f)->(%f,%f)\n",M1.at<float>(0,0),M1.at<float>(0,1),M2.at<float>(0,0),M2.at<float>(1,0));
            //printf("  (%f,%f)->(%f,%f)\n",M1.at<float>(2,0),M1.at<float>(2,1),M2.at<float>(2,0),M2.at<float>(3,0));

            solve(M1,M2,dst,DECOMP_SVD);

            Mat M(2, 2, CV_32F, Scalar(0));
            M.at<float>(0,0) = dst.at<float>(0,0);
            M.at<float>(0,1) = dst.at<float>(1,0);
            M.at<float>(1,0) = dst.at<float>(2,0);
            M.at<float>(1,1) = dst.at<float>(3,0);

            //printf("  %f %f\n  %f %f\n",M.at<float>(0,0),M.at<float>(0,1),M.at<float>(1,0),M.at<float>(1,1));
            //int q; cin >> q;

            // count inliers
            int inliers = 0;
            for(int i=0; i < match_size; i++){
                Mat m1(2, 1, CV_32F, Scalar(0));
                m1.at<float>(0,0) = (float) good_matches[z][i].first.i;
                m1.at<float>(1,0) = (float) good_matches[z][i].first.j;
                Mat m2(2, 1, CV_32F, Scalar(0));
                m2 = M * m1;

                float d1 = m2.at<float>(0,0) - good_matches[z][i].second.i;
                float d2 = m2.at<float>(1,0) - good_matches[z][i].second.j;
                if(d1*d1+d2*d2 < 50){
                    inliers++;
                }
            }

            if(inliers > max_inliers){
                max_inliers = inliers;
                bestM = M.clone();
            }

        }
        printf("max_inliers[%d]:%d\n",z,max_inliers);
        printf("  %f %f\n  %f %f\n",bestM.at<float>(0,0),bestM.at<float>(0,1),bestM.at<float>(1,0),bestM.at<float>(1,1));
        transformation.push_back(bestM);
    }
}

void align(){
    printf("alignment ...\n");
    Vec3b black(0,0,0);

    for(int z=0; z < TOTAL_IMAGES-1; z++){
        Mat panorama(cyl[0].rows, cyl[0].cols * 3, CV_8UC3, Scalar(0,0,0));
        cyl[z+1].copyTo(panorama(Rect(cyl[z].cols,0,cyl[z].cols,cyl[z].rows)));
        float m11 = transformation[z].at<float>(0,0);
        float m12 = transformation[z].at<float>(0,1);
        float m21 = transformation[z].at<float>(1,0);
        float m22 = transformation[z].at<float>(1,1);

        for(int i=0; i < cyl[z].rows; i++){
            for(int j=0; j < cyl[z].cols; j++){
                if(cyl[z].at<Vec3b>(i,j) == black ){ continue; }
                int new_i = (int) (i*m11 + j*m12);
                int new_j = (int) (i*m21 + j*m22);
                if(new_i > 0 && new_j > -cyl[z].cols && new_i < cyl[0].rows)
                    panorama.at<Vec3b>(new_i,new_j+cyl[z].cols) = cyl[z].at<Vec3b>(i,j);
            }
        }

        namedWindow( "panorama", CV_WINDOW_NORMAL );
        imshow( "panorama", panorama );
        waitKey(0);
    }
}
