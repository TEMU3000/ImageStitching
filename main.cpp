#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <utility>
#define TOTAL_IMAGES 18
using namespace cv;
using namespace std;

typedef struct feature_point{
    int x;
    int y;
    vector<uint8_t> descriptor;
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
vector<vector<pair<int,int>>> good_matches(TOTAL_IMAGES);

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

int main(int argc, char *argv[])
{
    if(readImages() != 0){
        printf("Error reading image files!\n");
        return -1;
    };
    //shrinkImages(0.15);

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

    waitKey(0);
    return 0;
}

int readImages(){
    printf("Reading image files ...\n");
    /*src[0] = imread("set4/DSC_0014.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[0].empty()) return -1;
    src[1] = imread("set4/DSC_0015.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[1].empty()) return -1;
    src[2] = imread("set4/DSC_0016.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[2].empty()) return -1;
    src[3] = imread("set4/DSC_0017.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[3].empty()) return -1;
    src[4] = imread("set4/DSC_0018.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[4].empty()) return -1;
    src[5] = imread("set4/DSC_0019.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[5].empty()) return -1;
    src[6] = imread("set4/DSC_0020.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[6].empty()) return -1;
    src[7] = imread("set4/DSC_0021.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[7].empty()) return -1;
    src[8] = imread("set4/DSC_0022.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[8].empty()) return -1;
    src[9] = imread("set4/DSC_0023.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[9].empty()) return -1;
    src[10] = imread("set4/DSC_0024.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[10].empty()) return -1;
    src[11] = imread("set4/DSC_0025.jpg", CV_LOAD_IMAGE_COLOR);
    if(src[11].empty()) return -1;*/

    src[0] = imread("parrington/prtn00.jpg", CV_LOAD_IMAGE_COLOR); if(src[0].empty()) return -1;
    src[1] = imread("parrington/prtn01.jpg", CV_LOAD_IMAGE_COLOR); if(src[1].empty()) return -1;
    src[2] = imread("parrington/prtn02.jpg", CV_LOAD_IMAGE_COLOR); if(src[2].empty()) return -1;
    src[3] = imread("parrington/prtn03.jpg", CV_LOAD_IMAGE_COLOR); if(src[3].empty()) return -1;
    src[4] = imread("parrington/prtn04.jpg", CV_LOAD_IMAGE_COLOR); if(src[4].empty()) return -1;
    src[5] = imread("parrington/prtn05.jpg", CV_LOAD_IMAGE_COLOR); if(src[5].empty()) return -1;
    src[6] = imread("parrington/prtn06.jpg", CV_LOAD_IMAGE_COLOR); if(src[6].empty()) return -1;
    src[7] = imread("parrington/prtn07.jpg", CV_LOAD_IMAGE_COLOR); if(src[7].empty()) return -1;
    src[8] = imread("parrington/prtn08.jpg", CV_LOAD_IMAGE_COLOR); if(src[8].empty()) return -1;
    src[9] = imread("parrington/prtn09.jpg", CV_LOAD_IMAGE_COLOR); if(src[9].empty()) return -1;
    src[10] = imread("parrington/prtn10.jpg", CV_LOAD_IMAGE_COLOR); if(src[10].empty()) return -1;
    src[11] = imread("parrington/prtn11.jpg", CV_LOAD_IMAGE_COLOR); if(src[11].empty()) return -1;
    src[12] = imread("parrington/prtn12.jpg", CV_LOAD_IMAGE_COLOR); if(src[12].empty()) return -1;
    src[13] = imread("parrington/prtn13.jpg", CV_LOAD_IMAGE_COLOR); if(src[13].empty()) return -1;
    src[14] = imread("parrington/prtn14.jpg", CV_LOAD_IMAGE_COLOR); if(src[14].empty()) return -1;
    src[15] = imread("parrington/prtn15.jpg", CV_LOAD_IMAGE_COLOR); if(src[15].empty()) return -1;
    src[16] = imread("parrington/prtn16.jpg", CV_LOAD_IMAGE_COLOR); if(src[16].empty()) return -1;
    src[17] = imread("parrington/prtn17.jpg", CV_LOAD_IMAGE_COLOR); if(src[17].empty()) return -1;

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
    printf("collect and circle feature points ...\n");
    for(int z=0; z < TOTAL_IMAGES; z++){
        vector<Feature_Point> image_fp;
        for(int i = 1; i < I[z].rows-1 ; i++ ) {
            for(int j = 1; j < I[z].cols-1 ; j++ ) {
                if( R[z].at<float>(i,j) > 750000000000.0 ) {
                    if(R[z].at<float>(i,j) > R[z].at<float>(i-1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j+1) ){
                        circle( src[z], Point( j, i ), 5, Scalar(255,190,30), 2, 8, 0 );

                        Feature_Point fp;
                        fp.x = i;
                        fp.y = j;
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-2,j-2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-2,j-1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-2,j));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-2,j+1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-2,j+2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-1,j-2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-1,j-1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-1,j));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-1,j+1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i-1,j+2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i,j-2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i,j-1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i,j));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i,j+1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i,j+2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+1,j-2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+1,j-1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+1,j));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+1,j+1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+1,j+2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+2,j-2));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+2,j-1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+2,j));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+2,j+1));
                        fp.descriptor.push_back(I[z].at<uint8_t>(i+2,j+2));

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
        vector<pair<int,int>> matchingpoints;
        int min_dist = 2147483647;
        for(int i=0; i < features[z].size(); i++){
            int dist = 2147483647;
            int matching = -1;
            for(int j=0; j < features[z+1].size(); j++){
                int d = calc_dist(&features[z][i], &features[z+1][j]);
                if(d < dist){
                    dist = d;
                    matching = j;
                }
            }
            if( dist < min_dist ) min_dist = dist;
            matchingpoints.push_back(make_pair(matching, dist));
        }
        printf("min_dist: %d\n", min_dist);
        Mat window1 = src[z].clone();
        Mat window2 = src[z+1].clone();
        for(int i=0; i < features[z].size(); i++){
            if( matchingpoints[i].second <= max(2*min_dist, 20000)){
                good_matches[z].push_back(make_pair(i, matchingpoints[i].first));
                circle( window1, Point( features[z][i].y, features[z][i].x ), 5, Scalar(20,20,255), 2, 8, 0 );
                circle( window2, Point( features[z+1][matchingpoints[i].first].y, features[z+1][matchingpoints[i].first].x ), 5, Scalar(20,20,255), 2, 8, 0 );
            }
        }

        char windowname[10]; char windowname2[10];
        sprintf(windowname, "img%d", z);
        namedWindow( windowname, CV_WINDOW_NORMAL );
        imshow( windowname, window1 );
        sprintf(windowname2, "img%d", z+1);
        namedWindow( windowname2, CV_WINDOW_NORMAL );
        imshow( windowname2, window2 );
        waitKey(0);
        destroyWindow(windowname);
        destroyWindow(windowname2);
    }
}

int calc_dist(const Feature_Point *a, const Feature_Point *b){
    int d = 0;
    for(int i=0; i < 25; i++){
        int tmp = (a -> descriptor[i]) - (b -> descriptor[i]);
        d += tmp * tmp;
    }
    return d;
}

