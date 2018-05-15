#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#define TOTAL_IMAGES 12
using namespace cv;
using namespace std;

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
/*Mat img1, img2, img3, img4, img5, img6, img7, img8, img9, img10;
Mat I1, I2, I3, I4, I5, I6, I7, I8, I9, I10;
Mat Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, Ix7, Ix8, Ix9, Ix10;
Mat Iy1, Iy2, Iy3, Iy4, Iy5, Iy6, Iy7, Iy8, Iy9, Iy10;
Mat IxSq1, IxSq2, IxSq3, IxSq4, IxSq5, IxSq6, IxSq7, IxSq8, IxSq9, IxSq10;
Mat IySq1, IySq2, IySq3, IySq4, IySq5, IySq6, IySq7, IySq8, IySq9, IySq10;
Mat Ixy1, Ixy2, Ixy3, Ixy4, Ixy5, Ixy6, Ixy7, Ixy8, Ixy9, Ixy10;
Mat SxSq1, SxSq2, SxSq3, SxSq4, SxSq5, SxSq6, SxSq7, SxSq8, SxSq9, SxSq10;
Mat SySq1, SySq2, SySq3, SySq4, SySq5, SySq6, SySq7, SySq8, SySq9, SySq10;
Mat Sxy1, Sxy2, Sxy3, Sxy4, Sxy5, Sxy6, Sxy7, Sxy8, Sxy9, Sxy10;
Mat R;*/

int readImages();
void shrinkImages(double m);
void intensity();
void gradient();
void products();
void gaussian();
void computeR(double k);

int main(int argc, char *argv[])
{
    if(readImages() != 0){
        printf("Error reading image files!\n");
        return -1;
    };
    shrinkImages(0.25);

    intensity();
    gradient();
    products();
    gaussian();

    computeR(0.04);

    printf("circle\n");
    for(int z=0; z < TOTAL_IMAGES; z++){
        for(int i = 1; i < I[z].rows-1 ; i++ ) {
            for(int j = 1; j < I[z].cols-1 ; j++ ) {
                if( (int) R[z].at<float>(i,j) > 1800000000 ) {
                    if(R[z].at<float>(i,j) > R[z].at<float>(i-1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j+1) ){
                        circle( src[z], Point( j, i ), 5, Scalar(0,0,255), 2, 8, 0 );
                    }
                }
            }
        }
        char windowname[50];
        sprintf(windowname, "img%d corner", z);
        namedWindow( windowname, CV_WINDOW_NORMAL );
        imshow( windowname, src[z] );
    }

    // OPENCV lib
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

    waitKey(0);
    return 0;
}

int readImages(){
    printf("Reading image files...\n");
    src[0] = imread("set4/DSC_0014.jpg", CV_LOAD_IMAGE_COLOR);
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
    if(src[11].empty()) return -1;

    return 0;
}

void shrinkImages(double m){
    printf("Resizing images to %fx ...\n", m);
    for(int i=0; i < TOTAL_IMAGES; i++){
        resize(src[i], src[i], Size((src[i]).cols * m, (src[i]).rows * m),0,0,INTER_LINEAR);
    }
    /*resize(img1,img1, Size(img1.cols * m, img1.rows * m),0,0,INTER_LINEAR);
    resize(img2,img2, Size(img2.cols * m, img2.rows * m),0,0,INTER_LINEAR);
    resize(img3,img3, Size(img3.cols * m, img3.rows * m),0,0,INTER_LINEAR);
    resize(img4,img4, Size(img4.cols * m, img4.rows * m),0,0,INTER_LINEAR);
    resize(img5,img5, Size(img5.cols * m, img5.rows * m),0,0,INTER_LINEAR);
    resize(img6,img6, Size(img6.cols * m, img6.rows * m),0,0,INTER_LINEAR);
    resize(img7,img7, Size(img7.cols * m, img7.rows * m),0,0,INTER_LINEAR);
    resize(img8,img8, Size(img8.cols * m, img8.rows * m),0,0,INTER_LINEAR);
    resize(img9,img9, Size(img9.cols * m, img9.rows * m),0,0,INTER_LINEAR);
    resize(img10,img10, Size(img10.cols * m, img10.rows * m),0,0,INTER_LINEAR);*/
}

void intensity(){
    printf("Convert to intensity...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        cvtColor( src[i], I[i], CV_BGR2GRAY );
    }
    /*cvtColor( img1, I1, CV_BGR2GRAY );
    cvtColor( img2, I2, CV_BGR2GRAY );
    cvtColor( img3, I3, CV_BGR2GRAY );
    cvtColor( img4, I4, CV_BGR2GRAY );
    cvtColor( img5, I5, CV_BGR2GRAY );
    cvtColor( img6, I6, CV_BGR2GRAY );
    cvtColor( img7, I7, CV_BGR2GRAY );
    cvtColor( img8, I8, CV_BGR2GRAY );
    cvtColor( img9, I9, CV_BGR2GRAY );
    cvtColor( img10, I10, CV_BGR2GRAY );*/
}

void gradient(){
    printf("Calculate x and y gradient...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        Sobel(I[i], Ix[i], CV_32F , 1, 0, 3, BORDER_DEFAULT);
        Sobel(I[i], Iy[i], CV_32F , 0, 1, 3, BORDER_DEFAULT);
    }
    /*Sobel(I1, Ix1, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I1, Iy1, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I2, Ix2, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I2, Iy2, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I3, Ix3, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I3, Iy3, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I4, Ix4, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I4, Iy4, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I5, Ix5, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I5, Iy5, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I6, Ix6, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I6, Iy6, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I7, Ix7, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I7, Iy7, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I8, Ix8, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I8, Iy8, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I9, Ix9, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I9, Iy9, CV_32F , 0, 1, 3, BORDER_DEFAULT);
    Sobel(I10, Ix10, CV_32F , 1, 0, 3, BORDER_DEFAULT);
    Sobel(I10, Iy10, CV_32F , 0, 1, 3, BORDER_DEFAULT);*/
}

void products(){
    printf("Compute products of derivatives...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        pow(Ix[i], 2.0, IxSq[i]);
        pow(Iy[i], 2.0, IySq[i]);
        multiply(Ix[i], Iy[i], Ixy[i]);
    }
    /*pow(Ix1, 2.0, IxSq1);    pow(Iy1, 2.0, IySq1);    multiply(Ix1, Iy1, Ixy1);
    pow(Ix2, 2.0, IxSq2);    pow(Iy2, 2.0, IySq2);    multiply(Ix2, Iy2, Ixy2);
    pow(Ix3, 2.0, IxSq3);    pow(Iy3, 2.0, IySq3);    multiply(Ix3, Iy3, Ixy3);
    pow(Ix4, 2.0, IxSq4);    pow(Iy4, 2.0, IySq4);    multiply(Ix4, Iy4, Ixy4);
    pow(Ix5, 2.0, IxSq5);    pow(Iy5, 2.0, IySq5);    multiply(Ix5, Iy5, Ixy5);
    pow(Ix6, 2.0, IxSq6);    pow(Iy6, 2.0, IySq6);    multiply(Ix6, Iy6, Ixy6);
    pow(Ix7, 2.0, IxSq7);    pow(Iy7, 2.0, IySq7);    multiply(Ix7, Iy7, Ixy7);
    pow(Ix8, 2.0, IxSq8);    pow(Iy8, 2.0, IySq8);    multiply(Ix8, Iy8, Ixy8);
    pow(Ix9, 2.0, IxSq9);    pow(Iy9, 2.0, IySq9);    multiply(Ix9, Iy9, Ixy9);
    pow(Ix10, 2.0, IxSq10);    pow(Iy10, 2.0, IySq10);    multiply(Ix10, Iy10, Ixy10);*/
}

void gaussian(){
    printf("Compute gaussian sums...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        GaussianBlur(IxSq[i], SxSq[i], Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
        GaussianBlur(IySq[i], SySq[i], Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
        GaussianBlur(Ixy[i], Sxy[i], Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    }
    /*GaussianBlur(IxSq1, SxSq1, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq2, SxSq2, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq3, SxSq3, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq4, SxSq4, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq5, SxSq5, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq6, SxSq6, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq7, SxSq7, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq8, SxSq8, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq9, SxSq9, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(IxSq10, SxSq10, Size(7,7), 2.0, 0.0, BORDER_DEFAULT);

    GaussianBlur(IySq1, SySq1, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq2, SySq2, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq3, SySq3, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq4, SySq4, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq5, SySq5, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq6, SySq6, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq7, SySq7, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq8, SySq8, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq9, SySq9, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(IySq10, SySq10, Size(7,7), 0.0, 2.0, BORDER_DEFAULT);

    GaussianBlur(Ixy1, Sxy1, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy2, Sxy2, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy3, Sxy3, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy4, Sxy4, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy5, Sxy5, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy6, Sxy6, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy7, Sxy7, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy8, Sxy8, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy9, Sxy9, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(Ixy10, Sxy10, Size(7,7), 2.0, 2.0, BORDER_DEFAULT);*/
}

void computeR(double k){
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

    /*Mat tmp1, tmp2;
    multiply(SxSq1, SySq1, tmp1);
    multiply(Sxy1, Sxy1, tmp2);

    Mat traceM = SxSq1 + SySq1;
    pow(traceM, 2.0, traceM);

    R = (tmp1 - tmp2) - k * traceM;*/
}



