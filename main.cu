
#include <iostream>
#include <assert.h>
#include <exception>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>


using namespace std;
using namespace cv;


// Function for calculating mean 
double  meanMeasure(Mat img) { 
    int nor = img.rows;
    int noc = img.cols* img.channels();
    double sum = 0, mean; 
    //data access
    #pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* data = img.ptr<double>(j);
        #pragma omp parallel for
        for(int i = 0; i<noc; i++){
            sum += static_cast<double>(data[i]);
        }        
    } 
     mean =  sum / (nor* noc);
     return mean; 
} 

// Function for calculating variance
double  varianceMeasure(Mat img, double mu) { 
    int nor = img.rows;
    int noc = img.cols* img.channels();
    double sum = 0, stdev; 
    //data access
    #pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* data = img.ptr<double>(j);
        #pragma omp parallel for
        for(int i = 0; i<noc; i++){
            data[i] -= mu;
            data[i] *= static_cast<double>(data[i]);
        }        
    } 
    #pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* data = img.ptr<double>(j);
        #pragma omp parallel for
        for(int i = 0; i<noc; i++){
            sum += static_cast<double>(data[i]);
        }        
    } 
    stdev = sum / (nor * noc);      
    return stdev; 
} 

// Function for calculating deviation
double hesitanceProbablitydeviationMeasure(Mat imgx, Mat  imgy){
    int nor = imgx.rows;
    int noc = imgx.cols* imgx.channels();
    double sumx = 0.0, sumy = 0.0; 
    double gma = meanMeasure(imgy);
    //data access I
#pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* datax = imgx.ptr<double>(j);
        double* datay = imgy.ptr<double>(j);
#pragma omp parallel for
        for(int i = 0; i<noc; i++){
            sumx += datax[i]*(datay[i]-gma)*(datay[i]-gma);
        }        
    } 
    //data access II
#pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* datay = imgy.ptr<double>(j);
#pragma omp parallel for
        for(int i = 0; i<noc; i++){
            sumy += datay[i];
        }        
    }         
   double deviaval =((6+sumx)/(7+sumy));    
    return deviaval;
}

// Function for calculating  Score
double hesitanceProbablitydeviationScoreMeasure(Mat imgx, Mat  imgy){
    int nor = imgx.rows;
    int noc = imgx.cols* imgx.channels();
    double sumx = 0.0, sumy = 0.0; 
    
    //data access I
#pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* datax = imgx.ptr<double>(j);
#pragma omp parallel for
        for(int i = 0; i<noc; i++){
            sumx += datax[i];
            std::cout<<sumx<<std::endl;
        }        
    } 
    //data access II
#pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* datay = imgy.ptr<double>(j);
#pragma omp parallel for
        for(int i = 0; i<noc; i++){
            sumy += datay[i];
        }        
    }         
   double score =((1+sumx)/(2+sumy));    
    return score;
}

// Function for  hesitance probablity measure
Mat hesitanceProbablityMeasure (Mat imgx, Mat imgy) {
    Mat X;
    int nor = imgx.rows;
    int noc = imgx.cols* imgx.channels();     
    //data access I
#pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* datax = imgx.ptr<double>(j);
        double* datay = imgy.ptr<double>(j);
#pragma omp parallel for
        for(int i = 0; i<noc; i++){
            datax[i] = (datax[i]*datay[i]);
        }        
    } 
    return X;
}

// Function for calculating hesitance probablity
Mat hesitanceProbablity(Mat img, double mnval, double stdev){
    Mat X;
    static const double inv_sqrt_2pi = 0.3989422804014327;
    int nor = img.rows;
    int noc = img.cols* img.channels();
    //data access
    #pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* data = img.ptr<double>(j);
        #pragma omp parallel for
        for(int i = 0; i<noc; i++){
             data[i]  = (inv_sqrt_2pi/stdev)*(expf(-0.5 * (data[i] - mnval) /(stdev*stdev)));             
        }         
    }     
    return X;
}

// Function for calculating S-hape  membership grade
Mat sMembershipGrade(Mat img){
    Mat X;
    int nor = img.rows;
    int noc = img.cols* img.channels();
    #pragma omp parallel for
    for(int j = 0; j<nor; j++){
        double* data = img.ptr<double>(j);
        #pragma omp parallel for
        for(int i = 0; i<noc; i++){
            data[i] = static_cast<double>(data[i]);
        }        
    }    
    return X;
}

//Main routines
int main(int argc, char** argv) {    
    try{    
    cv::Mat srcimg, img, img_edge, labels, centroids, img_color, stats;    
    // load image or show help if no image was provided
    srcimg = cv::imread( "src.png", cv::IMREAD_COLOR );
    
   if(srcimg.empty()) {
    std::cout << "\nExample 14-3: Drawing labeled connected componnents\n"<< endl; 
    return EXIT_FAILURE;
    }  
    cv::cvtColor(srcimg, img, COLOR_BGR2GRAY);
    
    cv::namedWindow("Image before threshold",WINDOW_OPENGL);
    cv::imshow("Image before threshold", srcimg);
    //function invocked
    Mat dimg, fuzzImg;
    img.convertTo(dimg,CV_64F, 1/255.0);
    //fuzzifes
    fuzzImg = sMembershipGrade(dimg);
    
    //PDF
    Mat hesPro;
    double meanval=  meanMeasure(dimg); 
    double varianceval= varianceMeasure(dimg, meanval);
    std::cout << "\nMean\t\n"<<meanval<< std::endl; 
    std::cout << "\nVariance\t\n"<<varianceval<< std::endl;      
    hesPro = hesitanceProbablity(fuzzImg, meanval, varianceval);

    //hesitance probablity measure
    Mat  hpmat;
    hpmat = hesitanceProbablityMeasure (fuzzImg, hesPro);
    //hesitance probablity score measure
    double Score;
    Score= hesitanceProbablitydeviationScoreMeasure(hpmat, hesPro);
    std::cout << "\nScore Value\t\n"<<Score<< std::endl; 
    
    // hesitance probablity deviation measure
    double deval;
    deval = hesitanceProbablitydeviationMeasure(hpmat, hesPro);
    std::cout << "\nScore Value\t\n"<<deval<< std::endl; 
    
    // threshold measure
    int a = int(Score*50);
    int b = int(deval*100);
    cv::threshold(img, img_edge, a, b, cv::THRESH_BINARY);
    cv::namedWindow("Image after threshold",WINDOW_OPENGL);
    cv::imshow("Image after threshold", img_edge);
    
    //connected Components With Stats
    int i, nccomps = cv::connectedComponentsWithStats (img_edge, labels, stats, centroids );
    
    std::cout << "Total Connected Components Detected: " << nccomps << std::endl;    
    vector<cv::Vec3b> colors(nccomps+1);
    colors[0] = cv::Vec3b(0,0,0);
    
    // background pixels remain black.
#pragma omp parallel for
    for( i = 1; i <= nccomps; i++ ) {
        colors[i] = cv::Vec3b(rand()%256, rand()%256, rand()%256);
        if( stats.at<int>(i-1, cv::CC_STAT_AREA) < 100 )
            // small regions are painted with black too.
            colors[i] = cv::Vec3b(0,0,0); 
    }
    img_color = cv::Mat::zeros(img.size(), CV_8UC3);
    #pragma omp parallel for
    for( int y = 0; y < img_color.rows; y++ )
        #pragma omp parallel for
        for( int x = 0; x < img_color.cols; x++ )
        {
            int label = labels.at<int>(y, x);
            CV_Assert(0 <= label && label <= nccomps);
            img_color.at<cv::Vec3b>(y, x) = colors[label];
        }
    cv::namedWindow("Segementation",WINDOW_OPENGL);
    cv::imshow("Segementation", img_color);
    cv::waitKey();
    cv::destroyAllWindows();

    }catch(std::exception &erb ){        
        std::cerr<<"Error found\n"<<erb.what()<<std::endl;
        return EXIT_FAILURE;        
    }   

    return EXIT_SUCCESS;
}
