#include <iostream>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "Config.h"
#include "Rect.h"
#include "mUtils.h"
#include "Tracker.h"

#define VOT_RECTANGLE
#include "votUtils/vot.h"


int main(int argc, char* argv[])
{

    std::string configPath;
    if(argc > 1)
    {
        configPath = argv[1];
    }
    else
    {
        configPath = "/Users/xiaofeidu/workspace/PAWSS/config_vot2014.txt";
    }

    // read config file
    Config conf(configPath);

//    // check config file
//    if(!conf.check())
//        return EXIT_FAILURE;



    VOT vot;
    cv::Rect cvInitBB;
    cvInitBB << vot.region();
    FloatRect initBB(cvInitBB.x, cvInitBB.y, cvInitBB.width, cvInitBB.height);

    // scale the bounding box
    float scaleW, scaleH;
    scaleFrame(initBB.Width(), initBB.Height(), scaleW, scaleH);

    initBB = FloatRect(initBB.XMin()*scaleW, initBB.YMin()*scaleH, initBB.Width()*scaleW, initBB.Height()*scaleH);
    conf.mSearchRadius = std::round((initBB.Width()+initBB.Height())/2);

    // declare the tracker
    Tracker tracker(conf);
    // To properly evaluate such trackers the random seed should not be fixed to a certain value.
    // The best way to ensure this is to initialize seed with a different value every time, for example using current time.
//    srand(conf.mSeed);
    srand(time(NULL));
    if(!conf.mQuietMode)
    {
        cv::namedWindow("result");
    }

    std::string imgPath = vot.frame();
    cv::Mat frameOrig = cv::imread(imgPath, cv::IMREAD_COLOR);
    // scale the frame
    cv::Mat frame, result;
    cv::resize(frameOrig, frame, cv::Size(std::round(frameOrig.cols*scaleW), std::round(frameOrig.rows*scaleH)));

    tracker.Initialise(frame, initBB);

    std::cout << "before while: " << imgPath << std::endl;
    while(!vot.end())
    {
        imgPath = vot.frame();
        std::cout << "during while: " << imgPath << std::endl;
        if(imgPath.empty()) break;
        cv::Mat frameOrig = cv::imread(imgPath, cv::IMREAD_COLOR);

        // scale the frame
        cv::Mat frame;
        cv::resize(frameOrig, frame, cv::Size(std::round(frameOrig.cols*scaleW), std::round(frameOrig.rows*scaleH)));

        // track the frame
        if(tracker.isInitialised())
        {
            tracker.Track(frame);
            const FloatRect& bb = tracker.getBB();
            cv::Rect cvbb(bb.XMin()/scaleW, bb.YMin()/scaleH, bb.Width()/scaleW, bb.Height()/scaleH);
            vot.report(cvbb);

            if(!conf.mQuietMode && conf.mDebugMode){
                tracker.Debug(frame);}
        }

        // show the tracking result or not
        if(!conf.mQuietMode)
        {
            frame.copyTo(result);
            rectangle(result, tracker.getBB(), cv::Scalar(255,0,0), 3);
            cv::putText(result, std::to_string(tracker.getScale()), cv::Point(10, 40), cv::FONT_HERSHEY_COMPLEX, 0.4, CV_RGB(0, 255, 0));
            cv::imshow("result", result);
            int key = cv::waitKey(1);
            if (key != -1)
            {
                if (key == 27 || key == 113) // esc q
                {
                    break;
                }
            }
        }
    }
    tracker.Reset();
    return EXIT_SUCCESS;
}

