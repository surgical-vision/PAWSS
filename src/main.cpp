#include <iostream>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "Config.h"
#include "Rect.h"
#include "mUtils.h"
#include "Tracker.h"

static const cv::Scalar COLOR_CYAN = CV_RGB(0, 255, 255);
static const cv::Scalar COLOR_RED = CV_RGB(255, 0, 0);
static const cv::Scalar COLOR_GREEN = CV_RGB(0, 255, 0);
static const cv::Scalar COLOR_BLUE = CV_RGB(0, 0, 255);
static const cv::Scalar COLOR_YELLOW = CV_RGB(255, 255, 0);
static const cv::Scalar COLOR_MAGENTA = CV_RGB(255, 0, 255);
static const cv::Scalar COLOR_WHITE = CV_RGB(255, 255, 255);
static const cv::Scalar COLOR_BLACK = CV_RGB(0, 0, 0);

bool mouse_drawing = false;
FloatRect roi;

// choose rectangle roi on the frame image using right click and drag
void chooseRoiCallBack(int event, int x, int y, int flags, void* params)
{
    if(event == cv::EVENT_RBUTTONDOWN)
    {
        mouse_drawing = true;
        roi = FloatRect(x, y, 0, 0);
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if(mouse_drawing)
            roi.Set(roi.XMin(), roi.YMin(), x - roi.XMin(), y - roi.YMin());
    }
    else if (event == cv::EVENT_RBUTTONUP)
    {
        mouse_drawing = false;
        roi.Set(roi.XMin(), roi.YMin(), x - roi.XMin(), y - roi.YMin());
    }
}


int main()
{
    // parameters
    bool deinterlace_flag = false;
    bool paused = false;
    bool saveFrame_flag = false;
    bool saveBB_flag = true;
    bool precision_flag = true;

    // read config file
    std::string configPath = "config.txt";
    Config conf(configPath);
    std::cout << conf << std::endl;

    // check config file
    if(!conf.check())
        return EXIT_FAILURE;

    int startFrame, endFrame;
    float scaleW, scaleH;
    bool gt_flag;
    std::string framesFilePath, gtFilePath, imgFormat, imgResultFormat;
    std::vector<FloatRect> gtBBs, rBBs;
    char imgPath[256];
    // loop through all the sequences
    for(int i=0; i<conf.mSeqNames.size(); ++i)
    {
        // parse frames file
        conf.mSeqName = conf.mSeqNames[i];
        framesFilePath = conf.mSeqBasePath + "/" + conf.mSeqName + "/" + conf.mSeqName+"_frames.txt";
        // read startFrame and endFrame from frames file
        if(!readFramesFile(framesFilePath, startFrame, endFrame))
            continue;
        gtFilePath = conf.mSeqBasePath+"/"+conf.mSeqName+"/"+conf.mSeqName+"_gt.txt";
        // read ground truth bounding box from gt file
        gt_flag = readGtFile(gtFilePath, gtBBs);

        imgFormat = conf.mSeqBasePath+"/"+conf.mSeqName+"/img/"+"%04d.jpg";
        if(saveFrame_flag)
        {
            std::string imgResultPath = conf.mSeqBasePath+"/"+conf.mSeqName+"/result/";
            mkdir(imgResultPath);
            imgResultFormat = imgResultPath+"r_%04d.jpg";
        }

        if(!gt_flag)
        {
            // manually set the initial bounding box
            sprintf(imgPath, imgFormat.c_str(), startFrame);
            cv::namedWindow("choose_roi");
            cv::setMouseCallback("choose_roi", chooseRoiCallBack);
            cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

            while(true)
            {
                cv::Mat img_copy= img.clone();
                rectangle(img_copy, roi, COLOR_GREEN);
                cv::putText(img_copy, "Idx: "+std::to_string(startFrame), cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 0.8, COLOR_YELLOW);
                cv::putText(img_copy, "["+std::to_string(int(roi.XMin()))+","+std::to_string(int(roi.YMin()))+","
                                         +std::to_string(int(roi.Width()))+","+std::to_string(int(roi.Height()))+"]",
                                         cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 0.8, COLOR_YELLOW);
                cv::imshow("choose roi", img_copy);
                int k = cv::waitKey(1);
                if(k == 'q')
                    break;
                if(k == 'f')
                {
                    if(startFrame != endFrame)
                    {
                        startFrame++;
                        sprintf(imgPath, imgFormat.c_str(), startFrame);
                        img = cv::imread(imgPath, cv::IMREAD_COLOR);
                    }
                }
                if(k == 'b')
                {
                    if(startFrame != 0)
                    {
                        startFrame--;
                        sprintf(imgPath, imgFormat.c_str(), startFrame);
                        img = cv::imread(imgPath, cv::IMREAD_COLOR);
                    }

                }
                if(k == 'i')
                    std::cout<<"ROI: tl "<<roi.XMin()<<" br "<<roi.YMin()<<std::endl;
                if(k == 'r')
                {
                    mouse_drawing = false;
                    roi = FloatRect(0, 0, 0, 0);
                }
                if (k==27)
                {
                    cv::destroyAllWindows();
                    return -1;
                }
            }
            cv::destroyWindow("choose_roi");
        }

        FloatRect initBB;
        if(gt_flag) {
            initBB = gtBBs[0]; }
        else {
            initBB = roi; }
        // scale the bounding box
        scaleFrame(initBB.Width(), initBB.Height(), scaleW, scaleH);
        initBB = FloatRect(initBB.XMin()*scaleW, initBB.YMin()*scaleH, initBB.Width()*scaleW, initBB.Height()*scaleH);
        conf.mSearchRadius = std::round((initBB.Width()+initBB.Height())/2);

        // declare the tracker
        Tracker tracker(conf);
        srand(conf.mSeed);
        if(!conf.mQuietMode)
        {
            cv::namedWindow("result");
        }

        rBBs.clear();
        for(int frameInd = startFrame; frameInd<=endFrame; ++frameInd)
        {
            cv::Mat frame, frameOrig, result;
            sprintf(imgPath, imgFormat.c_str(), frameInd);
            frameOrig = cv::imread(imgPath, cv::IMREAD_COLOR);
            if(frameOrig.empty())
            {
                std::cout << "error: could not read frame: " << imgPath << std::endl;
                if(gt_flag)
                    return EXIT_FAILURE;
                else
                    continue;
            }

            // deinterlace the frame or not
            if(!deinterlace_flag){
                deinterlace(frame); }
            // scale the frame
            cv::resize(frameOrig, frame, cv::Size(std::round(frameOrig.cols*scaleW), std::round(frameOrig.rows*scaleH)));

            if(frameInd == startFrame)
            {
                tracker.Initialise(frame, initBB);
                // store the bounding box
                FloatRect r(initBB.XMin()/scaleW, initBB.YMin()/scaleH, initBB.Width()/scaleW, initBB.Height()/scaleH);
                rBBs.push_back(r);
                //  save the result image or not
                if(saveFrame_flag)
                {
                    sprintf(imgPath, imgResultFormat.c_str(), frameInd);
                    rectangle(frameOrig, r, COLOR_CYAN, 3);
                    cv::imwrite(imgPath, frameOrig);
                }
            }
            if(tracker.isInitialised())
            {
                tracker.Track(frame);

                // store the bounding box
                if(frameInd != startFrame)
                {
                    const FloatRect& bb = tracker.getBB();
                    FloatRect r(bb.XMin()/scaleW, bb.YMin()/scaleH, bb.Width()/scaleW, bb.Height()/scaleH);
                    rBBs.push_back(r);
                    //  save the result image or not
                    if(saveFrame_flag)
                    {
                        sprintf(imgPath, imgResultFormat.c_str(), frameInd);
                        rectangle(frameOrig, r, COLOR_CYAN, 3);
                        cv::imwrite(imgPath, frameOrig);
                    }
                }

                if(!conf.mQuietMode && conf.mDebugMode){
                    tracker.Debug(frame, frameInd);}
            }

            // show the tracking result or not
            if(!conf.mQuietMode)
            {
                frame.copyTo(result);
                rectangle(result, tracker.getBB(), COLOR_CYAN, 3);

                cv::putText(result, std::to_string(frameInd)+'/'+std::to_string(endFrame), cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.4, CV_RGB(0, 255, 0));
                cv::putText(result, std::to_string(tracker.getScale()), cv::Point(10, 40), cv::FONT_HERSHEY_COMPLEX, 0.4, CV_RGB(0, 255, 0));

                cv::imshow("result", result);
                int key = cv::waitKey(paused ? 0 : 1);
                if (key != -1)
                {
                    if (key == 27 || key == 113) // esc q
                    {
                        break;
                    }
                    else if (key == 112) // p
                    {
                        paused = !paused;
                    }
                }
            }
        }

        // save the result bounding box or not
        if(saveBB_flag)
        {
            std::string bbFilePath = conf.mSeqBasePath+"/"+conf.mSeqName+"/"+conf.mBbFileName;
            writeResultBBFile(bbFilePath, rBBs);
        }
        // If there is ground truth and you want to evaluate the precision of the tracking result
        if(precision_flag && gt_flag)
        {
            // evaluate presicion rate
            std::vector<float> prec = estPrecision(rBBs, gtBBs);
            std::string precFilePath = conf.mSeqBasePath+"/"+conf.mSeqName+"/"+conf.mPrecFileName;
            writePrecisionFile(precFilePath, prec);
        }

        tracker.Reset();
    }
    return EXIT_SUCCESS;
}
