#include <iostream>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <PAWSS/mUtils.h>
#include <PAWSS/Tracker.h>
#include <PAWSS/macros.h>
#include <PAWSS/Rect.h>
#include <PAWSS/Config.h>

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
bool return_flag = false;
std::vector<FloatRect> rois;

// choose rectangle roi on the frame image using right click and drag
void chooseRoiCallBack(int event, int x, int y, int flags, void* params)
{
    PAWSS_UNUSED(flags);
    PAWSS_UNUSED(params);
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

void drawOnImage(cv::Mat& img)
{
    rectangle(img, roi, COLOR_RED);
    cv::putText(img, "["+std::to_string(int(roi.XMin()))+","+std::to_string(int(roi.YMin()))+","
                         +std::to_string(int(roi.Width()))+","+std::to_string(int(roi.Height()))+"]",
                         cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 0.8, COLOR_RED);

    for (size_t i=0; i < rois.size(); ++i)
    {
        rectangle(img, rois[i], COLOR_GREEN);
    }
}

bool trackSequence(Config& conf, const size_t seqIdx)
{
    bool deinterlace_flag = false;
    bool paused = false;

    int startFrame, endFrame;
    float scaleW, scaleH;
    std::string framesFilePath, imgFormat, imgResultFormat;
    char imgPath[256];

    // parse frames file
    assert(seqIdx < conf.mSeqNames.size());
    conf.mSeqName = conf.mSeqNames[seqIdx];
    framesFilePath = conf.mSeqBasePath + "/" + conf.mSeqName + "/" + conf.mSeqName+"_frames.txt";
    // read startFrame and endFrame from frames file
    if(!readFramesFile(framesFilePath, startFrame, endFrame))
        return false;
    imgFormat = conf.mSeqBasePath+"/"+conf.mSeqName+"/img/"+"%04d.jpg";

    if(conf.mSaveFrame)
    {
        std::string imgResultPath = conf.mSeqBasePath+"/"+conf.mSeqName+"/result/";
        mkdir(imgResultPath);
        std::cout << imgResultPath << std::endl;
        imgResultFormat = imgResultPath+"r_%04d.png";
    }

    cv::Mat tmp, img;
    sprintf(imgPath, imgFormat.c_str(), startFrame);
    tmp = cv::imread(imgPath, cv::IMREAD_COLOR);
    if(tmp.empty())
    {
        std::cout << "error: could not read frame: " << imgPath << std::endl;
        return false;
    }

    while(true) {
        cv::setMouseCallback("choose roi", chooseRoiCallBack, NULL);
        img = tmp.clone();
        drawOnImage(img);
        cv::imshow("choose roi", img);
        int k = cv::waitKey(1);
        if(k == 'i' || k == 'I')
        {
            for (size_t i=0; i< rois.size(); ++i)
            {
                std::cout << "ROI " << i << ": "<< rois[i] <<std::endl;
            }
        }
        else if(k == 'r' || k == 'R')
        {
            mouse_drawing = false;
            roi = FloatRect(0, 0, 0, 0);
        }
        else if(k == 'a' || k == 'A')
        {
            rois.push_back(roi);
            std::cout << "Added ROI: " << roi << std::endl;
        }
        else if(k == 'd' || k == 'D')
        {
            if (!rois.empty())
            {
                std::cout << "Deleted the latest ROI: " << rois[rois.size()-1] << std::endl;
                rois.pop_back();
            }
            else
            {
                std::cout << "ROIs is empty." << std::endl;
            }
        }
        else if(k == 27 || k == 'q' || k == 'Q')
        {
            cv::destroyWindow("choose roi");
            break;
        }
    }

    size_t target_num = rois.size();
    std::cout << "Target number: " << target_num << std::endl;
    std::vector<FloatRect>** rBBs;
    rBBs = new std::vector<FloatRect>*[target_num];
    for (size_t i = 0; i < target_num; ++i)
        rBBs[i] = new std::vector<FloatRect>;

    std::cout << "wa";
    // scale the bounding box
    scaleH = 0.3f;
    scaleW = scaleH;

    // declare the tracker
    std::cout << "wa2";
    std::vector<Tracker*> trackers;
    // declare the trackled location
    srand(conf.mSeed);
    if(!conf.mQuietMode)
    {
        cv::namedWindow("result");
    }

    int frameInd = startFrame;
    while(frameInd <= endFrame && !return_flag)
    {
        cv::Mat frame, frameOrig, result;
        sprintf(imgPath, imgFormat.c_str(), frameInd);
        frameOrig = cv::imread(imgPath, cv::IMREAD_COLOR);
        if(frameOrig.empty())
        {
            std::cout << "error: could not read frame: " << imgPath << std::endl;
            continue;
        }

        // deinterlace the frame or not
        if(!deinterlace_flag){
            deinterlace(frameOrig); }

        // scale the frame
        cv::resize(frameOrig, frame, cv::Size(std::round(frameOrig.cols*scaleW), std::round(frameOrig.rows*scaleH)));

        if(frameInd == startFrame)
        {
            for (size_t i=0; i<target_num; ++i)
            {
                FloatRect initBB = rois[i];
                initBB = FloatRect(initBB.XMin()*scaleW, initBB.YMin()*scaleH, initBB.Width()*scaleW, initBB.Height()*scaleH);
                conf.mSearchRadius = std::round((initBB.Width()+initBB.Height())/2);
                Tracker *tracker = new Tracker(conf);
                // new Tracker initialization
                tracker->Initialise(frame, initBB);
                trackers.push_back(tracker);

                FloatRect r(initBB.XMin()/scaleW, initBB.YMin()/scaleH, initBB.Width()/scaleW, initBB.Height()/scaleH);
                // store the bounding box
                rBBs[i]->push_back(r);

                //  save the result image or not
                if(conf.mSaveFrame)
                    rectangle(frameOrig, r, COLOR_CYAN, 3);
            }
            if(conf.mSaveFrame)
            {
                sprintf(imgPath, imgResultFormat.c_str(), frameInd);
                cv::imwrite(imgPath, frameOrig);
            }
        }
        else
        {
            for (size_t i = 0; i < target_num; ++i)
            {
                if(trackers[i]->isInitialised())
                {
                    trackers[i]->Track(frame);

                    // tracker debug
                    if(!conf.mQuietMode && conf.mDebugMode)
                    {
                        trackers[i]->Debug(frame);
                    }

                    // store the bounding box
                    const FloatRect& bb = trackers[i]->getBB();
                    FloatRect r(bb.XMin()/scaleW, bb.YMin()/scaleH, bb.Width()/scaleW, bb.Height()/scaleH);
                    rBBs[i]->push_back(r);
                    //  save the result image or not
                    if(conf.mSaveFrame)
                        rectangle(frameOrig, r, COLOR_CYAN, 3);
                }

                if(conf.mSaveFrame)
                {
                    sprintf(imgPath, imgResultFormat.c_str(), frameInd);
                    cv::imwrite(imgPath, frameOrig);
                }

                // show the tracking result or not
                if(!conf.mQuietMode)
                {
                    frame.copyTo(result);
                    for (size_t i=0; i < target_num; ++i)
                        rectangle(result, trackers[i]->getBB(), COLOR_CYAN, 3);
                    cv::putText(result, std::to_string(frameInd)+'/'+std::to_string(endFrame), cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.4, CV_RGB(0, 255, 0));
                    cv::imshow("result", result);

                    int key = cv::waitKey(paused ? 0 : 1);
                    if (key != -1)
                    {
                        if (key == 27 || key == 'q') // esc q
                        {
                            return_flag = true;
                            break;
                        }
                        else if (key == 112) // p
                        {
                            paused = !paused;
                        }
                    }
                }
            }
        }
        ++frameInd;
    }
    cv::destroyAllWindows();

    // save the result bounding box or not
    if(conf.mSaveBB && !conf.mUseCamera)
    {
        std::vector<std::string> bbFilePaths;
        for (size_t i = 0; i < target_num; ++i)
        {
            std::string bbFilePath = conf.mSeqBasePath+"/"+conf.mSeqName+"/"+conf.mBbFileNamePrefix+"_"+std::to_string(i)+"_bb.txt";
            bbFilePaths.push_back(bbFilePath);
        }
        writeMOTResultBBFile(bbFilePaths, rBBs);
    }

    // clean up
    for(int i=target_num-1; i>=0; --i)
    {
        trackers[i]->Reset();
        delete trackers[i];
        trackers.erase(trackers.begin() + i);
        target_num--;

        delete rBBs[i];
    }
    trackers.clear();
    delete[] rBBs;

    return true;
}

int main(int argc, char* argv[])
{
    std::string configPath;
    if(argc > 1)
    {
        configPath = argv[1];
    }
    else
    {
        configPath = "../../../config_mot.txt";
    }

    // read config file
    Config conf(configPath);
    conf.mUseCamera = 0;

    // check config file
    if(!conf.check())
        return EXIT_FAILURE;

//    std::cout << conf << std::endl;

    int seqNum = conf.mSeqNames.size();
    for(int i=0; i<seqNum; ++i)
    {
        trackSequence(conf, i);
    }


    return EXIT_SUCCESS;
}

