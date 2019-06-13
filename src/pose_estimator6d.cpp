/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include "pose_estimator6d.h"

using namespace std;
using namespace cv;

bool sortTemplateView(std::pair<float, TemplateView*> a, std::pair<float, TemplateView*> b)
{
    return a.first < b.first;
}


PoseEstimator6D::PoseEstimator6D(int width, int height, float zNear, float zFar, const cv::Matx33f &K, const cv::Matx14f &distCoeffs, vector<Object3D*> &objects)
{
    renderingEngine = RenderingEngine::Instance();
    optimizationEngine = new OptimizationEngine(width, height);
    
    SDT2D = new SignedDistanceTransform2D(8.0f);
    
    this->width = width;
    this->height = height;
    
    this->K = K;
    this->distCoeffs = distCoeffs;
    
    initUndistortRectifyMap(K, distCoeffs, cv::noArray(), K, Size(width, height), CV_16SC2, map1, map2);
    
    initialized = false;
    
    //start initialization
    renderingEngine->init(K, width, height, zNear, zFar, 4);
    
    renderingEngine->makeCurrent();
    
    for(int i = 0; i < objects.size(); i++)
    {
        objects[i]->setModelID(i+1);
        this->objects.push_back(objects[i]);
        this->objects[i]->initBuffers();
        this->objects[i]->generateTemplates();
        this->objects[i]->reset();
    }
    
    renderingEngine->doneCurrent();
    
    tmp = 0;
}


PoseEstimator6D::~PoseEstimator6D()
{
    renderingEngine->destroy();
    
    delete optimizationEngine;
    
    delete SDT2D;
}


void PoseEstimator6D::toggleTracking(cv::Mat &frame, int objectIndex, bool undistortFrame)
{
    if(objectIndex >= objects.size())
        return;
    
    if(undistortFrame)
        remap(frame, frame, map1, map2, INTER_LINEAR);
    
    if(!objects[objectIndex]->isInitialized())
    {
        objects[objectIndex]->initialize();
        
        renderingEngine->setLevel(0);
        
        renderingEngine->renderSilhouette(vector<Model*>(objects.begin(), objects.end()), GL_FILL);
        
        Mat mask = renderingEngine->downloadFrame(RenderingEngine::MASK);
        Mat depth = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
        
        float zNear = renderingEngine->getZNear();
        float zFar = renderingEngine->getZFar();
        
        objects[objectIndex]->getTCLCHistograms()->update(frame, mask, depth, K, zNear, zFar);
        
        initialized = true;
    }
    else
    {
        objects[objectIndex]->reset();
        
        initialized = false;
        for(int o = 0; o < objects.size(); o++)
        {
            initialized |= objects[o]->isInitialized();
        }
    }
}


void PoseEstimator6D::estimatePoses(cv::Mat &frame, bool undistortFrame, bool checkForLoss)
{
    if(undistortFrame)
        remap(frame, frame, map1, map2, INTER_LINEAR);

    vector<Mat> imagePyramid;
    
    Mat frameCpy = frame.clone();
    imagePyramid.push_back(frameCpy);
    
    for(int l = 1; l < 4; l++)
    {
        resize(frame, frameCpy, Size(frame.cols/pow(2, l), frame.rows/pow(2, l)));
        imagePyramid.push_back(frameCpy);
    }
    
    if(initialized)
    {
        optimizationEngine->minimize(imagePyramid, objects);
        
        renderingEngine->setLevel(0);
        
        renderingEngine->renderSilhouette(vector<Model*>(objects.begin(), objects.end()), GL_FILL);
        
        Mat mask = renderingEngine->downloadFrame(RenderingEngine::MASK);
        Mat depth = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
        
        float zNear = renderingEngine->getZNear();
        float zFar = renderingEngine->getZFar();
        
        Mat binned;
        parallel_for_(cv::Range(0, 8), Parallel_For_convertToBins(frame, binned, objects[0]->getTCLCHistograms()->getNumBins(), 8));
        
        for(int i = 0; i < objects.size(); i++)
        {
            if(objects[i]->isInitialized())
            {
                if(!objects[i]->isTrackingLost())
                {
                    float e = evaluateEnergyFunction(objects[i], mask, depth, binned, 0, 8);
                    
                    if(checkForLoss && (e > objects[i]->getQualityThreshold() || e == 0.0f))
                    {
                        objects[i]->setTrackingLost(true);
                        objects[i]->setPose(Matx44f());
                    }
                    else
                    {
                        objects[i]->getTCLCHistograms()->update(frame, mask, depth, K, zNear, zFar);
                    }
                }
                else
                {
                    relocalize(objects[i], imagePyramid);
                }
            }
        }
    }
}

void PoseEstimator6D::relocalize(Object3D *object, vector<Mat> &imagePyramid)
{
    vector<TemplateView*> templateViews = object->getTemplateViews();
    
    int numDistances = object->getNumDistances();
    
    int level = 3;
    
    // PREPARE FRAME FOR LOWEST LEVEL
    Mat binned;
    parallel_for_(cv::Range(0, 8), Parallel_For_convertToBins(imagePyramid[level], binned, object->getTCLCHistograms()->getNumBins(), 8));
    
    Mat prMap;
    parallel_for_(cv::Range(0, 8), Parallel_For_createPosteriorResponseMap(object->getTCLCHistograms(), binned, prMap, 8));
    
    parallel_for_(cv::Range(0, (int)templateViews.size()), Parallel_For_exhaustiveSearch(object, templateViews, binned, prMap, level, 4, -1));
    
    parallel_for_(cv::Range(0, (int)templateViews.size()), Parallel_For_exhaustiveSearch(object, templateViews, binned, prMap, level, 1, 2));
    
    
    // KEEP ONLY THE BEST MATCHING DISTANCE PER TEMPLATE
    vector<pair<float, TemplateView*> > errorKVMap0;
    
    for(int i = 0; i < templateViews.size(); i+=numDistances)
    {
        float minE = FLT_MAX;
        int minIdx = -1;
        for(int j = 0; j < numDistances; j++)
        {
            TemplateView *templateView = templateViews[i+j];
            Point3f offset = templateView->getCurrentOffset(level);
            
            if(offset.z < minE)
            {
                minE = offset.z;
                minIdx = j;
            }
        }
        if(minE > 0.0f && minIdx >= 0)
        {
            errorKVMap0.push_back(pair<float, TemplateView*>(minE, templateViews[i + minIdx]));
        }
    }
    
    sort(errorKVMap0.begin(), errorKVMap0.end(), sortTemplateView);
    
    level = 2;
    
    // PREPARE FRAME FOR 2ND LOWEST LEVEL
    parallel_for_(cv::Range(0, 8), Parallel_For_convertToBins(imagePyramid[level], binned, object->getTCLCHistograms()->getNumBins(), 8));
    
    vector<pair<float, TemplateView*> > errorKVMap;
    
    for(int i = 0; i < errorKVMap0.size()/2; i++)
    {
        float kve = errorKVMap0[i].first;
        TemplateView *templateView = errorKVMap0[i].second;
        
        if(kve > 0.0f && kve < 1.0f)
        {
            parallel_for_(cv::Range(0, (int)templateView->getNeighborTemplates().size()), Parallel_For_neighborSearch(object, templateView, binned, level, 1));
            
            for(int n = 0; n < templateView->getNeighborTemplates().size(); n++)
            {
                TemplateView* kvn = templateView->getNeighborTemplates()[n];
                
                float e = kvn->getCurrentOffset(level).z;
                
                if(e > 0.0f && e < 1.0f)
                    errorKVMap.push_back(pair<float, TemplateView*>(e, kvn));
            }
        }
    }
    
    sort(errorKVMap.begin(), errorKVMap.end(), sortTemplateView);
    
    parallel_for_(cv::Range(0, 8), Parallel_For_convertToBins(imagePyramid[0], binned, object->getTCLCHistograms()->getNumBins(), 8));
    
    float minE = FLT_MAX;
    int finalIdx = -1;
    Matx44f finalPose;
    
    bool trackingLost = true;
    
    for(int i = 0; i < std::min(4, (int)errorKVMap.size()); i++)
    {
        TemplateView *templateView = errorKVMap[i].second;
        
        Point3f offset = templateView->getCurrentOffset(level);;
        int offsetX = offset.x;
        int offsetY = offset.y;
        float offsetE = offset.z;
        
        if(offsetE > 0 && offsetE < 1.0f)
        {
            Rect roi = templateView->getROI(level);
            
            Vec3f offsetVec((-roi.x+offsetX)*pow(2, level)+imagePyramid[0].cols/2, (-roi.y+offsetY)*pow(2, level)+imagePyramid[0].rows/2, 1);
            
            Matx44f pose = templateView->getPose();
            
            offsetVec = K.inv()*offsetVec;
            pose(0, 3) = offsetVec[0]*pose(2, 3);
            pose(1, 3) = offsetVec[1]*pose(2, 3);
            
            object->setPose(pose);
            
            renderingEngine->setLevel(0);
            renderingEngine->renderSilhouette(vector<Model*>(objects.begin(), objects.end()), GL_FILL);
            
            Mat mask = renderingEngine->downloadFrame(RenderingEngine::MASK);
            Mat depth = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
            
            float zNear = renderingEngine->getZNear();
            float zFar = renderingEngine->getZFar();
            
            object->getTCLCHistograms()->updateCentersAndIds(mask, depth, K, zNear, zFar, 0);
            
            vector<Object3D*> tmp;
            tmp.push_back(object);
            
            optimizationEngine->minimize(imagePyramid, tmp, 2);
            
            float e = evaluateEnergyFunction(object, binned, 0, 8);
            
            if(e > 0.0f && e < minE)
            {
                minE = e;
                finalPose = object->getPose();
                
                if(e < object->getQualityThreshold())
                {
                    trackingLost = false;
                    finalIdx = i;
                }
            }
        }
    }
    if(!trackingLost)
    {
        object->setPose(finalPose);
        object->setTrackingLost(false);
        
        renderingEngine->setLevel(0);
        renderingEngine->renderSilhouette(vector<Model*>(objects.begin(), objects.end()), GL_FILL);
        
        Mat mask = renderingEngine->downloadFrame(RenderingEngine::MASK);
        Mat depth = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
        
        float zNear = renderingEngine->getZNear();
        float zFar = renderingEngine->getZFar();
        
        object->getTCLCHistograms()->updateCentersAndIds(mask, depth, K, zNear, zFar, 0);
        
    }
    else
    {
        object->setPose(Matx44f());
    }
}


cv::Rect PoseEstimator6D::computeBoundingBox(const std::vector<cv::Point3i> &centersIDs, int offset, int level, const cv::Size& maxSize)
{
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = -1, maxY = -1;
    
    for(int i = 0; i < centersIDs.size(); i++)
    {
        Point3i p = centersIDs[i];
        int x = p.x/pow(2, level);
        int y = p.y/pow(2, level);
        
        if(x < minX) minX = x;
        if(y < minY) minY = y;
        if(x > maxX) maxX = x;
        if(y > maxY) maxY = y;
    }
    
    minX -= offset;
    minY -= offset;
    maxX += offset;
    maxY += offset;
    
    if(minX < 0) minX = 0;
    if(minY < 0) minY = 0;
    if(maxX > maxSize.width) maxX = maxSize.width;
    if(maxY > maxSize.height) maxY = maxSize.height;
    
    return Rect(minX, minY, maxX - minX, maxY - minY);
}


float PoseEstimator6D::evaluateEnergyFunction(Object3D *object, const Mat &binned, int level, int threads)
{
    renderingEngine->setLevel(0);
    renderingEngine->renderSilhouette(vector<Model*>(objects.begin(), objects.end()), GL_FILL);
    
    Mat mask = renderingEngine->downloadFrame(RenderingEngine::MASK);
    Mat depth = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
    
    return evaluateEnergyFunction(object, mask, depth, binned, level, 8);
}


float PoseEstimator6D::evaluateEnergyFunction(Object3D *object, const Mat &mask, const Mat &depth, const Mat &binned, int level, int threads)
{
    float zNear = renderingEngine->getZNear();
    float zFar = renderingEngine->getZFar();
    
    TCLCHistograms *tclcHistograms = object->getTCLCHistograms();
    tclcHistograms->updateCentersAndIds(mask, depth, K, zNear, zFar, 0);
    
    vector<Point3i> centersIDs = tclcHistograms->getCentersAndIDs();
    
    if(centersIDs.size() > 0)
    {
        Rect roi = computeBoundingBox(centersIDs, tclcHistograms->getRadius(), 0, binned.size());
        
        Mat croppedMask = mask(roi).clone();
        Mat croppedDepth = depth(roi).clone();
        
        Mat sdt, xyPos;
        SDT2D->computeTransform(croppedMask, sdt, xyPos, 8, object->getModelID());
        
        Mat heaviside;
        parallel_for_(cv::Range(0, 8), Parallel_For_convertToHeaviside(sdt, heaviside, 8));
        
        return evaluateEnergyFunction(tclcHistograms, centersIDs, binned, heaviside, roi, roi.x, roi.y, level, 8);
    }
    else
        return 0.0f;
    
}

float PoseEstimator6D::evaluateEnergyFunction(TCLCHistograms *tclcHistograms, const vector<Point3i> &centersIDs, const Mat &binned, const Mat &heaviside, const Rect &roi, int offsetX, int offsetY, int level, int threads)
{
    float e = 0.0f;
    int N = roi.height;
    
    Mat eCollection = Mat::zeros(1, N, CV_32FC3);
    
    parallel_for_(cv::Range(0, N), Parallel_For_evaluateEnergy(tclcHistograms, centersIDs, binned, heaviside, roi, offsetX, offsetY, level, eCollection, N));
    
    int sum1 = 0;
    int sum2 = 0;
    
    for(int i = 0; i < eCollection.cols; i++)
    {
        Vec3f eCnt = eCollection.at<Vec3f>(0, i);
        e += eCnt[0];
        
        sum1 += (int)eCnt[1];
        sum2 += (int)eCnt[2];
    }
    
    if(sum1 && (float)sum1/sum2 > 0.5f)
    {
        e /= sum1;
    }
    else
    {
        e = FLT_MAX;
    }
    
    return e;
}


void PoseEstimator6D::reset()
{
    for(int i = 0; i < objects.size(); i++)
    {
        objects[i]->reset();
    }
    
    initialized = false;
}
