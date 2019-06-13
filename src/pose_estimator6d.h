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

#ifndef POSE_ESTIMATOR6D_H
#define POSE_ESTIMATOR6D_H

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>

#include "object3d.h"
#include "rendering_engine.h"
#include "optimization_engine.h"
#include "signed_distance_transform2d.h"
#include "template_view.h"

/**
 *  This class implements a region-based 6DOF pose estimator in form of a
 *  tracking and detection hybrid approach. It can estimate the poses of
 *  multiple rigid 3D objects based on given 3d models and the images of
 *  a single monocular RGB camera in real-time.
 */
class PoseEstimator6D
{
public:
    /**
     *  Constructor of the pose estimator initializing rectification
     *  maps for image undistorting and the rendering engine, given
     *  the 3x3 float intrinsic camera matrix
     *  K = [fx 0 cx]
     *      [0 fy cy]
     *      [0  0  1]
     *  and distortion coefficients (as needed by OpenCV).
     *  It also initializes the OpenGL rendering buffers for all
     *  provided 3D objects using the OpenGL context of the engine.
     *
     *  @param  width The width in pixels of the camera frame at full resolution.
     *  @param  height The height in pixels of the camera frame at full resolution.
     *  @param  zNear The distance of the OpenGL near plane.
     *  @param  zFar The distance of the OpenGL far plane.
     *  @param  K The intrinsic camera matrix.
     *  @param  distCoeffs The cameras lens distortion coefficients.
     *  @param  objects A collection of all 3D objects to be tracked.
     */
    PoseEstimator6D(int width, int height, float zNear, float zFar, const cv::Matx33f &K, const cv::Matx14f &distCoeffs, std::vector<Object3D*> &objects);
    
    ~PoseEstimator6D();
    
    /**
     *  Initializes/starts tracking based on the current camera frame
     *  for a specified 3D object using its initial pose by building
     *  the first set of tclc-histograms. If the object was already
     *  initialized this method will reset/stop tracking for it
     *  instead.
     *
     *  @param  frame The current camera frame  (RGB, uchar).
     *  @param  objectIndex The index of the object to be initialized.
     *  @param  undistortFrame A flag indicating whether the image should first be undistorted for initialization (default = true).
     */
    void toggleTracking(cv::Mat &frame, int objectIndex, bool undistortFrame = true);
    
    /**
     *  This method tries to track and detect the 6DOF poses of all
     *  currently initialized rigid 3D objects by minimizing the
     *  region-based cost function using tclc-histograms. During
     *  successful tracking, the pose is estimated frame-to-frame.
     *  If tracking has been lost for an object, the pose will be
     *  estimated using a template matching approach for pose detection
     *  also based on tclc-histograms.
     *  Within this method, the 3D objectives are updated with the
     *  new estimated poses which can be obtained by calling getPose()
     *  on each object afterwards.
     *
     *  @param frame  The current camera frame (RGB, uchar).
     *  @param undistortFrame A flag indicating whether the image should first be undistorted for initialization (default = true).
     *  @param undistortFrame A flag indicating whether it should be checked for a tracking loss after pose estimation (default = true).
     */
    void estimatePoses(cv::Mat &frame, bool undistortFrame = true, bool checkForLoss = true);
    
    /**
     *  Resets/stops pose tracking for all objects by clearing the
     *  respective sets of tclc-histograms.
     */
    void reset();
    
private:
    int width;
    int height;
    
    cv::Matx33f K;
    cv::Matx14f distCoeffs;
    
    cv::Mat map1;
    cv::Mat map2;
    
    std::vector<Object3D*> objects;
    
    RenderingEngine *renderingEngine;
    OptimizationEngine *optimizationEngine;

    SignedDistanceTransform2D *SDT2D;
    
    cv::Mat lastFrame;
    
    bool initialized;
    
    int tmp;
    
    void relocalize(Object3D *object, std::vector<cv::Mat> &imagePyramid);
    
    cv::Rect computeBoundingBox(const std::vector<cv::Point3i> &centersIDs, int offset, int level, const cv::Size &maxSize);
    
    float evaluateEnergyFunction(Object3D *object, const cv::Mat &binned, int level, int threads);
    
    float evaluateEnergyFunction(Object3D *object, const cv::Mat &mask, const cv::Mat &depth, const cv::Mat &binned, int level, int threads);
    
    float evaluateEnergyFunction(TCLCHistograms *tclcHistograms, const std::vector<cv::Point3i> &centersIDs, const cv::Mat &binned, const cv::Mat &heaviside, const cv::Rect &roi, int offsetX, int offsetY, int level, int threads);
    
    float evaluateEnergyFunction_local(TCLCHistograms *tclcHistograms, const std::vector<cv::Point3i> &centersIDs, const cv::Mat &binned, const cv::Mat &heaviside, const cv::Rect &roi, int offsetX, int offsetY, int level);
    
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, the region-based cost function is
 *  evaluated given the current camera image for a single object.
 */
class Parallel_For_evaluateEnergy: public cv::ParallelLoopBody
{
private:
    int* binsData;
    
    cv::Mat localFG;
    cv::Mat localBG;
    
    float* histogramsFGData;
    float* histogramsBGData;
    
    std::vector<cv::Point3i> _centersIDs;
    
    uchar *initializedData;
    
    int numHistograms;
    int radius;
    int radius2;
    
    int scale;
    
    int fullWidth;
    int fullHeight;
    
    int _offsetX;
    int _offsetY;
    
    float *hsData;
    
    cv::Rect _roi;
    
    float *_eCollection;
    
    int _threads;
    
public:
    Parallel_For_evaluateEnergy(TCLCHistograms *tclcHistograms, const std::vector<cv::Point3i> &centersIDs, const cv::Mat &bins, const cv::Mat& heaviside, const cv::Rect &roi, int offsetX, int offsetY, int level, cv::Mat &eCollection, int threads)
    {
        binsData = (int*)bins.ptr<int>();
        
        localFG = tclcHistograms->getLocalForegroundHistograms();
        localBG = tclcHistograms->getLocalBackgroundHistograms();
        
        histogramsFGData = (float*)localFG.ptr<float>();
        histogramsBGData = (float*)localBG.ptr<float>();
        
        _centersIDs = centersIDs;
        
        initializedData = tclcHistograms->getInitialized().data;
        
        numHistograms = (int)centersIDs.size();
        
        scale = pow(2, level);
        
        radius = tclcHistograms->getRadius();
        
        radius2 = radius*radius;
        
        fullWidth = bins.cols;
        fullHeight = bins.rows;
        
        _offsetX = offsetX;
        _offsetY = offsetY;
        
        hsData = (float*)heaviside.ptr<float>();
        
        _roi = roi;
        
        _eCollection = (float*)eCollection.ptr<float>();
        
        _threads = threads;
    }
    
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = _roi.height/_threads;
        
        int jEnd = r.end*range;
        if(r.end == _threads)
        {
            jEnd = _roi.height;
        }
        
        float *e = _eCollection + 3*r.start;
        
        for(int j = r.start*range; j < jEnd; j++)
        {
            int idx = j*_roi.width;
            
            for(int i = 0; i < _roi.width; i++, idx++)
            {
                float hsVal = hsData[idx];
                
                int px = i+_offsetX;
                int py = j+_offsetY;
                
                if(hsVal >= 0.0f && py >= 0 && py < fullHeight && px >= 0 && px < fullWidth)
                {
                    int pIdx = py * fullWidth + px;
                    
                    int binIdx = binsData[pIdx];
                    
                    e[2] += 1.0f;
                    
                    float pYFVal = 0;
                    float pYBVal = 0;
                    
                    int cnt = 0;
                    
                    for(int h = 0; h < numHistograms; h++)
                    {
                        cv::Point3i centerID = _centersIDs[h];
                        
                        if(initializedData[centerID.z])
                        {
                            int dx = centerID.x - scale*(i+_roi.x + 0.5f);
                            int dy = centerID.y - scale*(j+_roi.y + 0.5f);
                            
                            int distance = dx*dx + dy*dy;
                            
                            if(distance <= radius2)
                            {
                                float pyf = localFG.at<float>(centerID.z, binIdx);
                                float pyb = localBG.at<float>(centerID.z, binIdx);
                                
                                pyf += 0.0000001f;
                                pyb += 0.0000001f;
                                
                                pYFVal += pyf / (pyf + pyb);
                                
                                cnt++;
                            }
                        }
                    }
                    
                    if(cnt > 1)
                    {
                        pYFVal /= cnt;
                        pYBVal = 1.0f - pYFVal;
                        
                        e[0] += -log(hsVal * (pYFVal - pYBVal) + pYBVal);
                        e[1] += 1.0f;
                    }
                }
            }
        }
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, the RGB values per pixel
 *  of a color input image are converted to their corresponding histogram bin
 *  index.
 */
class Parallel_For_convertToBins: public cv::ParallelLoopBody
{
private:
    cv::Mat _frame;
    cv::Mat _binned;
    
    uchar *frameData;
    int *binnedData;
    
    int _numBins;
    
    int _binShift;
    
    int _threads;
    
public:
    Parallel_For_convertToBins(const cv::Mat &frame, cv::Mat &binned, int numBins, int threads)
    {
        _frame = frame;
        
        binned.create(_frame.rows, _frame.cols, CV_32SC1);
        _binned = binned;
        
        frameData = _frame.data;
        binnedData = (int*)_binned.ptr<int>();
        
        _numBins = numBins;
        
        _binShift = 8 - log(numBins)/log(2);
        
        _threads = threads;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = _frame.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _frame.rows;
        }
        
        for(int y = r.start*range; y < yEnd; y++)
        {
            uchar *frameRow = frameData + y*_frame.cols*3;
            int *binnedRow = binnedData + y*_binned.cols;
            
            int idx = 0;
            for(int x = 0; x < _frame.cols; x++, idx+=3)
            {
                int ru = (frameRow[idx] >> _binShift);
                int gu = (frameRow[idx + 1] >> _binShift);
                int bu = (frameRow[idx + 2] >> _binShift);
                
                int binIdx = (ru * _numBins + gu) * _numBins + bu;
                
                binnedRow[x] = binIdx;
            }
        }
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for each pixel of a color
 *  input imagec (represented by the corrsponding histogram bin index) the
 *  average foreground and backgorund posterior probalility across a set of
 *  pre-computed tclc-histograms is computed. If that foregorund probablilty
 *  is greater than the background probability, the value of the pixel in the
 *  resulting posterior response map is set to 255 and 0 otherwise.
 */
class Parallel_For_createPosteriorResponseMap: public cv::ParallelLoopBody
{
private:
    cv::Mat localFG;
    cv::Mat localBG;
    
    uchar *initializedData;
    
    int numHistograms;
    int numBins;
    
    cv::Mat _binned;
    cv::Mat _map;
    
    int *binnedData;
    uchar *mapData;
    
    int _threads;
    
public:
    Parallel_For_createPosteriorResponseMap(TCLCHistograms *tclcHistograms, const cv::Mat &binned, cv::Mat &map, int threads)
    {
        localFG = tclcHistograms->getLocalForegroundHistograms();
        localBG = tclcHistograms->getLocalBackgroundHistograms();
        
        initializedData = tclcHistograms->getInitialized().data;
        
        numHistograms = (int)tclcHistograms->getInitialized().cols;
        
        numBins = tclcHistograms->getNumBins();
        
        _binned = binned;
        
        map.create(_binned.rows, _binned.cols, CV_8UC1);
        _map = map;
        
        binnedData = (int*)_binned.ptr<int>();
        mapData = _map.data;
        
        _threads = threads;
    }
    
    virtual void operator()(const cv::Range &r) const
    {
        int range = _binned.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _binned.rows;
        }
        
        char *LUT = new char[numBins*numBins*numBins]();
        
        for(int y = r.start*range; y < yEnd; y++)
        {
            int *binnedRow = binnedData + y*_binned.cols;
            uchar *mapRow = mapData + y*_map.cols;
            
            for(int x = 0; x < _binned.cols; x++)
            {
                int binIdx = binnedRow[x];
                
                char resLUT = LUT[binIdx];
                
                if(resLUT)
                {
                    if(resLUT == 2)
                    {
                        mapRow[x] = 255;
                    }
                    else
                    {
                        mapRow[x] = 0;
                    }
                }
                else
                {
                    float pYFVal = 0.0f;
                    float pYBVal = 0.0f;
                    
                    int cnt = 0;
                    
                    for(int h = 0; h < numHistograms; h++)
                    {
                        if(initializedData[h])
                        {
                            float pyf = localFG.at<float>(h, binIdx);
                            float pyb = localBG.at<float>(h, binIdx);
                            
                            if(pyf > 0.0f || pyb > 0.0f)
                            {
                                pyf += 0.0000001f;
                                pyb += 0.0000001f;
                                
                                pYFVal += pyf / (pyf + pyb);
                                pYBVal += pyb / (pyf + pyb);
                            }
                            cnt++;
                        }
                    }
                    
                    if(cnt)
                    {
                        pYFVal /= cnt;
                        pYBVal /= cnt;
                    }
                    
                    bool isFG = pYFVal > pYBVal;
                    
                    if(isFG)
                    {
                        mapRow[x] = 255;
                        LUT[binIdx] = 2;
                    }
                    else
                    {
                        mapRow[x] = 0;
                        LUT[binIdx] = 1;
                    }
                }
            }
        }
        delete[] LUT;
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. It is the super class for parallelized template matching using
 *  either base templates or neighboring templates. Fot this, it provides an
 *  efficient method for cost function evaluation based on a compressed template
 *  representation.
 */
class Parallel_For_templateMatcher: public cv::ParallelLoopBody
{
public:
    
    float evaluateEnergyFunction(TCLCHistograms *tclcHistograms, const std::vector<PixelData> &compressedPixelData, const cv::Mat &binned, const cv::Rect &roi, int offsetX, int offsetY) const
    {
        float e = 0.0f;
        int sum = 0;
        
        int *binsData = (int*)binned.ptr<int>();
        
        cv::Mat localFG = tclcHistograms->getLocalForegroundHistograms();
        cv::Mat localBG = tclcHistograms->getLocalBackgroundHistograms();
        
        uchar *initializedData = tclcHistograms->getInitialized().data;
        
        int fullWidth = binned.cols;
        int fullHeight = binned.rows;
        
        for(int p = 0; p < compressedPixelData.size(); p++)
        {
            PixelData pixelData = compressedPixelData[p];
            
            float hsVal = pixelData.hsVal;
            
            int px = pixelData.x+offsetX;
            int py = pixelData.y+offsetY;
            
            if(py >= 0 && py < fullHeight && px >= 0 && px < fullWidth)
            {
                int pIdx = py*fullWidth + px;
                int binIdx = binsData[pIdx];
                
                float pYFVal = 0;
                float pYBVal = 0;
                
                int cnt = 0;
                for(int i = 0; i < pixelData.ids_size; i++)
                {
                    int hID = pixelData.ids[i];
                    if(initializedData[hID])
                    {
                        float pyf = localFG.at<float>(hID, binIdx);
                        float pyb = localBG.at<float>(hID, binIdx);
                        
                        pyf += 0.0000001f;
                        pyb += 0.0000001f;
                        
                        pYFVal += pyf / (pyf + pyb);
                        pYBVal += pyb / (pyf + pyb);
                        
                        cnt++;
                    }
                }
                
                if(cnt > 1)
                {
                    pYFVal /= cnt;
                    pYBVal /= cnt;
                    
                    e += -log(hsVal * (pYFVal - pYBVal) + pYBVal);
                    
                    sum++;
                }
            }
        }
        
        if(sum && (float)sum/compressedPixelData.size() > 0.5f)
            e /= sum;
        else
            e = FLT_MAX;
        
        return e;
    }
    
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, tempalte matching for all
 *  base templates across the whole image is performed in a sliding window manner.
 *  This is accelerated by using a posterior response map to quickly detect regions
 *  where the cost functioin must not be evaluated.
 */
class Parallel_For_exhaustiveSearch: public Parallel_For_templateMatcher
{
private:
    Object3D *object;
    std::vector<TemplateView*> templateViews;
    
    cv::Mat binned;
    cv::Mat prMap;
    
    int level;
    int step;
    int diameter;
    
public:
    Parallel_For_exhaustiveSearch(Object3D *object, std::vector<TemplateView*> &templateViews, const cv::Mat &binned, const cv::Mat &prMap, int level, int step, int diameter)
    {
        this->object = object;
        this->templateViews = templateViews;
        
        this->binned = binned;
        this->prMap = prMap;
        
        this->level = level;
        this->step = step;
        this->diameter = diameter;
    }
    
    float computeMapMaskMatch(const cv::Mat &map, const cv::Mat &mask, int etaF, int offsetX, int offsetY, int innerOffset) const
    {
        float score = 0.0f;
        
        uchar *mapData = map.data;
        uchar *maskData = mask.data;
        
        int fullWidth = map.cols;
        int fullHeight = map.rows;
        
        int cnt = 0;
        
        for(int j = innerOffset; j < mask.rows - innerOffset; j++)
        {
            int idx = j*mask.cols + innerOffset;
            
            for(int i = innerOffset; i < mask.cols - innerOffset; i++, idx++)
            {
                int px = i+offsetX;
                int py = j+offsetY;
                
                if(py >= 0 && py < fullHeight && px >= 0 && px < fullWidth)
                {
                    int pIdx = py * fullWidth + px;
                    
                    if(maskData[idx] && mapData[pIdx])
                    {
                        cnt++;
                    }
                }
            }
        }
        
        score = (float)cnt/etaF;
        
        return score;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        for(int t = r.start; t < r.end; t++)
        {
            TCLCHistograms *tclcHistograms = object->getTCLCHistograms();
            
            int innerOffset = tclcHistograms->getRadius()/pow(2, level);
            
            TemplateView *tv = templateViews[t];
            cv::Rect roi = tv->getROI(level);
            cv::Mat heaviside = tv->getHeaviside(level);
            std::vector<cv::Point3i> centersIDs = tv->getCentersAndIDs(level);
            
            cv::Mat mask = tv->getMask(level);
            int etaF = tv->getEtaF(level);
            
            std::vector<PixelData> compressedPixelData = tv->getCompressedPixelData(level);
            
            int xStart, xEnd, yStart, yEnd;
            
            if(diameter <= 0)
            {
                xStart = 0;;
                xEnd = binned.cols - roi.width;
                yStart = 0;;
                yEnd = binned.rows - roi.height;
            }
            else
            {
                cv::Point3f offset = tv->getCurrentOffset(level);
                xStart = offset.x - diameter;
                xEnd = offset.x + diameter+1;
                yStart = offset.y - diameter;
                yEnd = offset.y + diameter+1;
            }
            
            float minE = FLT_MAX;
            int finalX = 0;
            int finalY = 0;
            
            
            int initCnt = 0;
            for(int i = 0; i < centersIDs.size(); i++)
            {
                if(tclcHistograms->getInitialized().data[centersIDs[i].z])
                    initCnt++;
            }
            
            if((float)initCnt/centersIDs.size() > 0.5f)
            {
                std::vector<cv::Point2i> offsets;
                for(int offsetY = yStart; offsetY < yEnd; offsetY+=step)
                {
                    for(int offsetX = xStart; offsetX < xEnd; offsetX+=step)
                    {
                        if(computeMapMaskMatch(prMap, mask, etaF, offsetX, offsetY, innerOffset) > 0.5f)
                        {
                            offsets.push_back(cv::Point2i(offsetX, offsetY));
                            
                            float e = evaluateEnergyFunction(tclcHistograms, compressedPixelData, binned, roi, offsetX, offsetY);
                            
                            if(e < minE)
                            {
                                minE = e;
                                finalX = offsetX;
                                finalY = offsetY;
                            }
                        }
                    }
                }
            }
            
            cv::Point3f offset = cv::Point3f(finalX, finalY, minE);
            tv->setCurrentOffset(offset, level);
        }
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, template matching is performed
 *  for all neighboring templates corresponding to one base template at the 2D location
 *  where this base template matched best at the lower image pyramid level during the
 *  previous exhaustive search.
 */
class Parallel_For_neighborSearch: public Parallel_For_templateMatcher
{
private:
    Object3D *object;
    TemplateView *templateView;
    std::vector<TemplateView*> neighbors;
    
    cv::Mat binned;
    
    int offsetX0;
    int offsetY0;
    
    int centerX0;
    int centerY0;
    
    int level;
    int levelDiff;
    
public:
    Parallel_For_neighborSearch(Object3D *object, TemplateView *templateView, const cv::Mat &binned, int level, int levelDiff)
    {
        this->object = object;
        this->templateView = templateView;
        this->neighbors = templateView->getNeighborTemplates();
        
        this->binned = binned;
        
        cv::Point3f offset0 = templateView->getCurrentOffset(level);
        cv::Rect roi0 = templateView->getROI(level);
        
        this->offsetX0 = offset0.x*pow(2, levelDiff);
        this->offsetY0 = offset0.y*pow(2, levelDiff);
        
        this->centerX0 = roi0.x + roi0.width/2;
        this->centerY0 = roi0.y + roi0.height/2;
        
        this->level = level;
        this->levelDiff = levelDiff;
    }
    
    
    virtual void operator()( const cv::Range &r ) const
    {
        for(int t = r.start; t < r.end; t++)
        {
            TCLCHistograms *tclcHistograms = object->getTCLCHistograms();
            
            TemplateView *neighbor = neighbors[t];
            cv::Rect roi = neighbor->getROI(level);
            cv::Mat heaviside = neighbor->getHeaviside(level);
            std::vector<cv::Point3i> centersIDs = neighbor->getCentersAndIDs(level);
            
            std::vector<PixelData> compressedPixelData = neighbor->getCompressedPixelData(level);
            
            int centerX = roi.x + roi.width/2;
            int centerY = roi.y + roi.height/2;
            
            int offsetX = offsetX0 + (centerX0 - centerX);
            int offsetY = offsetY0 + (centerY0 - centerY);
            
            float e = evaluateEnergyFunction(tclcHistograms, compressedPixelData, binned, roi, offsetX, offsetY);
            
            cv::Point3f offset(offsetX, offsetY, e);
            
            neighbor->setCurrentOffset(offset, level);
        }
    }
};


#endif //POSE_ESTIMATOR6D_H
