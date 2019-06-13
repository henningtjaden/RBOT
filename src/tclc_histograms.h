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

#ifndef TCLC_HISTOGRAMS_H
#define TCLC_HISTOGRAMS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class Model;

/**
 *  This class implements an statistical image segmentation model based on temporary
 *  consistent, local color histograms (tclc-histograms). Here, each histogram corresponds
 *  to a 3D vertex of a given 3D model.
 */
class TCLCHistograms
{
public:
    /**
     *  Constructor that allocates both normalized and not normalized foreground
     *  and background histograms for each vertex of the given 3D model.
     *
     *  @param  model The 3D model for which the histograms are being created.
     *  @param  numBins The number of bins per color channel.
     *  @param  radius The radius of the local image region in pixels used for updating the histograms.
     *  @param  offset The minimum distance between two projected histogram centers in pixels during an update.
     */
    TCLCHistograms(Model *model, int numBins, int radius, float offset);
    
    ~TCLCHistograms();
    
    /**
     *  Updates the histograms from a given camera frame by projecting all histogram
     *  centers into the image and selecting those close or on the object's contour.
     *
     *  @param  frame The color frame to be used for updating the histograms.
     *  @param  mask The corresponding binary shilhouette mask of the object.
     *  @param  depth The per pixel depth map of the object used to filter histograms on the back of the object,
     *  @param  K The camera's instrinsic matrix.
     *  @param  zNear The near plane used to render the depth map.
     *  @param  zFar The far plane used to render the depth map.
     */
    void update(const cv::Mat &frame, const cv::Mat &mask, const cv::Mat &depth, cv::Matx33f &K, float zNear, float zFar);
    
    /**
     *  Computes updated center locations and IDs of all histograms that project onto or close
     *  to the contour based on the current object pose at a specified image pyramid level.
     *
     *  @param  mask The binary shilhouette mask of the object.
     *  @param  depth The per pixel depth map of the object used to filter histograms on the back of the object,
     *  @param  K The camera's instrinsic matrix.
     *  @param  zNear The near plane used to render the depth map.
     *  @param  zFar The far plane used to render the depth map.
     *  @param  level The image pyramid level to be used for the update.
     */
    void updateCentersAndIds(const cv::Mat &mask, const cv::Mat &depth, const cv::Matx33f &K, float zNear, float zFar, int level);
    
    /**
     *  Returns all normalized forground histograms in their current state.
     *
     *  @return The normalized foreground histograms.
     */
    cv::Mat getLocalForegroundHistograms();
    
    /**
     *  Returns all normalized background histograms in their current state.
     *
     *  @return The normalized background histograms.
     */
    cv::Mat getLocalBackgroundHistograms();
    
    /**
     *  Returns the locations and IDs of all histogram centers that where used for the last
     *  update() or updateCentersAndIds() call.
     *
     *  @return The list of all current center locations on or close to the contour and their corresponding IDs [(x_0, y_0, id_0), (x_1, y_1, id_1), ...].
     */
    std::vector<cv::Point3i> getCentersAndIDs();
    
    /**
     *  Returns a 1D binary mask of all histograms where a '1' means that the histograms
     *  corresponding to the index has been intialized before.
     *
     *  @return A 1D binary mask telling wheter each histogram has been initialized or not.
     */
    cv::Mat getInitialized();
    
    /**
     *  Returns the number of histogram bin per image channel as specified in the constructor.
     *
     *  @return The number of histogram bins per channel.
     */
    int getNumBins();
    
    /**
     *  Returns the number of histograms, i.e. verticies of the corresponding 3D model.
     *
     *  @return The number of histograms.
     */
    int getNumHistograms();
    
    /**
     *  Returns the radius of the local image region in pixels used for updating the
     *  histograms as specified in the constructor.
     *
     *  @return The minumum distance between two projected histogram centers in pixels.
     */
    int getRadius();
    
    /**
     *  Returns the minumum distance between two projected histogram centers during an update
     *  as specified in the constructor.
     *
     *  @return The minumum distance between two projected histogram centers in pixels.
     */
    float getOffset();
    
    /**
     *  Clears all histograms by resetting them to zero and setting their status to
     *  uninitialized
     */
    void clear();
    
    
private:
    int numBins;
    
    int _numHistograms;
    
    int radius;
    
    float _offset;
    
    cv::Mat notNormalizedFG;
    cv::Mat notNormalizedBG;
    
    cv::Mat normalizedFG;
    cv::Mat normalizedBG;
    
    cv::Mat initialized;
    
    Model* _model;
    
    std::vector<cv::Point3i> _centersIDs;
    
    std::vector<cv::Point3i> computeLocalHistogramCenters(const cv::Mat &mask);
    
    std::vector<cv::Point3i> parallelComputeLocalHistogramCenters(const cv::Mat &mask, const cv::Mat &depth, const cv::Matx33f &K, float zNear, float zFar, int level);
    
    void filterHistogramCenters(int numHistograms, float offset);
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for every projected histogram center on or
 *  close to the object's contour, a new foreground and background color histogram are computed
 *  within a local circular image region is computed using the Bresenham algorithm to scan the
 *  corresponding pixels.
 */
class Parallel_For_buildLocalHistograms: public cv::ParallelLoopBody
{
private:
    cv::Mat _frame;
    cv::Mat _mask;
    
    uchar* frameData;
    uchar* maskData;
    
    size_t frameStep;
    size_t maskStep;
    
    cv::Size size;
    
    std::vector<cv::Point3i> _centers;
    
    int _radius;
    
    int _numBins;
    
    int _binShift;
    
    int histogramSize;
    
    cv::Mat _sumsFB;
    
    int _m_id;
    
    int* localFGData;
    int* localBGData;
    
    int* _sumsFBData;
    
    int _threads;
    
public:
    Parallel_For_buildLocalHistograms(const cv::Mat &frame, const cv::Mat &mask, const std::vector<cv::Point3i> &centers, float radius, int numBins, cv::Mat &localHistogramsFG, cv::Mat &localHistogramsBG, cv::Mat &sumsFB, int m_id, int threads)
    {
        _frame = frame;
        _mask = mask;
        
        frameData = _frame.data;
        maskData = _mask.data;
        
        frameStep = _frame.step;
        maskStep = _mask.step;
        
        size = frame.size();
        
        _centers = centers;
        
        _radius = radius;
        
        _numBins = numBins;
        
        _binShift = 8 - log(numBins)/log(2);
        
        histogramSize = localHistogramsFG.cols;
        
        localFGData = (int*)localHistogramsFG.ptr<int>();
        localBGData = (int*)localHistogramsBG.ptr<int>();
        
        _sumsFB = sumsFB;
        
        _m_id = m_id;
        
        _sumsFBData = (int*)_sumsFB.ptr<int>();
        
        _threads = threads;
    }
    
    void processLine(uchar *frameRow, uchar* maskRow, int xl, int xr, int* localHistogramFG, int* localHistogramBG, int* sumFB) const
    {
        uchar* frame_ptr = (uchar*)(frameRow) + 3*xl;
        
        uchar* mask_ptr = (uchar*)(maskRow) + xl;
        uchar* mask_max_ptr = (uchar*)(maskRow) + xr;
        
        for( ; mask_ptr <= mask_max_ptr; mask_ptr += 1, frame_ptr += 3)
        {
            int pidx;
            int ru, gu, bu;
            
            ru = (frame_ptr[0] >> _binShift);
            gu = (frame_ptr[1] >> _binShift);
            bu = (frame_ptr[2] >> _binShift);
            pidx = (ru * _numBins + gu) * _numBins + bu;
            
            if(*mask_ptr == _m_id)
            {
                localHistogramFG[pidx]++;
                sumFB[0]++;
            }
            else
            {
                localHistogramBG[pidx]++;
                sumFB[1]++;
            }
        }
    }
    
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = (int)_centers.size()/_threads;
        
        int cEnd = r.end*range;
        if(r.end == _threads)
        {
            cEnd =  (int)_centers.size();
        }
        
        cv::Mat sumsFB = _sumsFB;
        
        for(int c = r.start*range; c < cEnd; c++)
        {
            int err = 0;
            int dx = _radius;
            int dy = 0;
            int plus = 1;
            int minus = (_radius << 1) - 1;
            
            int olddx = dx;
            
            cv::Point3i center = _centers[c];
            
            int inside = center.x >= _radius && center.x < size.width - _radius && center.y >= _radius && center.y < size.height - _radius;
            
            int* localHistogramFG = localFGData + c*histogramSize;
            int* localHistogramBG = localBGData + c*histogramSize;
            
            int* sumFB = _sumsFBData + c*2;
            
            while( dx >= dy )
            {
                int mask;
                int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
                int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;
                
                if( inside )
                {
                    uchar *frameRow0 = frameData + y11 * frameStep;
                    uchar *frameRow1 = frameData + y12 * frameStep;
                    
                    uchar *maskRow0 = maskData + y11 * maskStep;
                    uchar *maskRow1 = maskData + y12 * maskStep;
                    
                    processLine(frameRow0, maskRow0, x11, x12, localHistogramFG, localHistogramBG, sumFB);
                    if(y11 != y12) processLine(frameRow1, maskRow1, x11, x12, localHistogramFG, localHistogramBG, sumFB);
                    
                    frameRow0 = frameData + y21 * frameStep;
                    frameRow1 = frameData + y22 * frameStep;
                    
                    maskRow0 = maskData + y21 * maskStep;
                    maskRow1 = maskData + y22 * maskStep;
                    
                    if(olddx != dx)
                    {
                        if(y11 != y21) processLine(frameRow0, maskRow0, x21, x22, localHistogramFG, localHistogramBG, sumFB);
                        if(y12 != y22) processLine(frameRow1, maskRow1, x21, x22, localHistogramFG, localHistogramBG, sumFB);
                    }
                }
                else if( x11 < size.width && x12 >= 0 && y21 < size.height && y22 >= 0 )
                {
                    x11 = std::max( x11, 0 );
                    x12 = MIN( x12, size.width - 1 );
                    
                    if( (unsigned)y11 < (unsigned)size.height )
                    {
                        uchar *frameRow = frameData + y11 * frameStep;
                        uchar *maskRow = maskData + y11 * maskStep;
                        
                        processLine(frameRow, maskRow, x11, x12, localHistogramFG, localHistogramBG, sumFB);
                    }
                    
                    if( (unsigned)y12 < (unsigned)size.height && (y11 != y12))
                    {
                        uchar *frameRow = frameData + y12 * frameStep;
                        uchar *maskRow = maskData + y12 * maskStep;
                        
                        processLine(frameRow, maskRow, x11, x12, localHistogramFG, localHistogramBG, sumFB);
                    }
                    
                    if( x21 < size.width && x22 >= 0 && (olddx != dx))
                    {
                        x21 = std::max( x21, 0 );
                        x22 = MIN( x22, size.width - 1 );
                        
                        if( (unsigned)y21 < (unsigned)size.height )
                        {
                            uchar *frameRow = frameData + y21 * frameStep;
                            uchar *maskRow = maskData + y21 * maskStep;
                            
                            processLine(frameRow, maskRow, x21, x22, localHistogramFG, localHistogramBG, sumFB);
                        }
                        
                        if( (unsigned)y22 < (unsigned)size.height )
                        {
                            uchar *frameRow = frameData + y22 * frameStep;
                            uchar *maskRow = maskData + y22 * maskStep;
                            
                            processLine(frameRow, maskRow, x21, x22, localHistogramFG, localHistogramBG, sumFB);
                        }
                    }
                }
                
                olddx = dx;
                
                dy++;
                err += plus;
                plus += 2;
                
                mask = (err <= 0) - 1;
                
                err -= minus & mask;
                dx += mask;
                minus -= mask & 2;
            }
        }
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, each previously computed local foreground
 *  and background color histogram is merged with their normalized temporally consistent
 *  representation based on respective learning rates.
 */
class Parallel_For_mergeLocalHistograms: public cv::ParallelLoopBody
{
private:
    int histogramSize;
    
    cv::Mat _sumsFB;
    
    int* notNormalizedFGData;
    int* notNormalizedBGData;
    
    float* normalizedFGData;
    float* normalizedBGData;
    
    uchar* initializedData;
    
    std::vector<cv::Point3i> _centersIds;
    
    float _alphaF;
    float _alphaB;
    
    int* _sumsFBData;
    
    int _threads;
    
public:
    Parallel_For_mergeLocalHistograms(const cv::Mat &notNormalizedFG, const cv::Mat &notNormalizedBG, cv::Mat &normalizedFG, cv::Mat &normalizedBG, cv::Mat &initialized, const std::vector<cv::Point3i> centersIds, const cv::Mat &sumsFB, float alphaF, float alphaB, int threads)
    {
        histogramSize = notNormalizedFG.cols;
        
        notNormalizedFGData = (int*)notNormalizedFG.ptr<int>();
        notNormalizedBGData = (int*)notNormalizedBG.ptr<int>();
        
        normalizedFGData = (float*)normalizedFG.ptr<float>();
        normalizedBGData = (float*)normalizedBG.ptr<float>();
        
        initializedData = initialized.data;
        
        _centersIds = centersIds;
        
        _sumsFB = sumsFB;
        
        _alphaF = alphaF;
        _alphaB = alphaB;
        
        _sumsFBData = (int*)_sumsFB.ptr<int>();
        
        _threads = threads;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = _sumsFB.rows/_threads;
        
        int hEnd = r.end*range;
        if(r.end == _threads)
        {
            hEnd = _sumsFB.rows;
        }
        
        for(int h = r.start*range; h < hEnd; h++)
        {
            int cID = _centersIds[h].z;
            
            int* notNormalizedFG = notNormalizedFGData + h*histogramSize;
            int* notNormalizedBG = notNormalizedBGData + h*histogramSize;
            
            float* normalizedFG = normalizedFGData + cID*histogramSize;
            float* normalizedBG = normalizedBGData + cID*histogramSize;
            
            int totalFGPixels = _sumsFBData[h*2];
            int totalBGPixels = _sumsFBData[h*2 + 1];
            
            if(initializedData[cID] == 0)
            {
                for(int i = 0; i < histogramSize; i++)
                {
                    if(false)
                    {
                        normalizedFG[i] = (float)notNormalizedFG[i]/totalFGPixels;
                        normalizedBG[i] = (float)notNormalizedBG[i]/totalBGPixels;
                        
                    }
                    else
                    {
                        if(notNormalizedFG[i])
                        {
                            normalizedFG[i] = (float)notNormalizedFG[i]/totalFGPixels;
                        }
                        if(notNormalizedBG[i])
                        {
                            normalizedBG[i] = (float)notNormalizedBG[i]/totalBGPixels;
                        }
                    }
                }
                initializedData[cID] = 1;
            }
            else
            {
                for(int i = 0; i < histogramSize; i++)
                {
                    
                    if(false)
                    {
                        normalizedFG[i] = (float)notNormalizedFG[i]/totalFGPixels;
                        normalizedBG[i] = (float)notNormalizedBG[i]/totalBGPixels;
                    }
                    else
                    {
                        if(notNormalizedFG[i])
                        {
                            normalizedFG[i] = (1.0f - _alphaF)*normalizedFG[i] + _alphaF*(float)notNormalizedFG[i]/totalFGPixels;
                        }
                        if(notNormalizedBG[i])
                        {
                            normalizedBG[i] = (1.0f - _alphaB)*normalizedBG[i] + _alphaB*(float)notNormalizedBG[i]/totalBGPixels;
                        }
                    }
                }
                
            }
        }
    }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, every 3D histogram center is projected
 *  into the image plane. Those that do not project on or close to the object's contour are
 *  being filtered based on a given binary silhouette mask and depth map at a specified image
 *  pyramid level.
 */
class Parallel_For_computeHistogramCenters: public cv::ParallelLoopBody
{
private:
    std::vector<cv::Vec3f> _verticies;
    
    std::vector<cv::Point3i>* _centersIds;
    
    cv::Mat _depth;
    cv::Mat _mask;
    
    uchar* maskData;
    
    cv::Matx44f _T_cm;
    cv::Matx33f _K;
    
    float _zNear;
    float _zFar;
    
    int _m_id;
    
    int _level;
    
    int downScale;
    int upScale;
    
    int _threads;
    
public:
    Parallel_For_computeHistogramCenters(const cv::Mat &mask, const cv::Mat &depth, const std::vector<cv::Vec3f> &verticies, const cv::Matx44f &T_cm, const cv::Matx33f &K, float zNear, float zFar, int m_id, int level, std::vector<cv::Point3i>* centersIds, int threads)
    {
        _verticies = verticies;
        
        _depth = depth;
        
        _level = level;
        
        downScale = pow(2, 2 - level);
        
        upScale = pow(2, level);
        
        if(mask.type()%8 == 5)
        {
            mask.convertTo(_mask, CV_8UC1, 10000);
        }
        else
        {
            _mask = mask;
        }
        
        maskData = mask.data;
        
        _T_cm = T_cm;
        _K = K;
        
        _zNear = zNear;
        _zFar = zFar;
        
        _m_id = m_id;
        
        _centersIds = centersIds;
        
        _threads = threads;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = (int)_verticies.size()/_threads;
        
        int vEnd = r.end*range;
        if(r.end == _threads)
        {
            vEnd = (int)_verticies.size();
        }
        
        std::vector<cv::Point3i>* tmp = &_centersIds[r.start];
        
        for(int v = r.start*range; v < vEnd; v++)
        {
            cv::Vec3f V_m = _verticies[v];
            
            float X_m = V_m[0];
            float Y_m = V_m[1];
            float Z_m = V_m[2];
            
            float X_c = X_m*_T_cm(0, 0) + Y_m*_T_cm(0, 1) + Z_m*_T_cm(0, 2) + _T_cm(0, 3);
            float Y_c = X_m*_T_cm(1, 0) + Y_m*_T_cm(1, 1) + Z_m*_T_cm(1, 2) + _T_cm(1, 3);
            float Z_c = X_m*_T_cm(2, 0) + Y_m*_T_cm(2, 1) + Z_m*_T_cm(2, 2) + _T_cm(2, 3);
            
            float x = X_c/Z_c*_K(0, 0) + _K(0, 2);
            float y = Y_c/Z_c*_K(1, 1) + _K(1, 2);
            
            if(x >= 0 && x < _depth.cols && y >= 0 && y < _depth.rows)
            {
                float d = 1.0f - _depth.at<float>(y, x);
                
                float Z_d = 2.0f * _zNear * _zFar / (_zFar + _zNear - (2.0f*(d) - 1.0) * (_zFar - _zNear));
                
                if(fabs(Z_c - Z_d) < 1.0f || d == 1.0)
                {
                    int xi = (int)x;
                    int yi = (int)y;
                    
                    if(xi >= downScale && xi < _mask.cols - downScale && yi >= downScale && yi < _mask.rows - downScale)
                    {
                        uchar v0 = maskData[yi*_mask.cols + xi] == _m_id;
                        uchar v1 = maskData[yi*_mask.cols + xi + downScale] == _m_id;
                        uchar v2 = maskData[yi*_mask.cols + xi - downScale] == _m_id;
                        uchar v3 = maskData[(yi + downScale)*_mask.cols + xi] == _m_id;
                        uchar v4 = maskData[(yi - downScale)*_mask.cols + xi] == _m_id;
                        
                        if(v0*v1*v2*v3*v4 == 0)
                        {
                            tmp->push_back(cv::Point3i(x*upScale, y*upScale, v));
                        }
                    }
                }
            }
        }
    }
};

#endif /* TCLC_HISTOGRAMS_H */
