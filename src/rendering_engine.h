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

#ifndef RENDERING_ENGINE
#define RENDERING_ENGINE

#include <iostream>

#include <QOpenGLContext>
#include <QOffscreenSurface>

#include <QGLFramebufferObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_3_3_Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "transformations.h"
#include "model.h"

/**
 *  This class implements an OpenGL-based offscreen rendering engine for generating
 *  images of projected 3D meshes based on given object poses and camera instrinsics.
 *  It supports one or mutiple objects to be rendered as binary masks, depth maps,
 *  normal maps or phong-shaded. It also allows to perform all renderings according
 *  to a specified image pyramid level at lower resolutions. The class is  implemented
 *  as a singleton.
 */
class RenderingEngine : public QOpenGLFunctions_3_3_Core
{
public:
    enum FrameType {
        MASK,
        RGB,
        RGB_32F,
        DEPTH
    };
    
    RenderingEngine(void);
    
    ~RenderingEngine(void);
    
    static RenderingEngine *Instance(void)
    {
        if (instance == NULL) instance = new RenderingEngine();
        return instance;
    }
    
    /**
     *  Initializes the rendering engine instance given a 3x3 float
     *  intrinsic camera matrix
     *  K = [fx 0 cx]
     *      [0 fy cy]
     *      [0  0  1],
     *  a desired image resolution, a near and a far plane as well as the number
     *  of pyramidf level supported for the renderings.
     *
     *  @param  K The intrinsic camera matrix.
     *  @param  width The width in pixels of the rendered images at level 0.
     *  @param  height The height in pixels of the rendered images at level 0.
     *  @param  zNear The distance of the OpenGL near plane.
     *  @param  zFar The distance of the OpenGL far plane.
     *  @param  numLevels Number of supported pyramid levels with a downscale factor of 2.
     */
    void init(const cv::Matx33f &K, int width, int height, float zNear, float zFar, int numLevels);
    
    /**
     *  Returns the number of supported pyramid levels for rendering.
     *
     *  @return  The number of supported pyramid levels for rendering.
     */
    int getNumLevels();
    
    /**
     *  Sets a pyramid level to be used for rendering between 0 (full resolution)
     *  and getNumLevels() (the smallest resolution).
     *
     *  @param level The pyramid level to be used for rendering.
     */
    void setLevel(int level);
    
    /**
     *  Returns the current pyramid level used for rendering.
     *
     *  @return  The current pyramid level used for rendering.
     */
    int getLevel();
    
    /**
     *  Activates the OpenGL context of the rendering engine.
     */
    void makeCurrent();
    
    /**
     *  Deactivates the OpenGL context of the rendering engine.
     */
    void doneCurrent();
    
    /**
     *  Returns the OpenGL context of the rendering engine.
     *
     *  @return  The OpenGL context of the rendering engine.
     */
    QOpenGLContext *getContext();
    
    /**
     *  Returns the OpenGL ID of the frame buffer object used for offscreen rendering.
     *
     *  @return  The OpenGL ID of the frame buffer object used for offscreen rendering.
     */
    GLuint getFrameBufferID();
    
    /**
     *  Returns the OpenGL texture ID of the rendered color image.
     *
     *  @return  The OpenGL texture ID of the rendered color image.
     */
    GLuint getColorTextureID();
    
    /**
     *  Returns the OpenGL texture ID of the rendered depth buffer.
     *
     *  @return  The OpenGL texture ID of the rendered depth buffer.
     */
    GLuint getDepthTextureID();
    
    /**
     *  Returns the Z-distance of the near plane.
     *
     *  @return  The the Z-distance of the near plane.
     */
    float getZNear();
    
    /**
     *  Returns the Z-distance of the far plane.
     *
     *  @return  The the Z-distance of the far plane.
     */
    float getZFar();
    
    /**
     *  Returns a 4x4 float version of the intrinsic camera matrix wrt the current
     *  pyramid level
     *  K_4x4 = [fx/s 0 cx/s 0]
     *          [0 fy/s cy/s 0]
     *          [0  0    1   0]
     *          [0  0    0   1],
     *  with s = 1/2^level.
     *
     *  @return  A 4x4 float version of the intrinsic camera matrix wrt the current
     *  pyramid level.
     */
    cv::Matx44f getCalibrationMatrix();
    
    /**
     *  Renders a single model with a constant color and no shading in order to
     *  obtain a binary silhouette mask of it wrt its current pose.
     *
     *  @param model The model to be rendered.
     *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
     *  @param invertDepth Whether to invert the depth test during rendering (default = false).
     *  @param r The red intensity of the model surface albedo in [0, 1] (default = 1.0).
     *  @param g The green intensity of the model surface albedo in [0, 1] (default = 1.0).
     *  @param b The blue intensity of the model surface albedo in [0, 1] (default = 1.0).
     *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
     */
    void renderSilhouette(Model *model, GLenum polyonMode, bool invertDepth = false, float r = 1.0f, float g = 1.0f, float b = 1.0f, bool drawAll = false);
    
    /**
     *  Renders a single model wrt its current pose using Phong shading.
     *
     *  @param model The model to be rendered.
     *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
     *  @param r The red intensity of the model surface albedo in [0, 1] (default = 1.0).
     *  @param g The green intensity of the model surface albedo in [0, 1] (default = 0.5).
     *  @param b The blue intensity of the model surface albedo in [0, 1] (default = 0.0).
     *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
     */
    void renderShaded(Model *model, GLenum polyonMode, float r = 1.0f, float g = 0.5f, float b = 0.0f, bool drawAll = false);
    
    /**
     *  Renders the per pixel surface normals of single model wrt its current pose where the
     *  normal direction is mapped from (x, y, z) in [-1, 1] to (r, g, b) in [0, 1].
     *
     *  @param model The model to be rendered.
     *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
     *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
     */
    void renderNormals(Model *model, GLenum polyonMode, bool drawAll = false);
    
    /**
     *  Renders a multiple models in a common scene with a constant color and no shading
     *  in order to obtain a their binary silhouette masks with correct occlusions
     *  according to their current poses. If no colors are spefified each model will by default
     *  get rendered with a constant color corresponding to their model index in the red channel.
     *
     *  @param model The models to be rendered.
     *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
     *  @param invertDepth Whether to invert the depth test during rendering (default = false).
     *  @param colors A vector of colors to be used for each model (default = empty).
     *  @param drawAll Whether to draw all models even if they been not yet initlaized for tracking (default = false).
     */
    void renderSilhouette(std::vector<Model*> models, GLenum polyonMode, bool invertDepth = false, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);
    
    /**
     *  Renders a multiple models in a common scene wrt their current poses using Phong shading.
     *
     *  @param model The models to be rendered.
     *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
     *  @param colors A vector of colors to be used for each model (default = empty).
     *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
     */
    void renderShaded(std::vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f> &colors = std::vector<cv::Point3f>(), bool drawAll = false);
    
    /**
     *  Renders the per pixel surface normals of multiple models in a common scene wrt their
     *  current poses where the normal direction is mapped from (x, y, z) in [-1, 1] to (r, g, b)
     *  in [0, 1].
     *
     *  @param model The model to be rendered.
     *  @param polyonMode The OpenGL polygon mode to be used (e.g. GL_FILL).
     *  @param drawAll Whether to draw the model even if it has not yet been initlaized for tracking (default = false).
     */
    void renderNormals(std::vector<Model*> models, GLenum polyonMode, bool drawAll = false);
    
    /**
     *  Projects the eight corners of a model's bouding box into the image and computes the
     *  enclosing 2D bounding rect of these projections wrt the model's poae.
     *
     *  @param model The model of which the bounding box is to be projected.
     *  @param projections The resulting 2D coordinates of the projected bounding box corners.
     *  @param boundingRect The resulting 2D bounding rect of the 2D projections.
     */
    void projectBoundingBox(Model *model, std::vector<cv::Point2f> &projections, cv::Rect &boundingRect);
    
    /**
     *  Downloads the most recently rendered image from the GPU to the host memory and converts
     *  it to an OpenCV image depending on a given frametype. Use MASK to obtain a silhouette
     *  mask image (single channel, uchar), RGB to obtain a color image (RGB, uchar), RGB_32F
     *  to obtain color image with normalized intensities in [0, 1] (RGB, float) or DEPTH to
     *  obtain the depth buffer.
     *
     *  @param type The frame type to be downloaded and returned (e.g. MASK, RGB, RGB32F or DEPTH).
     *
     *  @return  The most recently rendered image according to the desired frame type.
     */
    cv::Mat downloadFrame(RenderingEngine::FrameType type);
    
    /**
     *  Destroys and deletes the current rendering engine singleton instance.
     */
    void destroy();

    
private:
    static RenderingEngine *instance;
    
    int width;
    int height;
    
    int fullWidth;
    int fullHeight;
    
    float zNear;
    float zFar;
    
    int numLevels;
    
    int currentLevel;
    
    std::vector<cv::Matx44f> calibrationMatrices;
    cv::Matx44f projectionMatrix;
    cv::Matx44f lookAtMatrix;
    
    QOffscreenSurface *surface;
    QOpenGLContext *glContext;
    
    GLuint frameBufferID;
    GLuint colorTextureID;
    GLuint depthTextureID;
    
    int angle;
    
    cv::Vec3f lightPosition;
    
    QString shaderFolder;
    QOpenGLShaderProgram *silhouetteShaderProgram;
    QOpenGLShaderProgram *phongblinnShaderProgram;
    QOpenGLShaderProgram *normalsShaderProgram;
    
    bool initRenderingBuffers();
    
    bool initShaderProgram(QOpenGLShaderProgram *program, QString shaderName);
    
};


#endif //RENDERING_ENGINE
