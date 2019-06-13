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

#include "rendering_engine.h"

#include <iostream>

using namespace std;
using namespace cv;


RenderingEngine* RenderingEngine::instance;

RenderingEngine::RenderingEngine(void)
{
    QSurfaceFormat glFormat;
    glFormat.setVersion(3, 3);
    glFormat.setProfile(QSurfaceFormat::CoreProfile);
    glFormat.setRenderableType(QSurfaceFormat::OpenGL);
    
    surface = new QOffscreenSurface();
    surface->setFormat(glFormat);
    surface->create();
    
    glContext = new QOpenGLContext();
    glContext->setFormat(surface->requestedFormat());
    glContext->create();
    
    silhouetteShaderProgram = new QOpenGLShaderProgram();
    phongblinnShaderProgram = new QOpenGLShaderProgram();
    normalsShaderProgram = new QOpenGLShaderProgram();
    
    calibrationMatrices.push_back(Matx44f::eye());
    
    projectionMatrix = Transformations::perspectiveMatrix(40, 4.0f/3.0f, 0.1, 1000.0);
    
    lookAtMatrix = Transformations::lookAtMatrix(0, 0, 0, 0, 0, 1, 0, -1, 0);
    
    currentLevel = 0;
}

RenderingEngine::~RenderingEngine(void)
{
    glDeleteTextures(1, &colorTextureID);
    glDeleteTextures(1, &depthTextureID);
    glDeleteFramebuffers(1, &frameBufferID);
    
    delete phongblinnShaderProgram;
    delete normalsShaderProgram;
    delete silhouetteShaderProgram;
    delete surface;
}

void RenderingEngine::destroy()
{
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    delete instance;
    instance = NULL;
}


void RenderingEngine::makeCurrent()
{
    glContext->makeCurrent(surface);
}


void RenderingEngine::doneCurrent()
{
    glContext->doneCurrent();
}

QOpenGLContext* RenderingEngine::getContext()
{
    return glContext;
}


GLuint RenderingEngine::getFrameBufferID()
{
    return frameBufferID;
}


GLuint RenderingEngine::getColorTextureID()
{
    return colorTextureID;
}


GLuint RenderingEngine::getDepthTextureID()
{
    return depthTextureID;
}

float RenderingEngine::getZNear()
{
    return zNear;
}

float RenderingEngine::getZFar()
{
    return zFar;
}

Matx44f RenderingEngine::getCalibrationMatrix()
{
    return calibrationMatrices[currentLevel];
}

void RenderingEngine::init(const Matx33f& K, int width, int height, float zNear, float zFar, int numLevels)
{
    this->width = width;
    this->height = height;
    
    fullWidth = width;
    fullHeight = height;
    
    this->zNear = zNear;
    this->zFar = zFar;
    
    this->numLevels = numLevels;
    
    projectionMatrix = Transformations::perspectiveMatrix(K, width, height, zNear, zFar, true);
    
    makeCurrent();
    
    initializeOpenGLFunctions();
    
    //FIX FOR NEW OPENGL
    uint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    calibrationMatrices.clear();
    
    for(int i = 0; i < numLevels; i++)
    {
        float s = pow(2, i);
        
        Matx44f K_l = Matx44f::eye();
        K_l(0, 0) = K(0, 0)/s;
        K_l(1, 1) = K(1, 1)/s;
        K_l(0, 2) = K(0, 2)/s;
        K_l(1, 2) = K(1, 2)/s;
        
        calibrationMatrices.push_back(K_l);
    }
    
    cout << "GL Version " << glGetString(GL_VERSION) << endl << "GLSL Version " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
    
    glEnable(GL_DEPTH);
    glEnable(GL_DEPTH_TEST);
    
    //INVERT DEPTH BUFFER
    glDepthRange(1, 0);
    glClearDepth(0.0f);
    glDepthFunc(GL_GREATER);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    
    glClearColor(0.0, 0.0, 0.0, 1.0);
    
    initRenderingBuffers();
    
    shaderFolder = "src/";
    
    initShaderProgram(silhouetteShaderProgram, "silhouette");
    initShaderProgram(phongblinnShaderProgram, "phongblinn");
    initShaderProgram(normalsShaderProgram, "normals");
    
    angle = 0;
    
    lightPosition = cv::Vec3f(0, 0, 0);
    
    doneCurrent();
}

int RenderingEngine::getNumLevels()
{
    return numLevels;
}

void RenderingEngine::setLevel(int level)
{
    currentLevel = level;
    int s = pow(2, currentLevel);
    width = fullWidth/s;
    height = fullHeight/s;
    
    width += width%4;
    height += height%4;
}


int RenderingEngine::getLevel()
{
    return currentLevel;
}


bool RenderingEngine::initRenderingBuffers()
{
    glGenTextures(1, &colorTextureID);
    glBindTexture(GL_TEXTURE_2D, colorTextureID);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    
    glGenTextures(1, &depthTextureID);
    glBindTexture(GL_TEXTURE_2D, depthTextureID);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glGenFramebuffers(1, &frameBufferID);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTextureID, 0);
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureID, 0);
    
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        cout << "error creating rendering buffers" << endl;
        return false;
    }
    return true;
}



bool RenderingEngine::initShaderProgram(QOpenGLShaderProgram *program, QString shaderName)
{
    if (!program->addShaderFromSourceFile(QOpenGLShader::Vertex, shaderFolder + shaderName + "_vertex_shader.glsl")) {
        cout << "error adding vertex shader from source file" << endl;
        return false;
    }
    if (!program->addShaderFromSourceFile(QOpenGLShader::Fragment, shaderFolder + shaderName + "_fragment_shader.glsl")) {
        cout << "error adding fragment shader from source file" << endl;
        return false;
    }
    
    if (!program->link()) {
        cout << "error linking shaders" << endl;
        return false;
    }
    return true;
}

void RenderingEngine::renderSilhouette(Model* model, GLenum polyonMode, bool invertDepth, float r, float g, float b, bool drawAll)
{
    vector<Model*> models;
    models.push_back(model);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(r, g, b));
    
    renderSilhouette(models, polyonMode, invertDepth, colors, drawAll);
}


void RenderingEngine::renderShaded(Model* model, GLenum polyonMode, float r, float g, float b, bool drawAll)
{
    vector<Model*> models;
    models.push_back(model);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(r, g, b));
    
    renderShaded(models, polyonMode, colors, drawAll);
}


void RenderingEngine::renderNormals(Model* model, GLenum polyonMode, bool drawAll)
{
    vector<Model*> models;
    models.push_back(model);
    
    renderNormals(models, polyonMode, drawAll);
}


void RenderingEngine::renderSilhouette(vector<Model*> models, GLenum polyonMode, bool invertDepth, const std::vector<cv::Point3f>& colors, bool drawAll)
{
    glViewport(0, 0, width, height);
    
    if(invertDepth)
    {
        glClearDepth(1.0f);
        glDepthFunc(GL_LESS);
    }
    
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    for(int i = 0; i < models.size(); i++)
    {
        Model* model = models[i];
        
        if(model->isInitialized() || drawAll)
        {
            Matx44f pose = model->getPose();
            Matx44f normalization = model->getNormalization();
            
            Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);
            
            Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;
            
            silhouetteShaderProgram->bind();
            silhouetteShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
            silhouetteShaderProgram->setUniformValue("uAlpha", 1.0f);
            
            Point3f color;
            if(i < colors.size())
            {
                 color = colors[i];
            }
            else
            {
                color = Point3f((float)(model->getModelID())/255.0f, 0.0f, 0.0f);
            }
            silhouetteShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));
            
            glPolygonMode(GL_FRONT_AND_BACK, polyonMode);
            
            model->draw(silhouetteShaderProgram);
        }
    }
    
    glClearDepth(0.0f);
    glDepthFunc(GL_GREATER);
    
    glFinish();
}


void RenderingEngine::renderShaded(vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f>& colors, bool drawAll)
{
    glViewport(0, 0, width, height);
    
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    for(int i = 0; i < models.size(); i++)
    {
        Model* model = models[i];
        
        if(model->isInitialized() || drawAll)
        {
            Matx44f pose = model->getPose();
            Matx44f normalization = model->getNormalization();
            
            Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);
            
            Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();
            
            Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;
            
            phongblinnShaderProgram->bind();
            phongblinnShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
            phongblinnShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
            phongblinnShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
            phongblinnShaderProgram->setUniformValue("uLightPosition1", QVector3D(0.1, 0.1, -0.02));
            phongblinnShaderProgram->setUniformValue("uLightPosition2", QVector3D(-0.1, 0.1, -0.02));
            phongblinnShaderProgram->setUniformValue("uLightPosition3", QVector3D(0.0, 0.0, 0.1));
            phongblinnShaderProgram->setUniformValue("uShininess", 100.0f);
            phongblinnShaderProgram->setUniformValue("uAlpha", 1.0f);
            
            Point3f color;
            if(i < colors.size())
            {
                color = colors[i];
            }
            else
            {
                color = Point3f(1.0, 0.5, 0.0);
            }
            phongblinnShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));
            
            glPolygonMode(GL_FRONT_AND_BACK, polyonMode);
            
            model->draw(phongblinnShaderProgram);
        }
    }
    
    glFinish();
}

void RenderingEngine::renderNormals(vector<Model*> models, GLenum polyonMode, bool drawAll)
{
    glViewport(0, 0, width, height);
    
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    
    for(int i = 0; i < models.size(); i++)
    {
        Model* model = models[i];
        
        if(model->isInitialized() || drawAll)
        {
            Matx44f pose = model->getPose();
            Matx44f normalization = model->getNormalization();
            
            Matx44f modelViewMatrix = lookAtMatrix*(pose*normalization);
            
            Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();
            
            Matx44f modelViewProjectionMatrix = projectionMatrix*modelViewMatrix;
            
            normalsShaderProgram->bind();
            normalsShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
            normalsShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
            normalsShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
            normalsShaderProgram->setUniformValue("uAlpha", 1.0f);
            
            glPolygonMode(GL_FRONT_AND_BACK, polyonMode);
            
            model->draw(normalsShaderProgram);
        }
    }
    
    glFinish();
}


void RenderingEngine::projectBoundingBox(Model* model, std::vector<cv::Point2f>& projections, cv::Rect& boundingRect)
{
    Vec3f lbn = model->getLBN();
    Vec3f rtf = model->getRTF();
    
    Vec4f Plbn = Vec4f(lbn[0], lbn[1], lbn[2], 1.0);
    Vec4f Prbn = Vec4f(rtf[0], lbn[1], lbn[2], 1.0);
    Vec4f Pltn = Vec4f(lbn[0], rtf[1], lbn[2], 1.0);
    Vec4f Plbf = Vec4f(lbn[0], lbn[1], rtf[2], 1.0);
    Vec4f Pltf = Vec4f(lbn[0], rtf[1], rtf[2], 1.0);
    Vec4f Prtn = Vec4f(rtf[0], rtf[1], lbn[2], 1.0);
    Vec4f Prbf = Vec4f(rtf[0], lbn[1], rtf[2], 1.0);
    Vec4f Prtf = Vec4f(rtf[0], rtf[1], rtf[2], 1.0);
    
    vector<Vec4f> points3D;
    points3D.push_back(Plbn);
    points3D.push_back(Prbn);
    points3D.push_back(Pltn);
    points3D.push_back(Plbf);
    points3D.push_back(Pltf);
    points3D.push_back(Prtn);
    points3D.push_back(Prbf);
    points3D.push_back(Prtf);
    
    Matx44f pose = model->getPose();
    Matx44f normalization = model->getNormalization();
    
    Point2f lt(FLT_MAX, FLT_MAX);
    Point2f rb(-FLT_MAX, -FLT_MAX);
    
    for(int i = 0; i < points3D.size(); i++)
    {
        Vec4f p = calibrationMatrices[currentLevel]*pose*normalization*points3D[i];
        
        if(p[2] == 0)
            continue;
        
        Point2f p2d = Point2f(p[0]/p[2], p[1]/p[2]);
        projections.push_back(p2d);
        
        if(p2d.x < lt.x) lt.x = p2d.x;
        if(p2d.x > rb.x) rb.x = p2d.x;
        if(p2d.y < lt.y) lt.y = p2d.y;
        if(p2d.y > rb.y) rb.y = p2d.y;
    }
    
    boundingRect.x = lt.x;
    boundingRect.y = lt.y;
    boundingRect.width = rb.x - lt.x;
    boundingRect.height = rb.y - lt.y;
}

Mat RenderingEngine::downloadFrame(RenderingEngine::FrameType type)
{
    Mat res;
    switch (type)
    {
        case MASK:
            res = Mat(height, width, CV_8UC1);
            glReadPixels(0, 0, res.cols, res.rows, GL_RED, GL_UNSIGNED_BYTE, res.data);
            break;
        case RGB:
            res = Mat(height, width, CV_8UC3);
            glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_UNSIGNED_BYTE, res.data);
            break;
        case RGB_32F:
            res = Mat(height, width, CV_32FC3);
            glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_FLOAT, res.data);
            break;
        case DEPTH:
            res = Mat(height, width, CV_32FC1);
            glReadPixels(0, 0, res.cols, res.rows, GL_DEPTH_COMPONENT, GL_FLOAT,  res.data);
            break;
        default:
            res = Mat::zeros(height, width, CV_8UC1);
            break;
    }
    return res;
}
