// Created by Marek Bruchaty @ github.com/marekbruchaty
// marekbruchaty@gmail.com

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/freeglut.h>
  #include <GL/freeglut_ext.h>
#endif

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

typedef struct {
    GLbyte r,g,b;
} Pixel;

typedef struct {
    GLdouble real,imag;
} Complex;

#define width 1280
#define height 720

Pixel image[height][width];

GLuint texture;
Pixel dark_blue;

int window_id;

//Drawing area
double real_min = -2.9; //left border
double real_max = 1.4; //right border
double img_min = -1.2; //top border
double img_max = img_min+(real_max-real_min)*height/width; //bottom border

//Real. and imag. factors
double real_factor = (real_max-real_min)/(width-1);
double imag_factor = (img_max-img_min)/(height-1);

int iterations = 0, thread_count = 1, step = 1, color_profile = 1;

//Color mapping
Pixel mapping[16];

void RenderFrame() {
    double real_factor = (real_max-real_min)/(width-1);
    double imag_factor = (img_max-img_min)/(height-1);

    double color_step = (double) iterations / 256;

    double start_time = omp_get_wtime();

    #pragma omp parallel for num_threads(thread_count)
    for(unsigned y=0; y<height; ++y) {
        double c_im = img_max - y*imag_factor;

        for(unsigned x=0; x<width; ++x) {
            double c_re = real_min + x*real_factor;

            double Z_re = c_re, Z_im = c_im;
            bool isInside = true;
            unsigned n;

            for(n=0; n<iterations; ++n) {
                double Z_re2 = Z_re*Z_re, Z_im2 = Z_im*Z_im;
                if(Z_re2 + Z_im2 > 4) {
                    isInside = false;
                    break;
                }
                Z_im = 2*Z_re*Z_im + c_im;
                Z_re = Z_re2 - Z_im2 + c_re;
            }

            if(isInside) {
                Pixel p;
                p.r = 0;
                p.g = 0;
                p.b = 0;
                image[y][x] = p;
            } else {
                unsigned dn = n;
                double color = (double) dn / color_step;
                int nc = (int) color;

                Pixel p;
                switch (color_profile) {
                    case 1:
                        if (n < iterations && n > 0) {
                            int id = n%16;
                            p = mapping[id];
                        }
                        break;
                    case 2:
                        p.r = 0;
                        p.g = nc;
                        p.b = 0;
                        break;
                    case 3:
                        p.r = 0;
                        p.g = 0;
                        p.b = nc;
                        break;
                    case 4:
                        p.r = nc;
                        p.g = 0;
                        p.b = nc;
                        break;
                    case 5:
                        p.r = nc;
                        p.g = 0;
                        p.b = 0;
                        break;
                    default:
                        break;
                }
                image[y][x] = p;
            }
        }
    }
    double end_time = omp_get_wtime();
    printf("Render time: %f\n", end_time-start_time);
}

//Color mapping. Predefined colors used for diff. levels
void init_color_mapping() {
    mapping[0].r = 66; mapping[0].g = 30; mapping[0].b = 15;
    mapping[1].r = 25; mapping[1].g = 7; mapping[1].b = 26;
    mapping[2].r = 9; mapping[2].g = 1; mapping[2].b = 47;
    mapping[3].r = 4; mapping[3].g = 4; mapping[3].b = 73;
    mapping[4].r = 0; mapping[4].g = 7; mapping[4].b = 100;
    mapping[5].r = 12; mapping[5].g = 44; mapping[5].b = 138;
    mapping[6].r = 24; mapping[6].g = 82; mapping[6].b = 177;
    mapping[7].r = 57; mapping[7].g = 125; mapping[7].b = 209;
    mapping[8].r = 134; mapping[8].g = 181; mapping[8].b = 229;
    mapping[9].r = 211; mapping[9].g = 236; mapping[9].b = 248;
    mapping[10].r = 241; mapping[10].g = 233; mapping[10].b = 191;
    mapping[11].r = 248; mapping[11].g = 201; mapping[11].b = 95;
    mapping[12].r = 255; mapping[12].g = 170; mapping[12].b = 0;
    mapping[13].r = 204; mapping[13].g = 128; mapping[13].b = 0;
    mapping[13].r = 153; mapping[13].g = 87; mapping[13].b = 0;
    mapping[13].r = 106; mapping[13].g = 52; mapping[13].b = 3;
}

// Initialize OpenGL state
void init() {

	  // Texture setup
    glEnable(GL_TEXTURE_2D);
    glGenTextures( 1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // Other
    glClearColor(0,0,0,0);
    gluOrtho2D(-1,1,-1,1);
    glLoadIdentity();
    glColor3f(1,1,1);
    init_color_mapping();
}

// Generate and display the image.
void display() {

    // Call user image generation
    RenderFrame();

    // Copy image to texture memory
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

    // Clear screen buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Render a quad
    glBegin(GL_QUADS);
        glTexCoord2f(1,0); glVertex2f(1,-1);
        glTexCoord2f(1,1); glVertex2f(1,1);
        glTexCoord2f(0,1); glVertex2f(-1,1);
        glTexCoord2f(0,0); glVertex2f(-1,-1);
    glEnd();

    // Display result
    glFlush();
    glutSwapBuffers();

}

//Keyboard interaction
void keypress(unsigned char key, int x, int y) {

    double real_diff = fabs(real_min - real_max) * 0.05;
    double img_diff = fabs(img_min - img_max) * 0.05;

    switch(key) {
        //Zoom in
        case 'S':
        case 's':
            real_min -= real_diff;
            real_max += real_diff;
            img_min -= img_diff;
            img_max += img_diff;
            break;
        //Zoom out
        case 'W':
        case 'w':
            real_min += real_diff;
            real_max -= real_diff;
            img_min += img_diff;
            img_max -= img_diff;
            break;
        //Decrement num. of iterations
        case 'A':
        case 'a':
            if(iterations > step) iterations -= step;
            printf("Iterations:\t%d\n",iterations);
            break;
        //Increment num. of iterations
        case 'D':
        case 'd':
            iterations += step;
            printf("Iterations:\t%d\n",iterations);
            break;
        case 'E':
        case 'e':
            step++;
            printf("Step:\t%d\n", step);
            break;
        case 'Q':
        case 'q':
            if (step > 1) step--;
            printf("Step:\t%d\n", step);
            break;
        //Increment num. of used threads
        case 'L':
        case 'l':
            thread_count += 1;
            printf("Threads:\t%d\n",thread_count);
            break;
        //Decrement num. of used threads
        case 'K':
        case 'k':
            if (thread_count > 1) thread_count -= 1;
            printf("Threads:\t%d\n",thread_count);
            break;
        //Change color
        case 'C':
        case 'c':
            color_profile++;
            if (color_profile > 5) color_profile = 1;
            printf("Color profile changed\n");
            break;
        //Exit app
        case 27:
            glutDestroyWindow(window_id);
            exit(EXIT_SUCCESS);
            break;
        default:
            break;
    }

    glutPostRedisplay();
}

void SpecialKeys(int key, int x, int y) {

    double realDif = fabs(real_min - real_max) * 0.05;
    double imagDif = fabs(img_min - img_max) * 0.05;

    switch (key) {
        case (char)GLUT_KEY_UP:
            img_min -= imagDif;
            img_max -= imagDif;
            break;
        case GLUT_KEY_DOWN:
            img_min += imagDif;
            img_max += imagDif;
            break;
        case GLUT_KEY_RIGHT:
            real_min += realDif;
            real_max += realDif;
            break;
        case GLUT_KEY_LEFT:
            real_min -= realDif;
            real_max -= realDif;
            break;
        default:
            break;
    }
    // Refresh the Window
    glutPostRedisplay();
}

// Main entry function
int main(int argc, char ** argv) {

    // Init GLUT
    glutInit(&argc, argv);
    glutInitWindowSize(width,height);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
    window_id = glutCreateWindow("Mandelbrot Set");

    // Run the control loop
    glutSpecialFunc(SpecialKeys);
    glutKeyboardFunc(keypress);
    glutDisplayFunc(display);

    init();

    //Main exec. loop
    glutMainLoop();

    return EXIT_SUCCESS;
}
