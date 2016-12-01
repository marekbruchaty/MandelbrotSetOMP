#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/freeglut.h>
  #include <GL/freeglut_ext.h>
#endif

// Glut is deprecated, and warnings are annoying
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define width 1280
#define height 720

#define END_TAG    0
#define DATA_TAG    1
#define RESULT_TAG  2
#define VAR_INIT_TAG 3

/*Glob variables**********************************************************************************/

typedef struct {
    GLbyte r,g,b;
} Pixel;

typedef struct {
    double real,imag;
} Complex;

GLuint texture;

// variables
double  real_min    =   -2.9,
        real_max    =    1.4,
        img_min     =   -1.2,
        img_max     = img_min+(real_max-real_min)*height/width,
        real_factor = (real_max-real_min)/(width-1),
        imag_factor = (img_max-img_min)/(height-1);
int     iterations  = 10, thread_count = 2, step = 1, color_profile = 1, window_id;

// framebuffer
Pixel image[height][width];

// color mapping
Pixel mapping[16];

// Proc. info
int nproc, rank;

// Debug mode
int debug = 1, omp_enabled = 0;

typedef struct {
    int iterations, color_profile, omp_enabled, thread_count;
} s_variables;

/*Prototypes**********************************************************************************/

void master();
void slave();
void init(int argc, char ** argv);
void init_mpi(int argc, char ** argv);
void display();
void render();
void keypress(unsigned char key, int x, int y);
void special_keypress(int key, int x, int y);
void calc_mandelbrot(unsigned row, unsigned *row_data);
Pixel calc_color(unsigned n);
void init_color_mapping();

/*Main**********************************************************************************/

int main(int argc, char ** argv) {
    init_mpi(argc, argv);

    // custom struct. for variable updating
    const int nitems = 4;
    int blocklengths[nitems] = {1,1,1,1};
    MPI_Datatype types[nitems] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_s_variables;
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(s_variables, iterations);
    offsets[1] = offsetof(s_variables, color_profile);
    offsets[2] = offsetof(s_variables, omp_enabled);
    offsets[3] = offsetof(s_variables, thread_count);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_s_variables);
    MPI_Type_commit(&mpi_s_variables);


    if (rank==0) {
        s_variables send;
        send.iterations = iterations;
        send.color_profile = color_profile;
        send.omp_enabled = omp_enabled;
        send.thread_count = thread_count;

        if (debug) printf("(i) Broadcasting initial variables\n");
        for (int i = 1; i < nproc; ++i) {
            MPI_Send(&send, 1, mpi_s_variables, i, VAR_INIT_TAG, MPI_COMM_WORLD);
        }

        init(argc, argv);
        master();
    }
    else {

        s_variables recv;

        MPI_Status status;
        MPI_Recv(&recv, 1, mpi_s_variables, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        iterations = recv.iterations;
        color_profile = recv.color_profile;
        omp_enabled = recv.omp_enabled;
        thread_count = recv.thread_count;

        slave();
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

/*Functions**********************************************************************************/

void work(int argc, char ** argv) {
        init_mpi(argc, argv);

    if (!rank) {
        init(argc, argv);
        master();
    }
    else {
        slave();
    }

    MPI_Finalize();
}

void master() {

    MPI_Status mpi_status;

    timeval t_start, t_end;
    double t_start_omp, t_end_omp;

    int row = 0;
    int row_data[width+1];

    if (debug) printf("(i) Time measurament starts\n");
    if (omp_enabled) t_start_omp = omp_get_wtime();
    else gettimeofday(&t_start, NULL);

    // Send row to all nodes, not to master 0
    if (debug) printf("(i) Sending initial rows to nodes\n");
    for (int i = 1; i < nproc; ++i) {
        MPI_Send(&row, 1, MPI_INT, i, DATA_TAG, MPI_COMM_WORLD);
        row++;
    }

    int complete_rows = 0;
    if (debug) printf("(i) While loop - assigning jobs\n");
    while (complete_rows < height) {
        MPI_Recv(&row_data, width+1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &mpi_status);

        int slave_done = mpi_status.MPI_SOURCE;
        int received_row = row_data[0]; // Node without job

        for (int column = 0; column < width; ++column) {
            image[received_row][column] = calc_color(row_data[column+1]);
        }

        if (debug) printf("(i) Done row #%d\n", received_row);
        complete_rows++;
        if (row < height) {
            MPI_Send(&row, 1, MPI_INT, slave_done, DATA_TAG, MPI_COMM_WORLD);
            row++;
        }
    }

    if (debug) printf("(i) Sending termination tags to slaves %d\n");
    for (int i = 1; i < nproc; ++i) {
        MPI_Send(0, 0, MPI_INT, i, END_TAG, MPI_COMM_WORLD);
    }

    if (debug) printf("(i) Time measurament ends\n");
    if (omp_enabled) {
        t_end_omp = omp_get_wtime();
        printf("(DONE) Total computation time %f seconds\n", t_end_omp - t_start_omp);
    }
    else {
        gettimeofday(&t_end, NULL);
        float total_time = ((t_end.tv_sec - t_start.tv_sec) * 1000000u + t_end.tv_usec - t_start.tv_usec) / 1.e6;
        printf("(DONE) Total computation time %.06lf seconds\n", total_time);

    }

    // Run the control loop
    glutSpecialFunc(special_keypress);
    glutKeyboardFunc(keypress);
    glutDisplayFunc(display);
    //display();
    if (debug) printf("(i) Running GLUT control loop\n"); 
    glutMainLoop();

}

void slave() {

    MPI_Status mpi_status;

    unsigned row = 0;
    unsigned row_data[width+1];

    int slave_rank;
    // Get rank ID
    MPI_Comm_rank(MPI_COMM_WORLD, &slave_rank);

    // Receive row num. from master
    MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_status);

    while (mpi_status.MPI_TAG == DATA_TAG) {

        if (mpi_status.MPI_TAG == END_TAG) exit(EXIT_SUCCESS);

        // Calculate whole row of data here
        calc_mandelbrot(row, row_data);

        // Send computed row data back to master
        MPI_Send(row_data, width+1, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
        // Get another job allocation
        MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_status);
    }

}

// Initialize OpenGL state
void init(int argc, char ** argv) {

    if (debug) printf("(i) Initializing GLUT\n"); 

    // Init GLUT
    glutInit(&argc, argv);
    glutInitWindowSize(width,height);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
    window_id = glutCreateWindow("Mandelbrot Set");

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

void init_mpi(int argc, char ** argv) {
    if (debug) printf("(i) Initializing MPI\n"); 

    // Mpi setup
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { 
        fprintf(stderr,"ERROR: MPI Initialization\n"); 
        exit(EXIT_FAILURE); 
    } 

    if (MPI_Comm_size(MPI_COMM_WORLD, &nproc) != MPI_SUCCESS) {
        fprintf(stderr,"ERROR: Coudn't get the process count\n");
    } 
    else if (nproc < 2) {
        fprintf(stderr, "ERROR: Number of processes must be at least 2\n"); 
        MPI_Finalize(); 
        exit(EXIT_FAILURE); 
    } 

    printf("Number of processors: %d\n", nproc); 

    if (MPI_Comm_rank(MPI_COMM_WORLD,&rank) != MPI_SUCCESS) { 
        fprintf(stderr,"ERROR: Couldn't get the rank of a process\n"); 
        MPI_Finalize(); 
        exit(EXIT_FAILURE); 
    }

}

Pixel calc_color(unsigned n) {

    double color_step = (double) iterations / 256;

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
        if (n == iterations || n == 0) {
            p.r = 0;
            p.g = 0;
            p.b = 0;
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
    return p;
}

void calc_mandelbrot(unsigned row, unsigned *row_data) {
    
    Complex c,z,z_sqr;
    double real_factor = (real_max-real_min)/(width-1);
    double imag_factor = (img_max-img_min)/(height-1);

    #pragma omp parallel for if(omp_enabled) private(c,z,z_sqr) num_threads(thread_count)
    for(unsigned x=0; x<width; ++x) {
        c.imag = img_max - row*imag_factor;
        c.real = real_min + x*real_factor;

        z.real = c.real, z.imag = c.imag;
        bool isInside = true;
        unsigned n;

        for(n=0; n<iterations; ++n) {
            z_sqr.real = z.real*z.real;
            z_sqr.imag = z.imag*z.imag;
            if(z_sqr.real + z_sqr.imag > 4) {
                isInside = false;
                break;
            }
            z.imag = 2*z.real*z.imag + c.imag;
            z.real = z_sqr.real - z_sqr.imag + c.real;
        }

        if(isInside) {
            row_data[x+1] = iterations;
        } else {
            row_data[x+1] = n;
        }
    }
    row_data[0] = row;
    //printf("(i) Render done\n");
}

void render_frame() {


    Complex c,z,z_sqr;

    double real_factor = (real_max-real_min)/(width-1);
    double imag_factor = (img_max-img_min)/(height-1);

    double color_step = (double) iterations / 256;

    #pragma omp parallel for num_threads(thread_count) private(c,z,z_sqr)
    for(unsigned y=0; y<height; ++y) {
        c.imag = img_max - y*imag_factor;

        for(unsigned x=0; x<width; ++x) {
            c.real = real_min + x*real_factor;

            z.real = c.real, z.imag = c.imag;
            bool isInside = true;
            unsigned n;

            for(n=0; n<iterations; ++n) {
                Complex z_sqr;
                z_sqr.real = z.real*z.real;
                z_sqr.imag = z.imag*z.imag;
                if(z_sqr.real + z_sqr.imag > 4) {
                    isInside = false;
                    break;
                }
                z.imag = 2*z.real*z.imag + c.imag;
                z.real = z_sqr.real - z_sqr.imag + c.real;
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
    printf("(i) Render done\n");
}

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

// Generate and display the image.
void display() {

    // Call user image generation
    //render_frame();
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
    glutPostRedisplay();
    glutSwapBuffers();

}

// handle basic keys (ESC is num. 27 in ASCII)
void keypress(unsigned char key, int x, int y) {

    double real_diff = fabs(real_min - real_max) * 0.05;
    double img_diff = fabs(img_min - img_max) * 0.05;

    switch(key) {
        case 'S':
        case 's':
            real_min -= real_diff;
            real_max += real_diff;
            img_min -= img_diff;
            img_max += img_diff;
            break;
        case 'W':
        case 'w':
            real_min += real_diff;
            real_max -= real_diff;
            img_min += img_diff;
            img_max -= img_diff;
            break;
        case 'A':
        case 'a':
            if(iterations > step) iterations -= step;
            printf("Iterations:\t%d\n",iterations);
            break;
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
        case 'L':
        case 'l':
            thread_count += 1;
            printf("Threads:\t%d\n",thread_count);
            break;
        case 'K':
        case 'k':
            if (thread_count > 1) thread_count -= 1;
            printf("Threads:\t%d\n",thread_count);
            break;
        case 'C':
        case 'c':
            color_profile++;
            if (color_profile > 5) color_profile = 1;
            printf("Color profile changed\n");
            break;
        case 27:
            glutDestroyWindow(window_id);
            exit(EXIT_SUCCESS);
            break;
        default:
            break;
    }

    glutPostRedisplay();
}

// handle arrow keys
void special_keypress(int key, int x, int y) {

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

    glutPostRedisplay();
}