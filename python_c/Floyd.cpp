//
// Created by Antoniano on 2022/4/11.
//

#include "Floyd.h"
#include <iostream>
#include <string>
#include <cmath>

using namespace std;




extern "C"{

    double get_dist(double x1,double y1, double x2, double y2){
        double dist1 = x2 - x1;
        double dist2 = y2 - y1;
        return sqrt(pow(dist1, 2) + pow(dist2, 2));
    }
    void FloydAlgorithm(double * graph,int batch,int node_num, double* dist){
    
    #pragma omp parallel for 
        for(int i=0;i<batch;i++)
        {
            for(int j=0;j<node_num;j++)
            {
                for(int k = 0;k<node_num;k++){
                    double x1 = graph[i*(node_num * 2) + j * 2];
                    double y1 = graph[i*(node_num * 2) + j * 2 + 1];
                    double x2 = graph[i*(node_num * 2) + k * 2];
                    double y2 = graph[i*(node_num * 2) + k * 2 + 1];
                    // printf("x1 %f y1 %f x2 %f y2 %f %d %d %d\n", x1, y1, x2, y2, i, j, k);
                    int pointer1 = i*(node_num * node_num) + j * node_num + k;
                    double distance = get_dist(x1, y1, x2, y2);
                    dist[pointer1] = distance;
                }
            }
        }

    }
}
