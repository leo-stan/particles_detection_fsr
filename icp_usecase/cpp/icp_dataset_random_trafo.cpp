#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc
//#include <bits/stdc++.h> 

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void
print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}

std::stringstream
print4x4MatrixWrite (const Eigen::Matrix4d & matrix)
{
  std::stringstream ss;
  ss << "Rotation matrix :\n";
  ss << "    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2);
  ss << "R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2);
  ss << "    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2);
  ss << "Translation vector :\n";
  ss << "t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3);
  return ss;
}

int main (int argc, char* argv[])
{
	// Parallelizing this computation
	int pid1 = fork();
    int pid2 = fork();
    int pid3 = fork();
    int ctr_1 = 0; 
    int ctr_2 = 0; 
    int ctr_3 = 0;
    if (pid1 > 0) ctr_1 = 1;
    if (pid2 > 0) ctr_2 = 1;
    if (pid3 > 0) ctr_3 = 1; 
    int pid_ctr = ctr_1*4+ctr_2*2+ctr_3;
    srand(pid_ctr+1); // Initialize random number generator
	 // The point clouds we will be using
	PointCloudT::Ptr cloud_target (new PointCloudT);  // Original point cloud
	PointCloudT::Ptr cloud_source_original (new PointCloudT);  // Transformed point cloud with obscurance in it
	PointCloudT::Ptr cloud_source_original_backup (new PointCloudT);
	PointCloudT::Ptr cloud_source_filtered (new PointCloudT);  // Transformed point cloud which is cleaned
	PointCloudT::Ptr cloud_source_filtered_backup (new PointCloudT);

	pcl::console::TicToc time;

  	int iterations = 50;  // Default number of ICP iterations
  	int number_scans[] = {30,54,55,66,66,64,42,74};
  	int record_array[] = {1,5,6,10,13,15,16,18};
  	// By coincidence one operation per process
    for(int record_ctr = pid_ctr; record_ctr < pid_ctr+1; ++record_ctr) 
    {
    	int record = record_array[record_ctr];
    	std::stringstream record_char;
    	record_char << record;
    	std::string folder = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/julian/" + record_char.str();
    	if (pcl::io::loadPCDFile<pcl::PointXYZ> (folder + "/target.pcd", *cloud_target) == -1) //* load the file
			{
				PCL_ERROR ("Couldn't read file target.pcd \n");
			  	return (-1);
			}
			
    	for(int scan = 0; scan < number_scans[record_ctr]; ++scan)
    	{
    		std::cout << "Record: " << record << "    Scan: " << scan << "\n"; 
			time.tic ();
			std::stringstream scan_char;
			scan_char << scan;
			if (pcl::io::loadPCDFile<pcl::PointXYZ> (folder + "/original/" + scan_char.str() + ".pcd", *cloud_source_original) == -1) //* load the file
			{
			  	PCL_ERROR ("Couldn't read file source_original.pcd \n");
			  	return (-1);
			}
			if (pcl::io::loadPCDFile<pcl::PointXYZ> (folder + "/filtered/" + scan_char.str() + ".pcd", *cloud_source_filtered) == -1) //* load the file
			{
			  	PCL_ERROR ("Couldn't read file source_filtered.pcd \n");
			  	return (-1);
			}

			//std::cout << "\nLoaded files in " << time.toc () << " ms\n" << std::endl;

			// Defining a rotation matrix and translation vector
			Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();

			// A rotation matrix 
			double theta_mean = M_PI / 8;  // The angle of rotation in radians
			// Bring in randomness
			double theta = 2 * theta_mean * (((double) rand()) / (double) RAND_MAX);

			// Trafo matrix
			transformation_matrix (0, 0) = cos (theta);
			transformation_matrix (0, 1) = -sin (theta);
			transformation_matrix (1, 0) = sin (theta);
			transformation_matrix (1, 1) = cos (theta);

			// A translation on Z axis (0.4 meters)
			transformation_matrix (2, 3) = 2 * 0.4 * (((double) rand()) / (double) RAND_MAX);
			// A translation on x axis (0.6 meters)
			transformation_matrix (0, 3) = 2 * 0.6 * (((double) rand()) / (double) RAND_MAX);

			// Executing the transformation
			*cloud_source_original_backup = *cloud_source_original;  // We backup cloud_icp into cloud_tr for later use
			*cloud_source_filtered_backup = *cloud_source_filtered;
			pcl::transformPointCloud (*cloud_source_original_backup, *cloud_source_original, transformation_matrix);
			pcl::transformPointCloud (*cloud_source_filtered_backup, *cloud_source_filtered, transformation_matrix);

			// The Iterative Closest Point algorithm
			// Original Cloud --------------------------------------------------------------
			//std::cout << "Original Cloud" << std::endl;
			time.tic ();
			pcl::IterativeClosestPoint<PointT, PointT> icp_1;
			icp_1.setMaximumIterations (1);
			double y_1[iterations];
			Eigen::Matrix4d transformation_matrices_1[iterations];

			for (int i = 0; i < iterations; i++)
			{
				icp_1.setInputSource (cloud_source_original);
			    icp_1.setInputTarget (cloud_target);
			    icp_1.align (*cloud_source_original);
			    y_1[i] = icp_1.getFitnessScore ();
			    if (i == 0)
			      	transformation_matrices_1[i] = icp_1.getFinalTransformation ().cast<double>() * transformation_matrix;
			    else
			      	transformation_matrices_1[i] = icp_1.getFinalTransformation ().cast<double>() * transformation_matrices_1[i-1];
			}
			  
			//std::cout << "Applied " << iterations << " ICP iteration(s) to original cloud in " << time.toc () << " ms" << std::endl;

			// Filtered Cloud ---------------------------------------------------------------
			//std::cout << "Cleaned Cloud" << std::endl;
			time.tic ();
			pcl::IterativeClosestPoint<PointT, PointT> icp_2;
			icp_2.setMaximumIterations (1);
			double y_2[iterations];
			Eigen::Matrix4d transformation_matrices_2[iterations];
			for (int i = 0; i < iterations; i++)
			{
			  	icp_2.setInputSource (cloud_source_filtered);
			    icp_2.setInputTarget (cloud_target);
			    icp_2.align (*cloud_source_filtered);
			    y_2[i] = icp_2.getFitnessScore ();
			    if (i == 0)
				    transformation_matrices_2[i] = icp_2.getFinalTransformation ().cast<double>() * transformation_matrix;
			    else
			    	transformation_matrices_2[i] = icp_2.getFinalTransformation ().cast<double>() * transformation_matrices_2[i-1];
			}
			 
			//std::cout << "Applied " << iterations << " ICP iteration(s) to filtered cloud in " << time.toc () << " ms" << std::endl;
			  
			std::ofstream file;
			file.open (folder + "/txt_files/" + scan_char.str() + "_scores.txt");
			for (int i = 0; i < iterations; i++)
			{
			  	file << y_1[i] << " ";
			}
			file << "\n ";
			for (int i = 0; i < iterations; i++)
			{
				file << y_2[i] << " ";
			}
			file << "\n ";
			file.close();
			//std::cout << "Obscured Pointcloud \n";
			file.open (folder + "/txt_files/" + scan_char.str() + "_trafo_original.txt");
			// Add Transformation from beginning
			for (int j = 0; j < 16; j++)
				file << transformation_matrix(j/4,j%4) << " ";
			// Add Transformations from iteration steps
			for (int i = 0; i < iterations; i++)
			{
				// print4x4Matrix(transformation_matrices_1[i]);
				for (int j = 0; j < 16; j++)
			  	    file << transformation_matrices_1[i](j/4,j%4) << " ";
			}
			file.close();
			//std::cout << "Cleaned Pointcloud \n";
			file.open (folder + "/txt_files/" + scan_char.str() + "_trafo_filtered.txt");
			// Add Transformation from beginning
			for (int j = 0; j < 16; j++)
			    file << transformation_matrix(j/4,j%4) << " ";
			  	// Add Transformations from iteration steps
			for (int i = 0; i < iterations; i++)
			{
				// print4x4Matrix(transformation_matrices_2[i]);
			  	for (int j = 0; j < 16; j++)
			  	    file << transformation_matrices_2[i](j/4,j%4) << " ";
			}
			file.close();
		}
  	}  
  	return (0);
}