import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

# For following
x = range(51)

# Now plot rotation errors

number_scans = [30,54,55,66,66,64,42,74]

# General
x_scores = range(50)
x_deviation = range(51)
# Scores
y_julian_scores = np.zeros((1,50))
y_leo_scores = np.zeros((1,50))
y_original_scores = np.zeros((1,50))
y_ideal_scores = np.zeros((1,50))

# Original
original_angles = np.zeros((1,51))
original_translations = np.zeros((1,51))
# Filtered
ideal_angles = np.zeros((1,51))
ideal_translations = np.zeros((1,51))
julian_angles = np.zeros((1,51))
julian_translations = np.zeros((1,51))
leo_angles = np.zeros((1,51))
leo_translations = np.zeros((1,51))

for ctr,record in enumerate([1,5,6,10,13,15,16,18]):
       folder_julian = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/julian/" + str(record) + "/txt_files/"
       folder_leo = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/leo/" + str(record) + "/txt_files/"
       folder_ideal = "/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/general/" + str(record) + "/txt_files/"

       for scan in range(0,number_scans[ctr]):
              # Scores ----------------------------------------------
              # 1) Julian:
              scores_file_julian = open(folder_julian + str(scan) + "_scores.txt", "r")
              scores_lines_julian = scores_file_julian.read().split(' ')
              scores_file_julian.close()
              # Concatenate all scores vectors
              y2_temp = np.zeros((1, 50))
              y2_temp[0, :] = np.asarray(scores_lines_julian[51:101], dtype=float)
              y_julian_scores = np.concatenate((y_julian_scores, y2_temp), axis = 0)
              # 2) Leo:
              scores_file_leo = open(folder_leo + str(scan) + "_scores.txt", "r")
              scores_lines_leo = scores_file_leo.read().split(' ')
              scores_file_leo.close()
              # Concatenate all scores vectors
              y2_temp = np.zeros((1, 50))
              y2_temp[0, :] = np.asarray(scores_lines_leo[51:101], dtype=float)
              y_leo_scores = np.concatenate((y_leo_scores, y2_temp), axis=0)
              # 3) Ideal and original:
              scores_file_ideal = open(folder_ideal + str(scan) + "_scores.txt", "r")
              scores_lines_ideal = scores_file_ideal.read().split(' ')
              scores_file_ideal.close()
              # Concatenate all scores vectors
              y1_temp = np.zeros((1, 50))
              y1_temp[0, :] = np.asarray(scores_lines_ideal[:50], dtype=float)
              y2_temp = np.zeros((1, 50))
              y2_temp[0, :] = np.asarray(scores_lines_ideal[51:101], dtype=float)
              y_original_scores = np.concatenate((y_original_scores, y1_temp), axis=0)
              y_ideal_scores = np.concatenate((y_ideal_scores, y2_temp), axis=0)

              # Original Trafo ---------------------------------------
              original_trafo_file = open(folder_ideal + str(scan) + "_trafo_original.txt", "r")
              original_trafo_lines = original_trafo_file.read().split(' ')
              original_trafo_file.close()
              # Rotations
              temp_rotations = np.zeros([51,3,3])
              for i in range (51):
                     for j in range(16):
                            if j/4 == 3 or j%4 == 3: # Outside of rotation part of transformation matrix
                                   continue
                            temp_rotations[i,j/4,j%4] = float(original_trafo_lines[i*16+j])
              temp_angles = np.zeros((1,51))
              for i in range(51):
                     diff = temp_rotations[i]
                     angle_temp,_ = cv2.Rodrigues(diff)
                     temp_angles[0,i] = np.linalg.norm(angle_temp)
              # Concatenate original angles vectors
              original_angles = np.concatenate((original_angles, temp_angles), axis = 0)
              # Translations
              temp_translations = np.zeros([51, 3])
              for i in range(51):
                     for j in range(16):
                            if j % 4 == 3 and j/3 < 4:  # Outside of rotation part of transformation matrix
                                   temp_translations[i, j / 4] = float(original_trafo_lines[i * 16 + j])
              temp_norm_translations = np.zeros((1,51))
              for i in range(51):
                     temp_norm_translations[0,i] = np.linalg.norm(temp_translations[i])
              # Concatenate original translation vectors
              original_translations = np.concatenate((original_translations, temp_norm_translations), axis = 0)
              del original_trafo_lines

              # Ideal Trafo ------------------------------------------------
              filtered_trafo_file = open(folder_ideal + str(scan) + "_trafo_filtered.txt", "r")
              filtered_trafo_lines = filtered_trafo_file.read().split(' ')
              filtered_trafo_file.close()
              # Rotations
              temp_rotations = np.zeros([51, 3, 3])
              for i in range(51):
                     for j in range(16):
                            if j / 4 == 3 or j % 4 == 3:  # Outside of rotation part of transformation matrix
                                   continue
                            temp_rotations[i, j / 4, j % 4] = float(filtered_trafo_lines[i * 16 + j])
              temp_angles = np.zeros((1,51))
              for i in range(51):
                     diff = temp_rotations[i]
                     angle_temp, _ = cv2.Rodrigues(diff)
                     temp_angles[0,i] = np.linalg.norm(angle_temp)
              # Concatenate original angles vectors
              ideal_angles = np.concatenate((ideal_angles, temp_angles), axis = 0)
              # Translations
              temp_translations = np.zeros([51, 3])
              for i in range(51):
                     for j in range(16):
                            if j % 4 == 3 and j / 3 < 4:  # Outside of rotation part of transformation matrix
                                   temp_translations[i, j / 4] = float(filtered_trafo_lines[i * 16 + j])
              temp_norm_translations = np.zeros((1,51))
              for i in range(51):
                     temp_norm_translations[0,i] = np.linalg.norm(temp_translations[i])
              # Concatenate original translation vectors
              ideal_translations = np.concatenate((ideal_translations, temp_norm_translations), axis = 0)

              # Julian Trafo ------------------------------------------------
              filtered_trafo_file = open(folder_julian + str(scan) + "_trafo_filtered.txt", "r")
              filtered_trafo_lines = filtered_trafo_file.read().split(' ')
              filtered_trafo_file.close()
              # Rotations
              temp_rotations = np.zeros([51, 3, 3])
              for i in range(51):
                     for j in range(16):
                            if j / 4 == 3 or j % 4 == 3:  # Outside of rotation part of transformation matrix
                                   continue
                            temp_rotations[i, j / 4, j % 4] = float(filtered_trafo_lines[i * 16 + j])
              temp_angles = np.zeros((1, 51))
              for i in range(51):
                     diff = temp_rotations[i]
                     angle_temp, _ = cv2.Rodrigues(diff)
                     temp_angles[0, i] = np.linalg.norm(angle_temp)
              # Concatenate original angles vectors
              julian_angles = np.concatenate((julian_angles, temp_angles), axis=0)
              # Translations
              temp_translations = np.zeros([51, 3])
              for i in range(51):
                     for j in range(16):
                            if j % 4 == 3 and j / 3 < 4:  # Outside of rotation part of transformation matrix
                                   temp_translations[i, j / 4] = float(filtered_trafo_lines[i * 16 + j])
              temp_norm_translations = np.zeros((1, 51))
              for i in range(51):
                     temp_norm_translations[0, i] = np.linalg.norm(temp_translations[i])
              # Concatenate original translation vectors
              julian_translations = np.concatenate((julian_translations, temp_norm_translations), axis=0)

              # Leo Trafo ------------------------------------------------
              filtered_trafo_file = open(folder_leo + str(scan) + "_trafo_filtered.txt", "r")
              filtered_trafo_lines = filtered_trafo_file.read().split(' ')
              filtered_trafo_file.close()
              # Rotations
              temp_rotations = np.zeros([51, 3, 3])
              for i in range(51):
                     for j in range(16):
                            if j / 4 == 3 or j % 4 == 3:  # Outside of rotation part of transformation matrix
                                   continue
                            temp_rotations[i, j / 4, j % 4] = float(filtered_trafo_lines[i * 16 + j])
              temp_angles = np.zeros((1, 51))
              for i in range(51):
                     diff = temp_rotations[i]
                     angle_temp, _ = cv2.Rodrigues(diff)
                     temp_angles[0, i] = np.linalg.norm(angle_temp)
              # Concatenate original angles vectors
              leo_angles = np.concatenate((leo_angles, temp_angles), axis=0)
              # Translations
              temp_translations = np.zeros([51, 3])
              for i in range(51):
                     for j in range(16):
                            if j % 4 == 3 and j / 3 < 4:  # Outside of rotation part of transformation matrix
                                   temp_translations[i, j / 4] = float(filtered_trafo_lines[i * 16 + j])
              temp_norm_translations = np.zeros((1, 51))
              for i in range(51):
                     temp_norm_translations[0, i] = np.linalg.norm(temp_translations[i])
              # Concatenate original translation vectors
              leo_translations = np.concatenate((leo_translations, temp_norm_translations), axis=0)

# Take away 1st dummy row
print y_julian_scores
y_julian_scores = y_julian_scores[1:]
y_leo_scores = y_leo_scores[1:]
y_original_scores = y_original_scores[1:]
y_ideal_scores = y_ideal_scores[1:]
original_angles = original_angles[1:]
original_translations = original_translations[1:]
ideal_angles = ideal_angles[1:]
ideal_translations = ideal_translations[1:]
julian_angles = julian_angles[1:]
julian_translations = julian_translations[1:]
leo_angles = leo_angles[1:]
leo_translations = leo_translations[1:]


# Add up all rows and compute mean
divider = len(original_angles)
print(divider)
y_julian_plot_scores = np.sum(y_julian_scores, axis = 0) / divider
y_leo_plot_scores = np.sum(y_leo_scores, axis = 0) / divider
y_original_plot_scores = np.sum(y_original_scores, axis = 0) / divider
y_ideal_plot_scores = np.sum(y_ideal_scores, axis = 0) / divider
original_plot_angles = np.sum(original_angles, axis = 0) / divider
original_plot_translations = np.sum(original_translations, axis = 0) / divider
ideal_plot_angles = np.sum(ideal_angles, axis = 0) / divider
ideal_plot_translations = np.sum(ideal_translations, axis = 0) / divider
julian_plot_angles = np.sum(julian_angles, axis = 0) / divider
julian_plot_translations = np.sum(julian_translations, axis = 0) / divider
leo_plot_angles = np.sum(leo_angles, axis = 0) / divider
leo_plot_translations = np.sum(leo_translations, axis = 0) / divider

# Define Figure
fig = plt.figure()
# Angle Plot
ax = plt.subplot()
ax.plot(x_deviation, 180.0/np.pi*original_plot_angles, 'g', label='Original')
ax.plot(x_deviation, 180.0/np.pi*leo_plot_angles, 'darkorange', label='Voxel Classification')
ax.plot(x_deviation, 180.0/np.pi*julian_plot_angles, 'r', label='Point Classification')
ax.plot(x_deviation, 180.0/np.pi*ideal_plot_angles, 'k.', label='Ground Truth')
ax.set(xlabel='Iteration', ylabel='Angular Deviation from Reference [deg]',
       title='Averaged Rotational Error of ICP')
legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
ax.grid()
fig.show()
fig.savefig('/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/angle_plot.jpg',dpi=300,
            bbox_inches = 'tight',pad_inches = 0)

# Define Figure
print(ideal_plot_translations)
print(julian_plot_translations)
fig = plt.figure()
# Translation Plot
ax = plt.subplot()
ax.plot(x_deviation, original_plot_translations, 'g', label='Original')
ax.plot(x_deviation, leo_plot_translations, 'darkorange', label='Voxel Classification')
ax.plot(x_deviation, julian_plot_translations, 'r', label='Point Classification')
ax.plot(x_deviation, ideal_plot_translations, 'k.', label='Ground Truth')
ax.set(xlabel='Iteration', ylabel='Translational Deviation from Reference [m]',
       title='Averaged Translational Error of ICP')
ax.set_ylim(top=0.8)
legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
ax.grid()
fig.show()
fig.savefig('/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/translation_plot.jpg',dpi=300,
            bbox_inches = 'tight',pad_inches = 0)

# Define Figure
fig = plt.figure()
# Scores Plot
ax = plt.subplot()
ax.plot(x_scores, y_original_plot_scores, 'g', label='Original')
ax.plot(x_scores, y_leo_plot_scores, 'darkorange', label='Voxel Classification')
ax.plot(x_scores, y_julian_plot_scores, 'r', label='Point Classification')
ax.plot(x_scores, y_ideal_plot_scores, 'k.', label='Ground Truth')
ax.set(xlabel='Iteration', ylabel='Score',
       title='Averaged Scores of ICP Algorithm')
legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
ax.grid()
fig.show()
fig.savefig('/media/juli/98F29C83F29C67722/SemesterProject/1_data/4_icp/score_plot.jpg',dpi=300,
            bbox_inches = 'tight',pad_inches = 0)

