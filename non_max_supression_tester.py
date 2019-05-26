import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from non_max_supression import nms

class nms_tester:
	def __init__(self):
		pass
	
	def jitter(self, bb, curr_idx, jitter_amount=3, iou_threshold=0.5):
		"""Jitter a box to simulate noisy bounding boxes output by a detector
	
		Using a uniform distribution, we take a given bb and jitter it to generate
		new and noisy bounding boxes with a pretty high final IoU
	
		Parameters:
		-bb (array): bounding bo to jitter
		-curr_idx (int): the current index in the final generated bb array to output the ref data
		-jitter_amount (int, default=3): The amount of jittering, higher means less chance of high IoU
										 and more variance in the jittered bb coordinates
		-iou_threshold (int, default=0.5): 


		Returns:
		-jittered_bbs (np.array): the final jittered bounding boxes
		-idxs_y (np.array): the reference data after nms

		"""
		
		# This array will contain all the jittered bb and the initial one
		jittered_bbs = [bb]
		# Randomly decide how many times we want to jitter the bb
		nb_jitter = np.random.randint(0, 10)
		
		# Array used to store the reference data
		idxs_y = [curr_idx]
		for i in range(nb_jitter):
			# We generate a jittered bb
			tmp = np.concatenate((bb[:4] + np.random.randint(0, jitter_amount, (4)), [np.random.uniform(0.0, 0.9)]))
			# IoU < iou_threshold we add it to the ref array because it won't be removed later by nms
			if nms.get_iou(bb, tmp)	< iou_threshold:
				idxs_y.append(i + curr_idx + 1)
			jittered_bbs.append(tmp)

		# Return the final jittered bb and the reference data
		return np.array(jittered_bbs), np.array(idxs_y)
	
	def gen_data(self, nb_bb, max_coord=100, max_size=40):
		"""Generate a given number of jittered bounding boxxes to simulate noisy data output by a detector
	
		Parameters:
		-max_coord (int, default=100): maximum (x, y) a bb can have
		-max_size (int, default=40): maximum (w, h) a bb can have

		Returns:
		-final_res (np.array): the final jittered bounding boxes
		-final_res[idxs_y] (np.array): the reference data after nms

		"""

		# Generate all the sample bounding box coordinates, sizes and percentages
		coords = np.array([np.random.randint(0, max_coord, (nb_bb)) for i in range(2)])
		size = np.array([np.random.randint(0, max_size, (nb_bb)) for i in range(2)])
		pc = np.ones((nb_bb))
		
		# Fill an array with the correct format:
		# [top_left, top_right, bottom_left, bottom_right, percentage]
		res = np.zeros((nb_bb, 5))
		res[:,0:2] = coords.T
		res[:,2:4] = coords.T + size.T
		res[:,4] = pc
		
		# Jitter and store the bounding boxes
		final_res, idxs_y = self.jitter(res[0], 0)
		for b in res[1:]:
			tmp_res, tmp_idx_y = self.jitter(b, len(final_res))
			final_res = np.concatenate((final_res, tmp_res))
			idxs_y = np.concatenate((idxs_y, tmp_idx_y))

		# Return the final jittered arrays and the reference data
		return final_res, final_res[idxs_y]
	
	def disp_results(self, data, data_final, data_y):
		"""Display the result to help compare input, reference and result data
	
		Parameters:
		-data (array): Input orgininal data
		-data_final (array): Reference data
		-data_y (array): Implementation result data

		Returns:
		-Nothing

		"""

		# Compute the figure size
		max_x = np.amax(data[:, 2] + data[:, 0])
		max_y = np.amax(data[:, 3] + data[:, 1])
		a = np.zeros((int(max_x) + 5, int(max_y) + 5))
		
		# Create 3 plots
		fig, ax = plt.subplots(3, figsize=(max_x, max_y))
		
		# Draw bounding boxes for the original data
		for i in range(data.shape[0]):
			rect = patches.Rectangle((data[i][0], data[i][1]), data[i][2] - data[i][0], data[i][3] - data[i][1], linewidth=1, edgecolor='g', facecolor='none')
			# Add the patch to the Axes
			ax[0].add_patch(rect)
		
		# Draw bounding boxes for the ref data
		for i in range(data_y.shape[0]):
			rect2 = patches.Rectangle((data_y[i][0], data_y[i][1]), data_y[i][2] - data_y[i][0], data_y[i][3] - data_y[i][1], linewidth=1, edgecolor='g', facecolor='none')
			ax[1].add_patch(rect2)
		
		# Draw bounding boxes for my implementation's result
		for i in range(data_final.shape[0]):
			rect3 = patches.Rectangle((data_final[i][0], data_final[i][1]), data_final[i][2], data_final[i][3], linewidth=1, edgecolor='g', facecolor='none')
			ax[2].add_patch(rect3)
			
		# Display the image
		ax[0].set_title("Input data")
		ax[0].imshow(a)
		ax[1].set_title("Helper/ref output (not 100% accurate)")
		ax[1].imshow(a)
		ax[2].set_title("My implementation")
		ax[2].imshow(a)
		plt.show()
		
if __name__ == "__main__":
	#Handle not enough cmd line args
	if len(sys.argv) < 2:
		sys.exit("Usage: python non_max_supression_tester.py number_of_bounding_boxes_to_generate")
	
	# Call all the helper functions
	nms = nms()
	nms_tester = nms_tester()
	data, data_y = nms_tester.gen_data(int(sys.argv[1]))
	nms.data = data
	#nms.get_data(sys.argv[1])
	start = time.perf_counter()
	nms.compute_nms()
	end = time.perf_counter()
	print("Generated and jittered", len(data), "boxes.. processing took", end - start, "s")
	nms_tester.disp_results(nms.data, nms.final_res, data_y)