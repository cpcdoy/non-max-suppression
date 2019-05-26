import numpy as np
import sys

class nms:
	"""Performs Non-Maximum Suppression (NMS)

	NMS is used to clean bounding box data that is output by an object detector
	
	"""
	
	def __init__(self):
		"""
		Constructor
		"""
		self.data = None
		self.final_res = None
	
	def get_data(self, path, delimiter=','):
		"""Parses bounding box data and processes it
		
		Reads a file where every line should be of the format:
		"top_left, top_right, width, height, score"
		
		And converts it to:
		"top_left, top_right, top_left + width, top_right + height, score"

		Parameters:
		-path (str): Path of the file to open.
		-delimiter (char, default=','): Used to set the delimiter by which to split 
										every line in the open file. Set by default to ',' for CSV.
		

		Returns:
		-Nothing
		-Sets self.data to the final corrected data

		"""
		# We store the final result in an array of arrays
		self.data = []
		# Open the file
		f = open(path, 'r')
		
		# For each line, we split by the delimiter and process the data
		for line in f:
			# Split the line by the delimiter
			l = line.split(delimiter)
			# Convert everything to a float
			l_clean = []
			for f, i in zip(l, range(len(l))):
				float_f = float(f)
				# Check for weird values
				# Coordinates can't be less than 0.0
				# Width, height can't be 0 or less
				# And 0.0 <= confidence <= 1.0
				if float_f < 0.0 or ((i == 2 or i == 3) and float_f <= 0) or (i == 4 and float_f > 1.0):
					continue
				l_clean.append(float_f)
			
			# We want to get the bottom right coordinate of the bounding box
			# So we just add "top_left + width" and "top_right + height"
			l_clean[2] += l_clean[0]
			l_clean[3] += l_clean[1]
			# Add the processed line to the array
			self.data.append(l_clean)
		
		# Convert to a Numpy array
		self.data = np.array(self.data)
		
	def get_iou(self, b1, b2):
		"""Computes IoU for 2 bounding boxes

		
	(x1b1,y1b1)###################
		       #                 #
		       #  (x1b2,y1b2)$$$$#$$$$$$$$$$$$$$$
		       #             $%%%#              $
		       #             $%%%#              $
		       ###################(x2b1,y2b1)   $
		       	             $                  $
		       	             $$$$$$$$$$$$$$$$$$$$(x2b2,y2b2)

		To get the intersection bounding box's coordinates and size, we do the following:
		
		Top left coordinate: (x1, y1) = (max(x1b1, x1b2), max(y1b1, y1b2))
		Bottom right coordinates: (x2, y2) = (min(x2b1, x2b2), min(y2b1, y2b2))
		
		width: w = x2 - x1
		height: h = y2 - y1
		
		The final total area (union) is :
		
		area of intersection = w * h
		area of union = area of b1 + area of b2 - 2 * area of intersection + area of intersection
					  = area of b1 + area of b2 - area of intersection
					  
		The final IoU is:
		
		IoU = area of intersection / area of union
		
		To avoid a divide by 0, we do add an epsilon:
		
		IoU = area of intersection / (area of union + epsilon)

		Parameters:
		-b1 (array): First bounding box
		-b2 (array): Second bounding box

		Returns:
		-iou (float): IoU between bounding boxes

		"""
		# Compute top left and bottom right coordinates of the intersection bb
		x1 = max(b1[0], b2[0])
		y1 = max(b1[1], b2[1])
		x2 = min(b1[2], b2[2])
		y2 = min(b1[3], b2[3])
	
		# Compute width and height based on the previous computation
		width = (x2 - x1)
		height = (y2 - y1)
		
		# This means that the two boxes don't intersect at all,
		# so it will give us negative width and height because:
		# x2 < x1 and y2 < y1, in this case
		if width < 0 or height < 0:
			return 0.0
		
		# Area of the intersection
		area_intersect = width * height
	
		# Compute the two initial bounding boxes' areas
		area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
		area_b2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
		
		# The total area of the union
		total_area = area_b1 + area_b2 - area_intersect
	
		""" 
		Avoid divide by 0 by not using a condition because
		we dont really care about size 0 bounding boxes.
		So we just add an epsilon=1e-5
		"""
		iou = area_intersect / (total_area + 1e-5)
		return iou
		
	def compute_nms(self, iou_threshold=0.5):
		"""Computes Non-Maximum Supression
		
		For this implementation, I sort the data by confidence in ascending order
		and also by the bottom right x (or y, it doesn't really matter) coordinate.
		I also create an array containing only indices, and this is the array I'm going
		to be working with and modify.
		
		This means that I start by taking the highest confidence bounding box, add it to
		the final array and get all the other boxes' IoU compared to this one (supposedly the best
		according to the object detector).
		
		After that I remove all the indices of the boxes that have an IoU > iou_threshold=0.5.
		
		The algorithm ends when all the indices have been reviewed.
	
		#################################################################################################
		#################################################################################################
		##
		##	Another idea I had, which is probably overkill for this usecase, would be to build a quadtree.
		##
		##	The quadtree would adaptatively split the bounding boxes in different regions of space.
		##	This would make it faster to compute and could handle a way bigger number of boxes and could 
		##	even enable the use of streaming different chunks of the tree in memory and to disk.
		##
		##	Obviously this idea would work only if the use case needed millions of boxes to be processed
		##	which is not the case here.
		##
		#################################################################################################
		#################################################################################################

		Parameters:
		-iou_threshold (float, default=0.5): Optional IoU threshold
		

		Returns:
		-Nothing
		-Sets self.final_res to an array containing the cleaned boxes

		"""
		
		# If there's no data, we just stop here
		if len(self.data) == 0:
			return []
		
		# Sort the data by confidence
		sortedby_confidence_data = np.array(sorted(self.data, key=lambda x : x[-1]))
		# Indices for the data, this is the only array that's modified
		sortedby_confidence_indices = np.array(range(len(sortedby_confidence_data)))
		# The final indices
		res = []
		
		# While we haven't processed all the indices that are remaining
		
		curr_len = len(sortedby_confidence_indices)
		while curr_len > 0:
			# Get the index of the highest confidence box
			highest_pc_index = curr_len - 1
			# Add it to the result because it's the one we're most confident about
			# Also, we're going to remove it soon from the indices
			res.append(sortedby_confidence_indices[highest_pc_index])
			last = sortedby_confidence_indices[:-1]
			# Compared the highest confidence box to all the others before it sorted by the botton x coordinate
			ious = np.array([self.get_iou(sortedby_confidence_data[sortedby_confidence_indices[-1]], b) for b in sortedby_confidence_data[last]])

			# Remove the highest confidence because we're done with it
			sortedby_confidence_indices = np.delete(sortedby_confidence_indices, highest_pc_index)
			# Remove all the boxes with IoU > threshold=0.5
			sortedby_confidence_indices = np.delete(sortedby_confidence_indices, np.where(ious > iou_threshold)[0])
			
			curr_len = len(sortedby_confidence_indices)

		# Here we get the actual boxes using the remaining indices
		self.final_res = sortedby_confidence_data[res]
		# We postprocess the data to get back the height and width of the boxes
		self.final_res[:, 2] -= self.final_res[:, 0]
		self.final_res[:, 3] -= self.final_res[:, 1]
		
	def write_res(self, path, delimiter=','):
		"""Writes the result in a file
	
	
		Parameters:
		-path: Path of the file to write the result to
		-delimiter (char, default=','): Optional IoU threshold


		Returns:
		-Nothing

		"""
		
		# Open the file
		f = open(path, "w")
		for bb in self.final_res:
			# Write every bb coordinate as an "int + delimiter + space" to match the output the subject ask for
			[f.write(str(int(i)) + delimiter + ' ') for i in bb[:-1]]
			# The last element is a float and doesn't need a space but a newline character
			f.write(str(bb[-1]) + '\n')

if __name__ == "__main__":
	# Handle not enough cmd line args
	if len(sys.argv) < 3:
		sys.exit("Usage: python non_max_supression.py in_detections.csv out_detections.csv")
	
	# Call all the helper functions
	nms = nms()
	nms.get_data(sys.argv[1])
	nms.compute_nms()
	nms.write_res(sys.argv[2])