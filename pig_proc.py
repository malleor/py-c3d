import matplotlib.pyplot as plt
from lmj import c3d
import string as s
import os
import struct
import numpy
from copy import deepcopy

class ForcePlates(object):
	def __init__(self, group):
		self.num_plates = abs(struct.unpack('h', group.params['USED'].bytes)[0])
		# corners
		corners_param = group.params['CORNERS']
		corners_flat = struct.unpack('f'*3*4*self.num_plates, corners_param.bytes)
		self.corners = numpy.reshape(corners_flat, (self.num_plates, 4, 3))
		# origin
		origin_param = group.params['ORIGIN']
		origin_flat = struct.unpack('f'*3*self.num_plates, origin_param.bytes)
		self.origin = numpy.reshape(origin_flat, (self.num_plates, 3))
	
	def write(self, group):
		pass
	
	def plot(self, fig=None):
		f = plt.figure(fig or 'Force plates'); plt.clf()
		a = f.gca(projection='3d')
		colors = map(lambda x: x, 'rbgcmyk'[::-1])
		for i in xrange(self.num_plates):
			c = colors.pop()
			# corners
			x, y, z = zip(*self.corners[i])
			a.scatter(x, y, z, c=c)
			# origin
			x, y, z = self.origin[i]
			a.scatter(x, y, z, c=c)
		plt.show()

		
class Sequence(object):
	''' A scalar/vector sequence for storing c3d analog and video data. '''
	
	def __init__(self, array, name=''):
		if type(array) != numpy.ndarray:
			array = numpy.array(array)
		
		self.array = deepcopy(array)
		self.name = name
	
	def __str__(self):
		dims = reduce(lambda s,x: s+'x'+str(x), self.array.shape[1:], '')[1:]
		return 'Sequence \'%s\' of %d frames, %s scalars each' % (self.name, self.array.shape[0], dims)


class C3DContent(object):
	def __init__(self, path):
		import logging
		
		log = logging.FileHandler(path+'_read.log', 'wt')
		log.setLevel(logging.DEBUG)
		logging.root.addHandler(log)
		
		# load the c3d
		self.file = path
		self._reader = c3d.Reader(open(path, 'rb')) # TODO: remove from self
		
		# get a copy of the header
		self.header = deepcopy(self._reader.header)
		
		# get a copy of parameters
		self.groups = deepcopy(self._reader._groups)
		
		# get video labels
		labels_raw = self.getgroup('POINT').params['LABELS']
		label_len, num_markers = labels_raw.dimensions
		extract_label = lambda i: labels_raw.bytes[label_len*i:label_len*(i+1)].strip(' ')
		mask_non_markers = lambda label: label if C3DContent._is_marker(label) else None
		video_labels = [mask_non_markers(extract_label(i)) for i in xrange(num_markers)]
		
		# get video labels
		labels_raw = self.getgroup('ANALOG').params['LABELS']
		label_len, num_markers = labels_raw.dimensions
		extract_label = lambda i: labels_raw.bytes[label_len*i:label_len*(i+1)].strip(' ')
		analog_labels = [extract_label(i) for i in xrange(num_markers)]
		
		# get a copy of video and analog data
		video, analog = zip(*self._reader.read_frames())
		num_video_sequences = video[0].shape[0]
		num_analog_sequences = analog[0].shape[1]
		make_video_sequence = lambda i: Sequence([f[i] for f in video], video_labels[i])
		make_analog_sequence = lambda i: Sequence([f[:,i] for f in analog], analog_labels[i])
		self.video = [make_video_sequence(i) for i in xrange(num_video_sequences) if video_labels[i]]
		self.analog = [make_analog_sequence(i) for i in xrange(num_analog_sequences)]
		logging.info('Extracted video sequences:')
		for s in self.video: logging.info('  ' + str(s))
		logging.info('Extracted analog sequences:')
		for s in self.analog: logging.info('  ' + str(s))
		
		# get force plates info
		fp_group = self.getgroup('FORCE_PLATFORM')
		self.force_plates = ForcePlates(fp_group)
		
		logging.root.removeHandler(log)
	
	def getgroup(self, name):
		return self.groups.get(name.upper(), None)

	def plot(self, marker_index_or_label, fig=None, limits=None, mstyle=None):
		'''Plots a video signal of given its index or label.'''
		# TODO: refactor to use sequence objects
		label, marker = self.getmarker(marker_index_or_label)
		if limits is not None:
			marker = marker[limits[0]:limits[1]]
		plt.figure(fig or 'Marker '+label)
		for i in xrange(4):
			plt.subplot(4, 1, i+1)
			plt.cla()
			plt.plot([v[i] for v in marker], marker=mstyle)
	
	def save(self, path):
		# TODO: refactor to use sequence objects
		raise NotImplementedError('\'save\' function is currently under construction')
		
		import logging
		
		log = logging.FileHandler(path+'_write.log', 'wt')
		log.setLevel(logging.DEBUG)
		logging.root.addHandler(log)
		
		# prepare output
		handle = open(path, 'wb')
		writer = c3d.Writer(handle)
		
		# write metadata
		writer.header = self.header
		writer._groups = self.groups
		writer.write_metadata()
		handle.flush()
		
		# write video and analog data
		frames = zip(self.video, self.analog)
		writer.write_frames(frames)
		
		logging.root.removeHandler(log)

	def getmarker(self, label):
		'''Retreives a sequence of a specific video signal.'''
		
		# TODO: select sequence
		raise NotImplementedError('\'getmarker\' function is currently under construction')
		
		return (sequence.name, sequence.array)

	def getmarkers(self):
		for index in video_labels.itervalues():
			yield self.getmarker(index)
	
	@staticmethod
	def _label_matches(base_label):
		base_label = base_label.split(':')[-1]
		base_label = base_label.lower()
		return lambda tested_label: tested_label.lower() == base_label or \
			tested_label.split(':')[-1].lower() == base_label
	
	@staticmethod
	def _is_marker(label):
		label = label.split(':')[-1]
		return len(label) in xrange(4,6) and label == label.upper()


def load(path):
	ext = os.path.splitext(path)[1].lower()
	if ext != '.c3d':
		raise ValueError('expected .c3d path, got '+ext)
	
	return C3DContent(path)

def save(content, dirpath, limits=None, separate=True, markers=None):
	'''Saves c3d trajectories into a txt file.
	
	content   - C3DContent instance with trajectories
	dirpath   - directory path to be written into
	limits    - tuple (beg,end) for trajectories cropping (default: no cropping)
	separate  - write each trajectory into a separate file? (default: yes)
	markers   - list of markers to be written (default: all)
	'''
	if not os.path.isdir(dirpath):
		dirpath = os.path.split(dirpath)[0]
	
	make_path = lambda filename: os.path.join(dirpath, filename+'.txt')
	h = open(make_path('all_markers'), 'wt') if not separate else None
	
	for label, marker in content.getmarkers():
		if markers is not None and len(filter(C3DContent._label_matches(label), markers)) == 0:
			continue
		
		filename = s.split(label, ':')[-1] if separate else None
		h = open(make_path(filename), 'wt') if separate else h
		
		marker = marker[limits[0]:limits[1]] if limits else marker
		
		for pt in marker:
			if pt[3] != -1.0:
				h.write('%f %f %f\n' % (pt[0],pt[1],pt[2]))
	
	markers_str = 'all markers' if markers is None else 'markers:\n' + str(markers) + '\n'
	output_str = 'directory:\n' + dirpath if separate else 'file:\n' + h.name
	print 'Written', markers_str, 'to', output_str
