import string as s
import os
import struct
from copy import deepcopy
import logging
from itertools import izip

from lmj import c3d

import numpy
from numpy import reshape
from numpy import array as ar

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


class ForcePlates(object):
	''' Information related to force plates measurement. '''
	
	def __init__(self, group, analog):
		self.num_plates = abs(struct.unpack('h', group.params['USED'].bytes)[0])
		
		# corners
		corners_param = group.params['CORNERS']
		corners_flat = struct.unpack('f'*3*4*self.num_plates, corners_param.bytes)
		self.corners = reshape(corners_flat, (self.num_plates, 4, 3))
		
		# origin
		origin_param = group.params['ORIGIN']
		origin_flat = struct.unpack('f'*3*self.num_plates, origin_param.bytes)
		self.origin = reshape(origin_flat, (self.num_plates, 3))
		
		# force data
		self.analog = analog
	
	def write(self, group):
		pass
	
	def plot(self, fig=None):
		''' Plots the force plates and measured 'footprints'. '''
		
		f = plt.figure(fig or 'Force plates'); plt.clf()
		a = f.gca(projection='3d')
		colors = 'rbgcmyk'
		
		prints = self.forces()
		
		for i in xrange(self.num_plates):
			c = colors[i]
			
			# corners
			x, y, z = zip(*self.corners[i])
			a.scatter(x, y, z, c=c)
			
			# origin
			x, y, z = self.origin[i]
			a.scatter(x, y, z, c=c)
			
			# footprints
			x, y, z = prints.next()
			every = 20
			a.scatter(x[::every], y[::every], z[::every], c=colors[i])
		
		plt.show()
	
	def _calc_forces(self, fx, fy, fz, mx, my, mz):
		min_load = 100. # [Nmm]
		xgen = (n/d if abs(m)>min_load else 0. for n,d,m in zip(my.array,fz.array,fz.array))
		ygen = (n/d if abs(m)>min_load else 0. for n,d,m in zip(mx.array,fz.array,fz.array))
		zgen = (0. for i in xrange(fx.array.size))
		for x, y, z in izip(xgen, ygen, zgen):
			yield (x, y, z)
	
	def forces(self):
		''' Calculates and returns forces trajectories for each force plate. '''
		
		for i in xrange(len(self.analog) / 6):
			# get force x,y in local coords
			pgen = self._calc_forces(*self.analog[i*6:i*6+6])
			x, y, z = zip(*filter(lambda pt: pt[0]*pt[1] != 0, pgen))
			
			# move to plate's center
			cx, cy, cz = numpy.mean(self.corners[i], 0)
			x = numpy.add(x, cx)
			y = numpy.add(y, cy)
			z = numpy.add(z, cz)
			
			yield (x, y, z)


class Sequence(object):
	''' A scalar/vector sequence for storing c3d analog and video data. '''
	
	def __init__(self, array, name=''):
		if type(array) != numpy.ndarray:
			array = ar(array)
		
		self.array = deepcopy(array)
		self.name = name
	
	def __str__(self):
		dims = reduce(lambda s,x: s+'x'+str(x), self.array.shape[1:], '')[1:] + ' scalars' if self.array.ndim > 1 else '1 scalar'
		return 'Sequence \'%s\' of %d frames, %s each' % (self.name, self.array.shape[0], dims)
	
	def crop(self, beg, end):
		''' Crops the sequence to given limits as frame indices. 
		Returns a copy of contained array.'''
		
		return Sequence(self.array[beg:end], self.name)


class C3DContent(object):
	''' A c3d file content. Contains 3-D and scalar sequences 
	along with metadata considering the measurement. '''
	
	def __init__(self, path):
		log = logging.FileHandler(path+'_read.log', 'wt')
		log.setLevel(logging.DEBUG)
		logging.root.addHandler(log)
		
		get_param_group = lambda name: self.groups.get(name.upper(), None)
		
		# load the c3d
		self.file = path
		self._reader = c3d.Reader(open(path, 'rb')) # TODO: remove from self
		
		# get a copy of the header
		self.header = deepcopy(self._reader.header)
		
		# get a copy of parameters
		self.groups = deepcopy(self._reader._groups)
		
		# get video labels
		labels_raw = get_param_group('POINT').params['LABELS']
		label_len, num_markers = labels_raw.dimensions
		extract_label = lambda i: labels_raw.bytes[label_len*i:label_len*(i+1)].strip(' ')
		mask_non_markers = lambda label: label if C3DContent._is_marker(label) else None
		video_labels = [mask_non_markers(extract_label(i)) for i in xrange(num_markers)]
		
		# get video labels
		labels_raw = get_param_group('ANALOG').params['LABELS']
		label_len, num_markers = labels_raw.dimensions
		extract_label = lambda i: labels_raw.bytes[label_len*i:label_len*(i+1)].strip(' ')
		analog_labels = [extract_label(i) for i in xrange(num_markers)]
		
		# get a copy of video and analog data
		video, analog = zip(*self._reader.read_frames())
		num_video_sequences = video[0].shape[0]
		num_analog_sequences = analog[0].shape[1]
		make_video_sequence = lambda i: Sequence([f[i] for f in video], video_labels[i])
		make_analog_sequence = lambda i: Sequence(numpy.append([], [f[:,i] for f in analog]), analog_labels[i])
		self.video = [make_video_sequence(i) for i in xrange(num_video_sequences) if video_labels[i]]
		self.analog = [make_analog_sequence(i) for i in xrange(num_analog_sequences)]
		logging.info('Extracted video sequences:')
		for s in self.video: logging.info('  ' + str(s))
		logging.info('Extracted analog sequences:')
		for s in self.analog: logging.info('  ' + str(s))
		
		# get force plates info
		fp_group = get_param_group('FORCE_PLATFORM')
		self.force_plates = ForcePlates(fp_group, self.analog)
		
		logging.root.removeHandler(log)

	def plot(self, label, fig=None, limits=None, mstyle=None):
		''' Plots a signal of given its index or label. '''
		
		sequence = self.find_video(label) or self.find_analog(label)
		if not sequence: 
			raise ValueError('sequence not found')
		array = sequence.array[limits[0]:limits[1]] if limits else sequence.array
		
		plt.figure(fig or 'Sequence \'%s\' - plots' % sequence.name)
		dims = array.shape[1] if array.ndim > 1 else 1
		for i in xrange(dims):
			plt.subplot(dims, 1, i+1)
			plt.cla()
			plt.plot([v[i] for v in array] if array.ndim > 1 else array, marker=mstyle)

	def plot3(self, label=None, fig=None, limits=None, color=None, clear=True):
		''' Plots a marker in 3D. '''
		
		if not label:
			# plot all markers
			from numpy.random import rand
			fig = fig or 'All sequences - 3D view'
			for s in self.video:
				self.plot3(label=s.name, fig=fig, limits=limits, color=color or tuple(rand(3)), clear=clear)
				clear = False
		else:
			# plot a specific marker
			sequence = self.find_video(label)
			if not sequence: 
				raise ValueError('sequence not found')
				
			f = plt.figure(fig or 'Sequence \'%s\' - 3D view' % sequence.name)
			if clear: plt.clf()
			a = f.gca(projection='3d')
			
			array = sequence.array[limits[0]:limits[1]] if limits else sequence.array
			x, y, z = tuple(numpy.split(array, 4, axis=1))[:3]
			
			a.scatter(x, y, z, c=color or 'r', label=sequence.name)
	
	def save(self, path):
		''' Save the content to a c3d file. '''
		
		log = logging.FileHandler(path+'_write.log', 'wt')
		log.setLevel(logging.DEBUG)
		logging.root.addHandler(log)
		
		# prepare output
		handle = open(path, 'wb')
		writer = c3d.Writer(handle)
		
		# prepare data packaging, update the header
		num_frames = self.video[0].array.shape[0]
		num_analog_samples = self.analog[0].array.shape[0]
		num_analog_samples_per_frame = num_analog_samples / num_frames
		self.header.point_count = len(self.video)
		self.header.analog_count = len(self.analog)*num_analog_samples_per_frame
		self.header.first_frame = 0
		self.header.last_frame = num_frames-1
		trial_group = self.groups.get('TRIAL', None)
		trial_group.params['ACTUAL_START_FIELD'].bytes = struct.pack('I', self.header.first_frame)
		trial_group.params['ACTUAL_END_FIELD'].bytes = struct.pack('I', self.header.last_frame)
		point_group = self.groups.get('POINT', None)
		point_group.params['USED'].bytes = struct.pack('H', self.header.point_count)
		
		# write metadata
		writer.header = self.header
		writer._groups = self.groups
		writer.write_metadata()
		handle.flush()
		logging.debug('Written metadata.')
		
		# write video and analog data
		logging.info('''Forming output frames: 
                  num_frames: %d
         num_video_sequences: %d
        num_analog_sequences: %d
num_analog_samples_per_frame: %d''' % (num_frames, len(self.video), len(self.analog), num_analog_samples_per_frame))
		get_marker = lambda i: [s.array[i] for s in self.video]
		get_analog_signal = lambda i: [s.array.view().reshape((num_frames,num_analog_samples_per_frame))[i] for s in self.analog]
		video = tuple(ar(get_marker(i)) for i in xrange(num_frames))
		analog = tuple(ar(get_analog_signal(i)).T for i in xrange(num_frames))
		frames = zip(video, analog)
		writer.write_frames(frames)
		logging.debug('Written frames.')
		
		logging.root.removeHandler(log)

	def find_video(self, label):
		'''Retreives a video sequence of a specific name.'''
		try:
			is_good_name = C3DContent._label_matches(label)
			return filter(lambda v: is_good_name(v.name), self.video)[0]
		except IndexError:
			return None

	def find_analog(self, label):
		'''Retreives an analog sequence of a specific name.'''
		try:
			is_good_name = C3DContent._label_matches(label)
			return filter(lambda v: is_good_name(v.name), self.analog)[0]
		except IndexError:
			return None
	
	def crop_sequences(self, beg, end):
		''' Crops all stored sequences to given limits as a tuple with 
		beginning and ending frames. '''
		
		num_frames = self.video[0].array.shape[0]
		num_analog_samples = self.analog[0].array.shape[0]
		num_analog_samples_per_frame = num_analog_samples / num_frames
		
		self.video = map(lambda s: s.crop(beg, end), self.video)
		self.analog = map(lambda s: s.crop(beg*num_analog_samples_per_frame, end*num_analog_samples_per_frame), self.analog)
		
		self.force_plates.analog = self.analog
	
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
	
	# TODO: refactor to use sequence objects
	raise NotImplementedError('\'save\' function is currently under construction')
	
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
