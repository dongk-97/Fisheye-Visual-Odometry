import numpy as np 
import cv2
import random

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1000


lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2

def dist(origin, pt):
    x = pt[0]
    y = pt[1]
    return np.sqrt((origin[0]-x)**2+(origin[1]-y)**2)

def is_near(l, pt):
    for ep in l:
        if dist(ep, pt)<20:
            return True
    return False


def eightpoint(s1, s2):
    idx = random.sample(range(0, len(s1)), 8)
    U = np.zeros((8,9))
    m1 = []
    m2 = []
    for i in range(8):
        p1 = s1[idx[i]]
        p2 = s2[idx[i]]
        m1.append(p1)
        m2.append(p2)
        p1 = np.array([p1])
        p2 = np.array([p2])
        P = np.dot(p1.T, p2).flatten()
        U[i, :] = P
    A = np.dot(U.T, U)
    return A, m1, m2

def findEssential(A):
    U, s, V = np.linalg.svd(A, full_matrices = True)
    F = V[8, :].reshape((3,3)).T
    U, s, V = np.linalg.svd(F, full_matrices = True)
    s[2] = 0
    S = np.diag(s)
    E = np.dot(U, np.dot(S,V))
    return E


def count_inlier(s1, s2, E, t, th):
    count = 0
    for i in range(len(s1)):
        p1 = np.array([s1[i]]).T
        p2 = np.array(s2[i])
        n1 = E.dot(p1).reshape(1,-1)[0]
        n2 = np.cross(p2, t)
        n1_n = np.sqrt(n1[0]**2+n1[1]**2+n1[2]**2)
        n2_n = np.sqrt(n2[0]**2+n2[1]**2+n2[2]**2)
        n1 = abs(n1/n1_n)
        n2 = abs(n2/n2_n)
        ang = np.arccos(np.dot(n1, n2))
        if ang<th:
            count = count + 1
    return count


def Ransac(s1, s2, it, th, ratio):
    num_point = len(s1)
    max_inlier = 0
    E_best = np.zeros((3,3))
    R_best = np.zeros((3,3))
    t_best = np.array([0,0,0])
    for i in range(it):
        A, m1, m2 = eightpoint(s1, s2)
        E = findEssential(A)
        R, t = findRt(E, m1, m2)
        inlier = count_inlier(s1, s2, E, t, th)
        if max_inlier<inlier:
            max_inlier = inlier
            E_best = E
            R_best = R
            t_best = t
            if max_inlier/num_point > ratio:
                break
    return E_best, R_best, t_best


def findRt(E,s1,s2):
	U, s, V = np.linalg.svd(E, full_matrices = True)
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	W_inv = np.array([[0,1,0],[-1,0,0],[0,0,1]])
	R1 = U.dot(W).dot(V)
	R2 = U.dot(W_inv).dot(V)
	R1 = R1*np.linalg.det(R1)
	R2 = R2*np.linalg.det(R2)
	p1 = np.array([s1[0]]).T
	p2 = np.array([s2[0]]).T
	d1 = R1.dot(p1)-p2
	d2 = R2.dot(p1)-p2
	R = R2
	if d1[0,0]**2+d1[1,0]**2+d1[2,0]**2<d2[0,0]**2+d2[1,0]**2+d2[2,0]**2:
		R = R1
	t = -U[:, 2]
	if t[2]<0:
		t = -t
	return R, t


def is_correct_R_mag(s1, s2, R):
	for i in range(8):
		p1 = np.array([s1[i]]).T
		p2 = np.array([s2[i]]).T
		d = np.abs(R.dot(p1))-np.abs(p2)
		if np.sqrt(d[0,0]**2+d[1,0]**2+d[2,0]**2)<0.000001:
			return True
	return False

def is_correct_R_dir(s1, s2, R):
	p1 = np.array([s1[0]]).T
	p2 = np.array([s2[0]]).T
	p12 = R.dot(p1)
	if p12[0,0]*p2[0,0]<0:
		return False
	return True

class FisheyeCamera:
	def __init__(self, width, height, cx, cy, a0, a1, a2, a3, a4, r):
		self.width = width
		self.height = height
		self.cx = cx
		self.cy = cy
		self.a0 = a0
		self.a1 = a1
		self.a2 = a2
		self.a3 = a3
		self.a4 = a4
		self.r = r


class VisualOdometry:
	def __init__(self, cam, annotations):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.pt_prev = None
		self.desc_prev = None
		self.pt_curr = None
		self.desc_curr = None
		self.center = (cam.cx, cam.cy)
		self.radius = cam.r
		self.trueX, self.trueY, self.trueZ = 0, 0, 0 
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.bf = cv2.BFMatcher()
		self.startX, self.startY, self.startZ = 0, 0, 0
		with open(annotations) as f:
			self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id):  #specialized for KITTI odometry dataset
		ss = self.annotations[frame_id+390-1].strip().split()
		x_prev = float(ss[1])
		y_prev = float(ss[2])
		z_prev = float(ss[3])
		ss = self.annotations[frame_id+390].strip().split()
		x = float(ss[1])
		y = float(ss[2])
		z = float(ss[3])
		if frame_id == 2:
			self.startX, self.startY, self.startZ = x, y, z
		self.trueX, self.trueY, self.trueZ = x-self.startX, y-self.startY, z-self.startZ
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, self.cur_R, self.cur_t = self.findERt(self.px_ref, self.px_cur, 50, 0.0002, 0.8)
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, R, t = self.findERt(self.px_ref, self.px_cur, 50, 0.0002, 0.8)
		absolute_scale = self.getAbsoluteScale(frame_id)
		self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
		self.cur_R = R.dot(self.cur_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = self.circleCrop(img)
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame

	def circleCrop(self, img):
		mask = np.zeros_like(img)
		rows, cols = mask.shape
		mask = cv2.circle(mask, center = self.center , radius = self.radius, color = (255,255,255), thickness=-1)
		result = np.bitwise_and(img, mask)
		return result
	
	def toSphere(self, mp1, mp2):
		s1 = []
		s2 = []
		for i in range(mp1.shape[0]):
			s1.append(self.est_sphere((mp1[i,0], mp1[i,1])))
			s2.append(self.est_sphere((mp2[i,0], mp2[i,1])))
		return s1, s2

	def est_sphere(self, pt):
		ro = dist(self.center, pt)
		x = pt[0]-self.center[0]
		y = pt[1]-self.center[1]
		z = np.abs(self.cam.a0+self.cam.a1*ro+self.cam.a2*ro**2+self.cam.a3*ro**3+self.cam.a4*ro**4)
		n = np.sqrt(x**2+y**2+z**2)
		x = x/n
		y = y/n
		z = z/n
		return [x, y, z]

	def findERt(self, mp1, mp2, it, th, ratio):
		s1, s2 = self.toSphere(mp1, mp2)
		E, R, t = Ransac(s1, s2, it, th, ratio)
		t = t.reshape(-1, 1)
		return E, R, t

