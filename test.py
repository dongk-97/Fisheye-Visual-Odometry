import numpy as np 
import cv2

from visual_odometry import FisheyeCamera, VisualOdometry


cam = FisheyeCamera(640.0, 480.0, 320, 240, -179.47183, 0, 0.002316744, -3.6359684*10**(-6), 2.0546507*10**(-8), 256)
vo = VisualOdometry(cam, '/home/dongk/다운로드/rpg_urban_fisheye_info/info/groundtruth.txt') # CHANGE THIS DIRECTORY PATH

traj = np.zeros((600,600,3), dtype=np.uint8)

for img_id in range(2500):
	img = cv2.imread('/home/dongk/다운로드/rpg_urban_fisheye_data/data/img/'+'img'+str(img_id+390+1).zfill(4)+'_0.png', 0)  # CHANGE THIS DIRECTORY PATH
	
	vo.update(img, img_id)

	cur_t = vo.cur_t
	if(img_id > 2):
		x, y, z = cur_t[0,0], cur_t[1,0], cur_t[2,0]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x)+290, int(z)+90
	true_x, true_y = int(-vo.trueX)+290, int(vo.trueY)+90

	cv2.circle(traj, (draw_x,draw_y), 1, (255.0,0), 2)
	cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 1)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "x=%2fm y=%2fm z=%2fm"%(x,y,z)+ "x=%2fm z=%2fm"%(-vo.trueX, vo.trueY)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

cv2.imwrite('map.png', traj)
