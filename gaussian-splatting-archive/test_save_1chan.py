import glob
import cv2

li = sorted(
    glob.glob("/path/to/XoFTR/METU_VisTIR/cloudy/scene_1/forward_warp_back_thermal_depth_final/*.png")
)

for ele in li:
    I = cv2.imread(ele, -1)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(ele, I)
