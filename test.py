import sys
# sys.path.append(r'C:\Users\D\Desktop\cnncode')
import ColinearCondition as CC


cornerPointx = [516500.0, 516749.9, 516579.7, 516763.8]
cornerPointy = [3787397.8, 3787373.5, 3787230.6, 3787224.6]
lpstrDem = r"E:\ChenLab\Xuzhou-Aerial\DEM\Xuzhou_DEM_03m.tif"
lpstrCam = r"E:\ChenLab\Xuzhou-Aerial\Images\029518.cam"
cal = CC.ConverXYZ2Imge(lpstrDem, lpstrCam)
out = cal.convert_xyzcorner_2_img(cornerPointx, cornerPointy)
# out = cal.convert_xyzcorner_2_img(517112.2, 3787400.2)
# print(out)
# print(out.x, out.y)
print(out[0].x, out[0].y)
print(out[1].x, out[1].y)
print(out[2].x, out[2].y)
print(out[3].x, out[3].y)
print(help(CC))
