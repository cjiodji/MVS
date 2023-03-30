import numpy as sy
from math import *
import tkinter as tk
import easygui as g
import tkinter.filedialog
import tkinter.messagebox as tm
from tkinter import messagebox

g.msgbox('打开文件', ok_button='选择文件')
path = tk.filedialog.askopenfilename()  # 读取文件路径
test = sy.loadtxt(path, delimiter=',', skiprows=0)  # 读取文件数据
print(len(test))
print('原始数据如下((x,y),(X,Y,Z)):\n', test)
m = eval(input('请输入比例尺（m）：'))
f = eval(input('请输入主距（m）：'))
x0, y0 = 0, 0
xy = []
XYZ = []
num = 0
for i in range(len(test)):
    xy.append([test[i][0] / 1000, test[i][1] / 1000])
    # 除以1000是因为统一单位
    XYZ.append([test[i][2], test[i][3], test[i][4]])
# 分别形成像点控制点的矩阵
A = sy.mat(sy.zeros((8, 6)))
# 四个控制点，每个是2行6列，四个控制点是8行6列
# v=ax-l
# mat函数用来将列表转换成相应矩阵 zeros函数是生成指定维数的全0数组zeros8部分，里面有2个内容
L = sy.mat(sy.zeros((8, 1)))
R = sy.mat(sy.zeros((3, 3)))
pds = sy.mat(sy.zeros((4, 3)))
XS0 = 0
YS0 = 0
# 旋转矩阵
for i in range(len(test)):
    XS0 = (test[i][2] + XS0) / 4
    YS0 = (test[i][3] + YS0) / 4
    ZS0 = m * f
print('线元素的初始坐标如下(XS0,YS0,ZS0):\n')
print(XS0)
print(YS0)
print(ZS0)
# Xs0=sumxti/t为初始值
pi = 0
w = 0
k = 0
diedai = 0

"""
while(True)：
a1 = cos(pi)*cos(k)-sin(pi)*sin(w)*sin(k)
a2 = (-1.0) * cos(pi) * sin(k) - sin(pi) * sin(w) * cos(k)
    a3 = (-1.0) * sin(pi) * cos(w)
    b1 = cos(w) * sin(k)
    b2 = cos(w) * cos(k)
    b3 = (-1.0) * sin(w)
    c1 = sin(pi) * cos(k) + cos(pi) * sin(w) * sin(k)
    c2 = (-1.0) * sin(pi) * sin(k) + cos(pi) * sin(w) * cos(k)
    c3 = cos(pi) * cos(w)
    R=sy.mat([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
#第四步，计算旋转矩阵RA[i * 2,0] = (-1.0) * f / (ZS0 )
for i in range(0,len(XYZ)):

        x = xy[i][0]
        y = xy[i][1]
        Xp = a1*(XYZ[i][0]-XS0) + b1*(XYZ[i][1]-YS0) + c1*(XYZ[i][2]-ZS0);
        Yp = a2*(XYZ[i][0]-XS0) + b2*(XYZ[i][1]-YS0) + c2*(XYZ[i][2]-ZS0);
        Zp = a3*(XYZ[i][0]-XS0) + b3*(XYZ[i][1]-YS0) + c3*(XYZ[i][2]-ZS0);
#求平均数
        A[2*i, 0] = (a1*f + a3*(x-x0))/Zp
        A[2*i, 1] = (b1*f + b3*(x-x0))/Zp
        A[2*i, 2] = (c1*f + c3*(x-x0))/Zp
        A[2*i, 3] = (y-y0)*sin(w)-(((x-x0)*((x-x0)*cos(k) - (y-y0)*sin(k)))/f+ f*cos(k))*cos(w);
        A[2*i, 4] = -f*sin(k) - (x-x0)*((x-x0)*sin(k) +(y-y0))*cos(k)/ f
        A[2*i, 5] = y-y0;
        A[2*i+1, 0] = (a2*f + a3*(y-y0)) / Zp
        A[2 * i + 1, 1] = (b2*f + b3*(x-x0))/ Zp
        A[2 * i + 1, 2] = (c2*f + c3*(x-x0))/Zp
        A[2 * i + 1, 3] = -(x-x0)*sin(w) - ((y-y0)*(x*cos(k) - y)*sin(k)) / f - f*sin(k)*cos(w);
        A[2 * i + 1, 4] = -f*cos(k) - (y-y0)/ f*((x-x0)*sin(k) + (y-y0)*cos(k))
        A[2 * i + 1, 5] = -(x-x0)
    #a11等系数'''
        L[i * 2,0]=x+f*(Xp/Zp)
        L[i * 2 + 1,0] =y+f*(Yp/Zp)
    #常数项l
Result = [0]*(9+4)
#存改正数的
Result=((A.T*A).I)*A.T*L
XS0+=Result[0]
YS0+=Result[1]
ZS0+=Result[2]
pi+=Result[3]
w+=Result[4]
k+=Result[5]
diedai=diedai+1
if (fabs (Result[3]) < 1e-6) and (fabs (Result[4]) < 1e-6) and (fabs (Result[5]) < 1e-6):
    break
if diedai >60:
    print("Error:overtime")
    break
a1=cos(pi)*cos(k)-sin(pi)*sin(w)*sin(k)
a2=(-1.0) * cos(pi) * sin(k) - sin(pi) * sin(w) * cos(k)
a3=(-1.0) * sin(pi) * cos(w)
b1=cos(w) * sin(k)
b2=cos(w) * cos(k)
b3=(-1.0) * sin(w)
c1=sin(pi) * cos(k) + cos(pi) * sin(w) * sin(k)
c2=(-1.0) * sin(pi) * sin(k) + cos(pi) * sin(w) * cos(k)
c3=cos(pi) * cos(w)
rotate=sy.mat([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
list1=["XS0:",XS0,"YS0:",YS0,"ZS0:",ZS0,"pi:",pi,"w:",w,"k:",k]
print('计算结果\n','外方位元素：\n:'"XS0:",XS0,'\n',"YS0:",YS0,'\n',"ZS0:",ZS0,'\n','PI:',pi,'\n','W:',w,'\n',"K:",k,'\n')
print('旋转矩阵\n',rotate)
print('迭代次数为：',diedai)
messagebox.showinfo("外方位元为：",list1)
"""
