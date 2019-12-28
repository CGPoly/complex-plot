import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)  # fig.add_subplot(111, projection='3d')
res = 0.01

y_1_list = list()
z_1_list = list()
y_2_list = list()
z_2_list = list()
y_3_list = list()
z_3_list = list()
y_4_list = list()
z_4_list = list()
x_list = np.arange(-5 + res, 3 + res, res)
for x_var in x_list:
    try:
        tmp = (-1) ** complex(x_var)
        y_1_list.append(tmp.real)
        z_1_list.append(tmp.imag)
        
        tmp = complex(-abs(x_var)) ** complex(x_var)
        y_2_list.append(tmp.real)
        z_2_list.append(tmp.imag)
        
        tmp = complex(-x_var) ** complex(x_var)
        y_3_list.append(tmp.real)
        z_3_list.append(tmp.imag)
        
        tmp = np.e ** (np.pi * 1j * x_var)
        y_4_list.append(tmp.real)
        z_4_list.append(tmp.imag)
    except RuntimeWarning:
        pass
    pass
ax.plot(x_list, y_1_list, z_1_list, label="(-1)**x")
# ax.plot(x_list, y_2_list, z_2_list, label="(-|x|)**x")
# ax.plot(x_list, y_3_list, z_3_list, label="(-x)**x")
# ax.plot(x_list, y_4_list, z_4_list, label="((x)**-x)**0.5")
ax.plot(x_list, y_4_list, z_4_list, label="e^(pi*i*x)")
plt.xlabel("input Reel")
plt.ylabel("output Reel")
ax.legend()
plt.show()
