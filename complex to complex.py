import numpy as np
import matplotlib.pyplot as plt
import math
# import scipy.special

import operator
from functools import reduce


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def rec(function: callable, x: complex, n: int) -> complex:
    if n <= 1:
        return function(x)
    return function(rec(function, x, n-1))


class Plotter:
    def __init__(self, resolution: int = 256, distance: float = 7.0, format_of_image: tuple = (3, 2.1), squish: bool = False):
        self.format = format_of_image
        self.squish = squish
        self.resolution = resolution
        self.distance = distance
    
    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> tuple:
        c = s * v
        x = c * (1 - abs(((h / 60) % 2) - 1))
        m = v - c
        rs, gs, bs, = (c, x, 0) if 0 <= h < 60 else ((x, c, 0) if 60 <= h < 120 else ((0, c, x) if 120 <= h < 180 else (
            (0, x, c) if 180 <= h < 240 else ((x, 0, c) if 240 <= h < 300 else (c, 0, x)))))
        return rs + m, gs + m, bs + m
    
    @staticmethod
    def contour_plot(polar: tuple, angle: bool) -> tuple:
        r0 = 0.0
        r1 = 5e-2
        while polar[0] > r1:
            r0 = r1
            r1 = r1 * np.e
        r = (polar[0] - r0) / (r1 - r0)
        # q1 = 1 - (2 * r) ** 10 if r < 0.5 else 1 - (2 * (1 - r)) ** 10
        q1 = 1 - r**10
        v = 1
        if angle:
            phi = polar[1]
            phi /= np.pi
            phi *= 180
            phi %= 20
            phi /= 18
            # v = 1 - phi ** 10
            v = 1 - (2 * phi) ** 10 if phi < 0.5 else 1 - (2 * (1 - phi)) ** 10
        return Plotter.hsv_to_rgb((polar[1] / math.pi) * 180, 1, 0.4 + 0.6 * v * q1)
    
    @staticmethod
    def kart_to_polar(x: float, y: float) -> tuple:
        r = abs(x + y * 1j)
        return r, 0 if r == 0 else (math.acos(x / r) if y >= 0 else (2 * math.pi - math.acos(x / r)))
    
    def rescale(self, i, l):
        if self.squish:
            return (i - self.resolution / 2) / self.resolution * self.distance, \
                   (l - int(self.resolution * self.format[1] / self.format[0]) / 2) / int(self.resolution * self.format[1] / self.format[0]) * self.distance
        return (i - self.resolution / 2) / self.resolution * self.distance, \
               (l - self.resolution / 2) / self.resolution * self.distance + self.distance/6
    
    @staticmethod
    def color_correct(image):
        return np.rot90(1 - (1 / (image + 1)))
    
    @staticmethod
    def query(x: float, y: float, contour: bool = False, angle: bool = True) -> tuple:
        polar = Plotter.kart_to_polar(x, y)
        if contour:
            return Plotter.contour_plot(polar, angle)
        return Plotter.hsv_to_rgb((polar[1] / math.pi) * 180, 1, polar[0])
    
    def plot_func(self, function: callable, contour: bool = False, angle: bool = False) -> np.ndarray:
        image = np.ndarray((int(self.resolution), int(self.resolution * self.format[1] / self.format[0]), 3))
        for i in range(image.shape[0]):
            for l in range(int(image.shape[1])):
                try:
                    i_tmp, l_tmp = self.rescale(i, l)
                    num = (i_tmp + l_tmp * 1j)
                    res = function(num)
                    image[i, l] = self.query(res.real, res.imag, contour, angle)
                except OverflowError:
                    image[i, l] = 0
                    print("Overflow (i =" + str(i) + ", l = " + str(l) + ")")
                except ZeroDivisionError:
                    image[i, l] = 0
                    print("Zero division (i =" + str(i) + ", l = " + str(l) + ")")
        return self.color_correct(image) if not contour else np.rot90(image)


if __name__ == "__main__":
    """default: resolution=4096, distance=5"""
    p = Plotter(resolution=1024, distance=3.5)
    # plot1 = p.plot_func(lambda x: sum([i**(-x) for i in range(1, 1000)]))
    # plot1 = p.plot_func(lambda x: (x**2)*np.sin(1/x) if x.real ** 2 + x.imag ** 2 != 0 else 0)
    
    """Sierpinski Curves"""
    # func = lambda x: x**2-1/(16 * (x**2)) if x.real ** 2 + x.imag ** 2 != 0 else 0
    # func = lambda x: (x**2)-0.593/(2*x**2) if x.real ** 2 + x.imag ** 2 != 0 else 0
    # plot1 = p.plot_func(lambda x: func(func(func(func(func(func(func(x))))))))
    """Collatz Problem"""
    # func = lambda x: 1/4 * (2 + 7*x - (2 + 5*x)*np.cos(np.pi*x))
    # plot1 = p.plot_func(lambda x: rec(func, x, 2), True, True)
    """Fatou"""
    # func = lambda x: (1+0.2j)*np.sin(x)
    # # plot1 = p.plot_func(lambda x: rec(func, x, 17))
    """Euklid"""
    # a, b = -1, 1
    # func = lambda x: ((x-a)/(x-b))**2 if abs((x-b)) != 0 else 0
    # plot1 = p.plot_func(lambda x: rec(func, x, 3))
    """Pringsheim"""
    # func = lambda x, r: x/(1-x+r) if abs(1-x+r) != 0 else 0
    # plot1 = p.plot_func(lambda x: func(x, func(x, func(x, func(x, func(x, func(x, func(x, func(x, func(x, func(x,
    #                     func(x, func(x, func(x, func(x, func(x, func(x, func(x, func(x, func(x, func(x, 0)))))))))))))))))))), True, True)
    """Halmos"""
    # a = [np.exp(0*2*np.pi*1j), np.exp(0.2*2*np.pi*1j), np.exp(0.4*2*np.pi*1j), np.exp(0.6*2*np.pi*1j), np.exp(0.8*2*np.pi*1j)]
    # s = [20/5, 16/5, 20/5, 16/5, 20/5]
    # func = lambda x: prod([((a[i]-x)/(a[i]+x))**s[i] if abs(a[i]+x) != 0 else 0 for i in range(0, 5)])
    # plot1 = p.plot_func(func, True, True)
    """Jacobi"""
    # q = 0.5+0.5j
    # exp_func = lambda x: np.exp(2*np.pi*1j*x)
    # func = lambda x: sum([(q**(i**2))*x**i if x.real != 0 or x.imag != 0 else 0 for i in range(-20, 20)])
    # plot1 = p.plot_func(func)
    """riemann zeta"""
    # func = lambda x: sum([1/(n**x) for n in range(1, 10)])
    # func = lambda x: sum([np.e**(-x*np.log(n)) for n in range(1, 10)])
    # plot1 = p.plot_func(func, True, True)
    """the eye of the eye (own)"""
    # func = lambda x: sum([x**n for n in range(0, 20)])
    # plot1 = p.plot_func(lambda x: func(func(x)))
    """color correction at it's best"""
    # func = lambda x: 1 - 1 / x
    # plot1 = p.plot_func(func, True, True)
    """mandelbrot"""
    # func = lambda prev, x: prev ** 2 + x
    # plot1 = p.plot_func(lambda x: func(func(func(func(func(func(0, x), x), x), x), x), x), True, True)
    """gamma"""
    # plot1 = p.plot_func(scipy.special.gamma, False, False)
    """func1"""
    # func = lambda x: (x+4)/(x**5-3*1j*x**3+2)
    # plot1 = p.plot_func(func, True, True)
    """Hyper-exponentiation"""
    # func = lambda x: x**(x**x)
    # plot1 = p.plot_func(func, True, True)
    """Riesz"""
    fac = lambda x: prod([i for i in range(1, x + 1)])
    rz = lambda x: sum([1 / (n ** x) for n in range(1, 10)])
    func = lambda x: sum([(((-1) ** (n + 1)) / (fac(n - 1) * rz(2 * n))) * x ** n for n in range(1, 20)])
    plot1 = p.plot_func(func, True, False)
    """Ramanujan"""
    # func = lambda x: 1+sum([(x**(n**2))/prod([(1+x**i)**2 for i in range(1, n+1)]) for n in range(0, 10)])
    # plot1 = p.plot_func(func, True, True)
    """Binomialkoeffizient"""
    # k = 5
    # func = lambda x: prod([(x+1-i)/i for i in range(1, k)])
    # plot1 = p.plot_func(lambda x: rec(func, x, 3), True, True)
    """Stirling # approximates x!"""
    # func = lambda x: np.sqrt(2*np.pi*x) * (x/np.e)**x
    # plot1 = p.plot_func(func, True, True)
    """Burnside # approximates x!"""
    # func = lambda x: np.sqrt(2 * np.pi) * (x+1/2)**(x+1/2)*np.e**(-1*(x+1/2))
    # plot1 = p.plot_func(func, True, True)
    
    plt.imshow(plot1, interpolation='bicubic', origin="upper")
    plt.imsave("image.png", plot1)
    plt.show()
