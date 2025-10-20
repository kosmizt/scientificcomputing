import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def reta(a,b):
    """para uma equação da forma y=ax+b"""
    x = np.linspace(-10, 10)
    y = a*x + b

    plt.plot(x,y, color='m')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

def parabola(a,b,c):
    """para uma equação da forma y=ax^2+bx+c"""
    x = np.linspace(-10, 10) 
    y = a*(x**2) + b*x + c

    plt.plot(x,y, color='m')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

def plano(a,b,c,d):
    """para uma equação da forma ax+by+cz+d=0"""
    x = np.linspace(-10,10)
    y = np.linspace(-10,10)

    x, y = np.meshgrid(x, y)
    z = (-a*x - b*y - d)/(c+1e-10) #evitar divisão por zero

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d') 
    ax.plot_surface(x,y,z, color='m', alpha=0.5) 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def epi_hipo(R, r, k):
    """R = raio do maior círculo
    r = raio do menor círculo
    k = repetições
    desenha epiciclóides ou hipociclóides (quando r<0).
    para desenhar hipotrocóides, expandir o ciclo para 720 para melhor visualização e alternar alguns sinais da parametrização:
    x = (R - r) * np.cos(np.radians(s)) + r * np.cos(np.radians((R - r)) / r) * s)
    y = (R - r) * np.sin(np.radians(s)) - r * np.sin(np.radians((R - r)) / r) * s)"""
    if r < 0:
        print("hipociclóide")
    else:
        print("epiciclóide")

    if r > 0 and R == r:
        print("cardióide")
    elif r > 0 and R/2 == r:
        print("nefróide")

    if r<0 and R/2 == -r:
        print("par de al-Tusi")
    elif r<0 and R/3 == -r:
        print("deltóide")
    elif r<0 and R/4 == -r:
        print("astróide")
    elif r<0 and R/5 == -r:
        print("pentóide")
    elif r<0 and R/6 == -r:
        print("hexóide")

    circumference = k * 360 
    x_points = []
    y_points = []

    for s in range(0, circumference):
        x = (R + r) * np.cos(np.radians(s)) - r * np.cos(np.radians(((R + r) / r) * s))
        y = (R + r) * np.sin(np.radians(s)) - r * np.sin(np.radians(((R + r) / r) * s))
        x_points.append(x)
        y_points.append(y)

    rc = plt.Line2D((0, 0), (0, 0), linewidth=1, color="k") #raio do circulo exterior
    circler = plt.Circle((0, 0), r, color='#e5047a', fill=False) #círculo exterior (cuja trajetória marca a curva)
    circleR = plt.Circle((0, 0), R, color='#5b2eb7', fill=False) #círculo interior
    punct = plt.Circle((0, 0), r / 10.0, color="k") 
    fig, ax = plt.subplots()
    ax.set_xlim(-5 * (R + r), 5 * (R + r))  
    ax.set_ylim(-5 * (R + r), 5 * (R + r))
    line, = ax.plot([], [], color='#166b28')
    ax.add_artist(rc)
    ax.add_artist(circler)
    ax.add_artist(circleR)
    ax.add_artist(punct)
    ax.add_artist(line)

    def init(): 
        circler.center = (0, 0)
        punct.center = (0, 0)
        ax.add_patch(circler)
        ax.add_patch(punct)
        return circler, punct

    def animate(i):
        x = (r + R) * np.cos(np.radians(i))
        y = (r + R) * np.sin(np.radians(i))
        xi = x_points[i]
        yi = y_points[i]
        rc.set_data((x, xi), (y, yi))
        circler.center = (x, y)
        punct.center = (xi, yi)
        line.set_data(x_points[:i], y_points[:i])
        return circler, punct, rc, line

    anim = FuncAnimation(fig, animate, init_func=init, frames=circumference, interval=1, blit=True)

    plt.grid()
    plt.show()

def espiral_log(a, k, p, n):
    """"a e k da equação polar r=a*e^(k*phi), p como número de pontos, n para os eixos
    desenha uma espiral logarítmica (náutilo)"""

    phi = np.linspace(0, 4*np.pi, p)
    x = a*np.exp(k*phi) * np.cos(phi)
    y = a*np.exp(k*phi) * np.sin(phi)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='#e5047a')
    ax.set_aspect('equal')
    ax.set_xlim(-n, n)
    ax.set_ylim(-n, n)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        xdata = x[:i]
        ydata = y[:i]
        line.set_data(xdata, ydata)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=p, interval=0.1, blit=True)

    plt.grid()
    plt.show()

def espiral_hiper(a, k, p, n):
    """uma espiral hiperbólica (galáctica):
    x = a*(cos(phi))/phi
    y = a*(sin(phi))/phi"""

    phi = np.linspace(0, 4*np.pi, p)
    x = a*((np.cos(phi))/(phi+1e-10))
    y = a*((np.sin(phi))/(phi+1e-10))

    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='#ec0ed1')
    ax.set_aspect('equal')
    ax.set_xlim(-n, n)
    ax.set_ylim(-n, n)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        xdata = x[:i]
        ydata = y[:i]
        line.set_data(xdata, ydata)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=p, interval=0.1, blit=True)

    plt.grid()
    plt.show()

def lituus_spiral(a, p, xlim, ylim):
    """uma curva cujo angulo t é inversamente proporcional ao quadrado do raio"""
    t = np.linspace(0.01, 4 * np.pi, p)
    r = a/(np.sqrt(t))
    x = r*np.cos(t)
    y = r*np.sin(t)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='#dd57a6')
    ax.set_aspect('equal')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        xdata = x[:i]
        ydata = y[:i]
        line.set_data(xdata, ydata)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=p, interval=5, blit=True)

    plt.grid()
    plt.show()

def lemniscata(a):
    """lemniscata de Bernoulli, a curva algébrica de (x^2+y^2)^2=a^2(x^2-y^2)"""
    def plot_lemniscata(a):
        t = np.linspace(0, 2*np.pi, 1000)
        x = a*np.cos(t)/(1 + np.sin(t)**2)
        y = a*np.cos(t)*np.sin(t)/(1 + np.sin(t)**2)

        return x, y

    x_points, y_points = plot_lemniscata(a)

    fig, ax = plt.subplots()
    ax.set_xlim(-1.5*a, 1.5*a)
    ax.set_ylim(-1.5*a, 1.5*a)
    line, = ax.plot([], [], color='#5b2eb7')

    def init():  
        line.set_data([], [])
        return line,

    def animate(i): 
        line.set_data(x_points[:i], y_points[:i])
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_points), interval=3, blit=True)

    plt.grid()
    plt.show()

def rosa(a,k,f):
    """desenha uma rosa polar de k pétalas (simétrica) (r(t)= a*cos(kt+f))
    parametrização:
    x = a*cos(kt)*cos(t)
    y = a*cos(kt)*sint(t)
    adiciona-se f para a sua rotação (em radianos, experimentar np.pi/4)


    k = 2 -> quadrifólio. a curva algébrica de (x^2+y^2)^2=a^2(x^2-y^2)
    k = 3 -> trifólio. a curva algébrica de (x^2+y^2)^2=a(x^3-3xy^2)
    (k ímpar retorna possuem k pétalas, k par possuem 2k pétalas)"""
    def plot(a,k,f):
        t = np.linspace(0, 2*np.pi, 1000)
        x = a*np.cos(t)*np.cos(k*(t+f))
        y = a*np.sin(t)*np.cos(k*(t+f))

        return x, y

    x_points, y_points = plot(a,k,f)

    fig, ax = plt.subplots()
    ax.set_xlim(-1.5*a, 1.5*a)
    ax.set_ylim(-1.5*a, 1.5*a)
    line, = ax.plot([], [], color='#f94105')

    def init():  
        line.set_data([], [])
        return line,

    def animate(i): 
        line.set_data(x_points[:i], y_points[:i])
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_points), interval=3, blit=True)

    plt.grid()
    plt.show()

def lissajous(a,b,A,B,d):
    """curva de Lissajous: a curva de parametrização:
    x = A sin(a*t + d); sendo d um ângulo de rotação
    y = B sin(b*t)"""
    def plot_lissajous(a,b,A,B,d):
        t = np.linspace(0, 2*np.pi, 1000)
        x = A*(np.sin(a*t+d))
        y = B*(np.sin(b*t))

        return x, y

    x_points, y_points = plot_lissajous(a,b,A,B,d)

    fig, ax = plt.subplots()
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    line, = ax.plot([], [], color='#530c74')

    def init():  
        line.set_data([], [])
        return line,

    def animate(i): 
        line.set_data(x_points[:i], y_points[:i])
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_points), interval=3, blit=True)

    plt.grid()
    plt.show()

def helix(a, b):
    """para uma hélice parametrizada
    x(t) = a*cos(t)
    y(t) = a*sin(t)
    z(t) = b*t"""
    t = np.linspace(0, 10*np.pi,1000)

    def update(frame): 
        t_frame = t[:frame]
        x = a*np.cos(t_frame)
        y = a*np.sin(t_frame)
        z = b*t_frame

        ax.clear()
        ax.plot(x, y, z, color='#9020b2')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(t), repeat=False, interval=2)

    plt.show()

def conhelix(r, a):
    """uma espiral cônica, de parametrização.
    uma forma de espiral de Pappus
    x = t*r*cos(a*t)
    y = t*r*sin(a*t)
    z = t"""
    t = np.linspace(0, 10, 1000)

    def update(frame):
        t_frame = t[:frame+1]
        x = t_frame*r*np.cos(a*t_frame)
        y = t_frame*r*np.sin(a*t_frame)
        z = t_frame

        ax.clear()
        ax.plot(x, y, z, lw=2, color='#2999ae')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=len(t), repeat=False, interval=1)
    plt.show()

def butterfly(p, xlim, ylim):
    """a curva borboleta transcedental
    p = numero de pontos.
    xlim, ylim = dimensão para plotar"""
    t = np.linspace(0, 30*np.pi, p)
    x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4*t) - np.sin(t/12)**5)
    y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4*t) - np.sin(t/12)**5)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='#1f9994')

    ax.set_aspect('equal')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        xdata = x[:i]
        ydata = y[:i]
        line.set_data(xdata, ydata)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=p, interval=10, blit=True)

    plt.grid()
    plt.show()

def viviani():
    """la finestra di viviani è una curva ottenuta dall'intersezione di una sfera con un cilindro, con lo raggio dalla sfera uguale allo diametro del cilindro."""
    def grid(ax):
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    def update(frame, curve, ball):
        ball.set_data([curve._verts3d[0][frame]], [curve._verts3d[1][frame]])
        ball.set_3d_properties([curve._verts3d[2][frame]])
        return ball,
    
    def plot_combined():
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        
        curve = plot_curve(ax)
        grid(ax)
        ball, = ax.plot([curve._verts3d[0][0]], [curve._verts3d[1][0]], [curve._verts3d[2][0]], 'bo')
        ani = animation.FuncAnimation(fig, update, frames=len(curve._verts3d[0]), fargs=(curve, ball), blit=True, interval=10, repeat=False)
        
        plt.show()
    
    def plot_curve(ax):
        a = 1
        n = 100
        t = np.linspace(0, 4 * np.pi, n)
        x = a * (1 + np.cos(t))
        y = a * np.sin(t)
        z = 2 * a * np.sin(t / 2)
        ng = n // 2 + 1
        theta = t[:ng]
        
        for phi in np.linspace(-np.pi, np.pi, 32):
            ax.plot(2 * a * np.sin(theta) * np.cos(phi), 2 * a * np.cos(theta) * np.cos(phi),
                    2 * a * np.sin(phi), 'purple', alpha=0.2, lw=0.5)
            ax.plot(2 * a * np.cos(theta) * np.cos(phi), [2 * a * np.sin(phi)] * ng,
                    2 * a * np.sin(theta) * np.cos(phi), 'purple', alpha=0.2, lw=0.5)
            ax.plot(a * np.sin(theta) + a, a * np.cos(theta), 2 * a * phi / np.pi,
                    'purple', alpha=0.2, lw=1)
        
        curve, = ax.plot(x, y, z, 'k', lw=3)
        return curve
    
    plot_combined()

