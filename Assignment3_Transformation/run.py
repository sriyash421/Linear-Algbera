import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3, suppress=True)
dets = []
cond_nos = []

def get_3dpoints() :
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    x_ = x.flatten()
    y_ = y.flatten()
    z_ = z.flatten()
    points = np.array([np.array([x_[i],y_[i],z_[i]]) for i in range(x_.shape[0])])
    return points, x, y, z

def get_2dpoints() :
    u = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(u)
    y = np.sin(u)
    points = np.array([np.array([x[i],y[i]]) for i in range(x.shape[0])])
    return points

def get_condition_no(T) :
    return np.linalg.cond(T)

def get_determinant(T) :
    return np.linalg.det(T)

def transform3D_2D(fname, T) :
    points, x,y,z = get_3dpoints()
    transformed_points = np.matmul(points, T.T)
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x,y,z, color='b')
    ax.set_title("Unit sphere")
    
    ax = fig.add_subplot(122)
    ax.plot(transformed_points[:,0],transformed_points[:,1], color='b')
    ax.set_title("Tranformed ellipse")
    ax.axis('equal')
    
    text = f"Transformation Matrix =\n{T}\nCondition Number = {get_condition_no(T)}"
    plt.suptitle(text, fontsize=15)
    plt.figtext(0.1, 0.9, chr(fname+96)+".", fontsize=20)
    plt.tight_layout(pad=5)
    plt.savefig(str(fname))

def transform2D_3D(fname, T) :
    points = get_2dpoints()
    transformed_points = np.matmul(points, T.T)
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    ax.plot(points[:,0],points[:,1], color='b')
    ax.set_title("Unit circle")
    ax.axis('equal')
    
    ax = fig.add_subplot(122, projection='3d')
    x = transformed_points[:,0]
    y = transformed_points[:,1]
    z = transformed_points[:,2]
    ax.plot(x, y, z, color='b')
    ax.set_title("Tranformed ellipsoid")
    
    text = f"Transformation Matrix =\n{T}\nCondition Number = {get_condition_no(T)}"
    plt.suptitle(text, fontsize=15)
    plt.figtext(0.1, 0.9, chr(fname+96)+".", fontsize=20)
    plt.tight_layout(pad=5)
    plt.savefig(str(fname))

def transform2D_2D(fname, T) :
    points = get_2dpoints()
    transformed_points = np.matmul(points, T.T)
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    ax.plot(points[:,0], points[:,1], color='b')
    ax.set_title("Unit circle")
    ax.axis('equal')
    
    ax = fig.add_subplot(122)
    ax.plot(transformed_points[:,0],transformed_points[:,1], color='b')
    ax.set_title("Tranformed ellipse")
    ax.axis('equal')
    determinant = get_determinant(T)
    cond_no = get_condition_no(T)
    
    text = f"Transformation Matrix =\n{T}\nCondition Number = {cond_no}\nInvertible = {determinant!=0}\nDeterminant = {determinant}"
    plt.suptitle(text, fontsize=15)
    plt.figtext(0.1, 0.9, chr(fname+96)+".", fontsize=20)
    plt.tight_layout(pad=5)
    plt.savefig(str(fname))
    
    dets.append(determinant)
    cond_nos.append(cond_no)

T_2d = [np.array([[1,0.9],[0.9,0.8]]), np.array([[1,0],[0,-10]])]
T_2d.extend([np.array([[1,1],[1,i]]) for i in [10,5,1,0.1,0.01,0.0001,0]])

transform2D_3D(1,np.array([[-1/(2**(1/2.0)), 0], [0, -1/(2**(1/2.0))], [-1, 1]]))
transform3D_2D(2,np.array([[-2, 1, 2], [0, 2, 0]]))

for i, T in enumerate(T_2d) :
    transform2D_2D(i+3, T)

fig = plt.figure(figsize=(20,10))
# fig.suptitle("determinant value vs condition number", fontsize=20, fontsize=15)
plt.scatter(dets, cond_nos)
plt.xlabel("Determinant")
plt.ylabel("Condition Number")
plt.savefig("line")