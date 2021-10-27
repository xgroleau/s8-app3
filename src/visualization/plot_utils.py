import matplotlib.pyplot as plt
import numpy as np


def plot_1d(coasts, forests, streets, bins=100):
    plt.figure()
    plt.hist([coasts, forests, streets], bins, alpha=0.5, label=["coasts", "forests", "streets"], color=["blue", "green", "red"])

def plot_2d(coasts, forests, streets):
    ax = plt.axes(projection='3d')
    ax.scatter(coasts, forests, streets)

def plot_3d(coasts, forests, streets):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(coasts[:,0], coasts[:,1], coasts[:,2], alpha=0.5, label="coasts", color="blue")
    ax.scatter(forests[:,0], forests[:,1], forests[:,2], alpha=0.5, label="forests", color="green")
    ax.scatter(streets[:,0], streets[:,1], streets[:,2], alpha=0.5, label="streets", color="red")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
