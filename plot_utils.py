import matplotlib.pyplot as plt
import numpy as np


def plot_1d(coasts, forests, streets, bins=100):
    plt.figure()
    plt.hist([coasts, forests, streets], bins, alpha=0.5, label=["coasts", "forests", "streets"], color=["red", "green", "blue"])


def plot_2d(coasts, forests, streets):
    ax = plt.axes(projection='3d')
    ax.scatter(coasts, forests, streets)