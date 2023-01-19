import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf


def metric_plot(metric, PATH):
    figplot = plt.figure()
    plt.plot(metric)
    plt.grid(which='both', axis="both")
    plt.title("Metric evolution / integral=" + str(np.sum(metric)))
    plt.set_ylabel("Metric")
    plt.set_xlabel("Step")
    figplot.savefig(PATH)


def plotlib_draw(yTest_l, yPred_l, PATH, show=False):
    plt.figure()
    legend = []
    for ind, ((_, y_test), y_pred) in enumerate(zip(yTest_l, yPred_l)):
        plt.plot(y_test)
        plt.plot(y_pred)
        name1 = "Y_test" + str(ind)
        name2 = "Y_pred" + str(ind)
        legend.append(name1)
        legend.append(name2)
    plt.legend(legend)
    if show == True:
        plt.show()


def plotly_draw(yTest_l,  yPred_l, metric, PATH, show=False):
    fig = go.Figure()
    abscisse = np.linspace(1, np.size(np.array(yPred_l[0])),
                           np.size(np.array(yPred_l[0])))
    for ind, (y_test, y_pred) in enumerate(zip(yTest_l, yPred_l)):
        name1 = "Y_test" + str(ind)
        name2 = "Y_pred" + str(ind)
        fig.add_trace(go.Scatter(x=abscisse, y=np.array(y_pred).flatten(),
                                 mode='lines',
                                 name=name2))
        fig.add_trace(go.Scatter(x=abscisse, y=np.array(y_test).flatten(),
                                 mode='lines',
                                 name=name1))

    fig.add_trace(go.Scatter(x=abscisse, y=np.array(metric).flatten(),
                             mode='lines',
                             name="Metric"))
    if show == True:
        fig.show()
    fig.write_html(PATH)

