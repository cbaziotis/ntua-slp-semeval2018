import numpy


def plot_line(viz, vals, plot_name, legend):
    """
    Plot a line plot using visdom
    Args:
        viz (Visdom): visdom pointer
        vals (ndarray): a list of line data points to plot
        plot_name (str): the name of the plot
        legend (list): a list of string names, corresponding to each line

    Returns:

    """

    # squeeze the values if there is only one line
    if len(vals.shape) > 1 and vals.shape[-1] > 1:
        vals = vals.squeeze()

    # calculate the steps
    steps = [i for i in range(1, vals.shape[-1] + 1)]

    # initial plot
    if vals.shape[-1] == 2:
        viz.line(
            X=numpy.array(steps),
            Y=numpy.transpose(vals),  # wtf visdom logic
            win=plot_name,
            opts=dict(title=plot_name, legend=legend))

    # update plot
    elif vals.shape[-1] > 2:
        _y = [vals[-1]] if len(vals.shape) == 1 else [vals[:, -1]]
        _x = [vals.shape[-1]] if len(vals.shape) == 1 \
            else [numpy.repeat([vals.shape[-1]], vals.shape[0], axis=0)]
        viz.line(
            X=numpy.array(_x),
            Y=numpy.array(_y),
            win=plot_name, update="append")
