import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt, animation
from matplotlib.colors import ListedColormap
from numpy import linspace
from copy import deepcopy

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16,
                     'font.serif': "Times New Roman",
                     'font.family': "serif",
                     "figure.titlesize": 24,
                     "axes.titlesize": 24,
                     "axes.labelsize": 20,
                     "savefig.pad_inches": 0,
                     "legend.fontsize": 16,
                     "axes.labelpad": 1,
                     "text.color": "black",
                     "axes.labelcolor": "black",
                     "xtick.color": "black",
                     "ytick.color": "black",
                     "legend.framealpha": 0.9,
                     "figure.dpi": 400,
                     "savefig.dpi": 400,
                     "patch.facecolor": "white"})

plot_colors = ["#ca0020", "#f4a582", "#ffffff", "#92c5de", "#0060b0"]


def plot_2d(model, x_param, y_param, fixed_params=None, param_ranges=None, x_and_y_steps=100):
    """
    Plot a 2d slice from the model.
    Params:
        model: the model to use for predictions.
        x_param: the varying parameter to plot on the x axis.
        y_param: the varying parameter to plot on the y axis.
        fixed_params: values the other parameters are fixed at.
        param_ranges: ranges through which x_param and y_param will be varied.
        x_and_y_steps: the size of the heatmap will be x_and_y_steps X x_and_y_steps.
    """
    plot_fixed_params = deepcopy(model.default_params)
    plot_param_ranges = deepcopy(model.param_ranges)
    if fixed_params:
        for key in fixed_params:
            if key not in plot_fixed_params:
                print(f"\"{key}\" not a valid parameter name. Ignoring.")
            else:
                plot_fixed_params[key] = fixed_params[key]
    if param_ranges:
        for key in param_ranges:
            if key not in plot_param_ranges:
                print(f"\"{key}\" not a valid parameter name. Ignoring.")
            else:
                plot_param_ranges[key] = param_ranges[key]

    # Inconveniently, the GPs were trained on sigma, instead of actual interaction distance, so we need to divide by 3 before the GP can predict points.
    plot_fixed_params["Interaction distance"] = plot_fixed_params["Interaction distance"] / 3
    plot_param_ranges["Interaction distance"] = (plot_param_ranges["Interaction distance"][0] / 3, plot_param_ranges["Interaction distance"][1] / 3)

    param_names = [k for k, v in plot_fixed_params.items()]
    x_index = param_names.index(x_param)
    y_index = param_names.index(y_param)
    x_vals = linspace(plot_param_ranges[x_param][0], plot_param_ranges[x_param][1], x_and_y_steps)
    y_vals = linspace(plot_param_ranges[y_param][0], plot_param_ranges[y_param][1], x_and_y_steps)

    # Assemble points to predict:
    points_to_predict = []
    for x in range(x_and_y_steps):
        for y in range(x_and_y_steps):
            cur = [v for k, v in plot_fixed_params.items()]
            cur[x_index] = x_vals[x]
            cur[y_index] = y_vals[y]
            points_to_predict.append(cur)

    # Get predictions from model.
    data = pd.DataFrame(points_to_predict, columns=param_names)
    mean, lower, upper = model.predict(data, drive_cols=False)
    mean, lower, upper = mean.numpy(), lower.numpy(), upper.numpy()
    if model.model_type  == "composite":
        # Shift predictions from [-1, 1] to [0, 1].
        mean += 1
        mean /= 2
        lower += 1
        lower /= 2
        upper += 1
        upper /= 2

    # Arrange predictions for plotting.
    plot_data = [[-1 for i in range(x_and_y_steps)] for i in range(x_and_y_steps)]
    for i in range(len(mean)):
        if upper[i] < 0.5:
            # If upper of 95%CI is < 0.5, the model is confident of failure.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 0
        elif lower[i] > 0.5:
            # If lower of 95%CI is > 0.5, the model is confident of success.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 1
        elif mean[i] > 0.5:
            # Less than 95% confident prediction of success.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 0.75
        else:
            # Less than 95% confident prediction of failures.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 0.25

    # Plot:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.set_tight_layout({"pad":0.1, "w_pad":0.0, "h_pad":0.0})
    plot_cmap = ListedColormap(plot_colors)
    ax.pcolormesh(plot_data, cmap=plot_cmap, rasterized=True, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Convert back from  sigma to interaction distance.
    plot_param_ranges["Interaction distance"] = (plot_param_ranges["Interaction distance"][0] * 3, plot_param_ranges["Interaction distance"][1] * 3)

    ax.set_xticks(linspace(0, x_and_y_steps, 6))
    ax.set_yticks(linspace(0, x_and_y_steps, 6))
    ax.set_xlabel(x_param)
    ax.set_xticklabels([f"{i:.2f}" for i in linspace(plot_param_ranges[x_param][0], plot_param_ranges[x_param][1], 6)])
    ax.set_ylabel(y_param)
    ax.set_yticklabels([f"{i:.2f}" for i in linspace(plot_param_ranges[y_param][0], plot_param_ranges[y_param][1], 6)])
    ax.set_title(f"{model.drive_type}{' with resistance' if model.resistance_simulated else ' without resistance'}")


def animated_plot(model, x_param, y_param, z_param, fixed_params=None, param_ranges=None, x_and_y_steps=100, z_steps=10):
    """
    Plot an animated heatmap, where two parameters are displayed as a third parameter is varied.
    Params:
        model: the model to use for predictions.
        x_param: the varying parameter to plot on the x axis.
        y_param: the varying parameter to plot on the y axis.
        z_param: the parameter that is varied through the course of the animation.
        fixed_params: values the other parameters are fixed at.
        param_ranges: ranges through which x_param and y_param will be varied.
        x_and_y_steps: the size of the heatmap will be x_and_y_steps X x_and_y_steps.
        z_steps: the number of steps for the time parameter.
    """
    plot_fixed_params = deepcopy(model.default_params)
    plot_param_ranges = deepcopy(model.param_ranges)
    if fixed_params:
        for key in fixed_params:
            if key not in plot_fixed_params:
                print(f"\"{key}\" not a valid parameter name. Ignoring.")
            else:
                plot_fixed_params[key] = fixed_params[key]
    if param_ranges:
        for key in param_ranges:
            if key not in plot_param_ranges:
                print(f"\"{key}\" not a valid parameter name. Ignoring.")
            else:
                plot_param_ranges[key] = param_ranges[key]

    # Inconveniently, the GPs were trained on sigma, instead of actual interaction distance, so we need to divide by 3 before the GP can predict points.
    plot_fixed_params["Interaction distance"] = plot_fixed_params["Interaction distance"] / 3
    plot_param_ranges["Interaction distance"] = (plot_param_ranges["Interaction distance"][0] / 3, plot_param_ranges["Interaction distance"][1] / 3)

    z_vals = linspace(plot_param_ranges[z_param][0], plot_param_ranges[z_param][1], z_steps)
    # Plot:
    fig = plt.figure(figsize=(10, 10))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0])
    ax.set_aspect('equal')
    fig.suptitle(f"{model.drive_type}{' w/ resistance' if model.resistance_simulated else ' w/o resistance'}", y=0.95)
    return animation.FuncAnimation(fig, animate, frames=z_steps, interval=8000/z_steps,
            fargs=(ax, model, x_param, y_param, z_param, plot_fixed_params, plot_param_ranges, x_and_y_steps, z_vals))


def animate(frame_number, ax, model, x_param, y_param, z_param, plot_fixed_params, plot_param_ranges, x_and_y_steps, z_vals):
    """
    This function is called by the animation.
    Updates the animation each frame.
    """
    ax.clear()
    param_names = [k for k, v in plot_fixed_params.items()]
    x_index = param_names.index(x_param)
    y_index = param_names.index(y_param)
    x_vals = linspace(plot_param_ranges[x_param][0], plot_param_ranges[x_param][1], x_and_y_steps)
    y_vals = linspace(plot_param_ranges[y_param][0], plot_param_ranges[y_param][1], x_and_y_steps)

    # Put the current value of the ze parameter in the "fixed parameters" for this frame.
    plot_fixed_params[z_param] = z_vals[frame_number]

    # Assemble points to predict:
    points_to_predict = []
    for x in range(x_and_y_steps):
        for y in range(x_and_y_steps):
            cur = [v for k, v in plot_fixed_params.items()]
            cur[x_index] = x_vals[x]
            cur[y_index] = y_vals[y]
            points_to_predict.append(cur)

    # Get predictions from model.
    data = pd.DataFrame(points_to_predict, columns=param_names)
    mean, lower, upper = model.predict(data, drive_cols=False)
    mean, lower, upper = mean.numpy(), lower.numpy(), upper.numpy()
    if model.model_type  == "composite":
        # Shift predictions from [-1, 1] to [0, 1].
        mean += 1
        mean /= 2
        lower += 1
        lower /= 2
        upper += 1
        upper /= 2

    # Arrange predictions for plotting.
    plot_data = [[-1 for i in range(x_and_y_steps)] for i in range(x_and_y_steps)]
    for i in range(len(mean)):
        if upper[i] < 0.5:
            # If upper of 95%CI is < 0.5, the model is confident of failure.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 0
        elif lower[i] > 0.5:
            # If lower of 95%CI is > 0.5, the model is confident of success.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 1
        elif mean[i] > 0.5:
            # Less than 95% confident prediction of success.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 0.75
        else:
            # Less than 95% confident prediction of failures.
            plot_data[i % x_and_y_steps][i // x_and_y_steps] = 0.25

    # Plot:
    plot_cmap = ListedColormap(plot_colors)
    ax.pcolormesh(plot_data, cmap=plot_cmap, rasterized=True, vmin=0, vmax=1)
    ax.set_aspect('equal')

    # Convert back from  sigma to interaction distance.
    plot_param_ranges["Interaction distance"] = (plot_param_ranges["Interaction distance"][0] * 3, plot_param_ranges["Interaction distance"][1] * 3)

    ax.set_xticks(linspace(0, x_and_y_steps, 6))
    ax.set_yticks(linspace(0, x_and_y_steps, 6))
    ax.set_xlabel(x_param)
    ax.set_xticklabels([f"{i:.2f}" for i in linspace(plot_param_ranges[x_param][0], plot_param_ranges[x_param][1], 6)])
    ax.set_ylabel(y_param)
    ax.set_yticklabels([f"{i:.2f}" for i in linspace(plot_param_ranges[y_param][0], plot_param_ranges[y_param][1], 6)])
    ax.set_title(f"  {z_param} = {z_vals[frame_number]:.3f}", loc='left', fontdict={'verticalalignment':'top'}, fontsize=16)


def hplot(names, first, first_conf, total, total_conf, ax):
    """
    Plot a horizontal bar graph of first order and total order effects.
    """
    y_pos = [i for i in range(len(names))]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    plt.setp(ax.get_yticklabels(), fontsize=16)
    xt = [0,0.2,0.4,0.6,0.8,1.0]
    ax.set_xticks(xt)
    xlabels = [f"{i:.1f}" for i in xt]
    xlabels[0] = f" {xlabels[0]}"
    xlabels[-1] = f"{xlabels[-1]} "
    ax.set_xticklabels(xlabels)
    ax.barh(y_pos, total, height=0.4, xerr=total_conf, color=plot_colors[3])
    ax.barh(y_pos, first, height=0.4, xerr=first_conf, color=plot_colors[4])
    ax.set_xlim(left=0, right=1)
    ax.set_xlabel("Sensitivity index")


def hplot_second_order(names, vals, conf, ax):
    """
    Plot a horizontal bar graph of second order effects.
    """
    y_pos = [i for i in range(len(vals))]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    plt.setp(ax.get_yticklabels(), fontsize=16)
    ax.barh(y_pos, vals, height=0.4, xerr=conf)
    ax.set_xlim(left=0)
    ax.set_xlabel("Sensitivity index")


def fix_y_axis_strings(s):
    """
    Format strings more nicely for second order plots.
    """
    if type(s) is tuple:
        return (f"{s[0]} \\   \n{s[1]}")
    return s


def process_sa(sa, res):
    """
    Process data in sensitivity analysis to prepare for plotting.
    """
    if res:
        sa.columns = [f"{c} w/ resistance" for c in sa.columns]
    sa.columns = [c.lower().capitalize().replace('First order', 'First order effects') for c in sa.columns]
    conf_cols = sa.columns.str.contains('_conf')
    confs = sa.loc[:, conf_cols]
    confs.columns = [c.replace('_conf', '') for c in confs.columns]
    main_cols = sa.loc[:, ~conf_cols]
    main_cols.index = [fix_y_axis_strings(c) for c in main_cols.index]
    names = [i for i in main_cols.index]
    vs = pd.DataFrame(data=main_cols)
    vals = vs.to_numpy().tolist()
    v = []
    for i in vals:
        for j in i:
            v.append(j)
    conf = sa.to_numpy()
    conf= conf.tolist()
    conf = [i[1] for i in conf]
    return names, v, conf


def sa_plot(sensitivity_analysis, limit_num_second_order_bars=12):
    """
    Plots bar charts of a sensitivity analysis.
    """
    sa = sensitivity_analysis.copy()
    # Keep only the largest second order effects, as specified:
    if not limit_num_second_order_bars or limit_num_second_order_bars > len(sa[2]):
        limit_num_second_order_bars = len(sa[2])
    sa[2] = sa[2].nlargest(limit_num_second_order_bars, 'Second Order').sort_index()
    res = False
    if "Resistance rate" in sa[0].index:
        res = True

    # Assemble the graph for first order and total effects:
    names, first, first_confs = process_sa(sa[1], res)
    names, total, total_confs = process_sa(sa[0], res)
    size = (12,  0.75 * (len(sa[0]) + len(sa[2])))
    fig = plt.figure(figsize=size)
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig,
                          height_ratios=[len(sa[0]), len(sa[2])])
    fig.set_tight_layout({"pad":0.1, "w_pad":2.0, "h_pad":2.0})
    ax = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])
    hplot(names, first, first_confs, total, total_confs, ax)
    ax.set_title(sa[3])
    b_patch = mpatches.Patch(color=plot_colors[4], label="First order effects")
    lb_patch = mpatches.Patch(color=plot_colors[3], label="Total effects")
    ax.legend(handles=[b_patch,lb_patch], loc=1)

    # Assemble the graph for the second order effects:
    names, vals, conf = process_sa(sa[2], res)
    hplot_second_order(names, vals, conf, ax2)
    ax2.set_title('Second order effects')
