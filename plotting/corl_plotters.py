import numpy as np
import pickle
import datetime
from pathlib import Path
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['figure.figsize'] = 5.5, 7

font = {'size': 8}

matplotlib.rc('font', **font)

# yellow = [255, 225, 25]
# blue = [0, 130, 200]  # Viable
# lavender = [230, 190, 255]  # optimistic
# maroon = [128, 0, 0]  # cautious
# navy = [0, 0, 128]
# grey = [128, 128, 128]  # unviable, not yet failed
# red = [230, 25, 75]  # failed
# green = [60, 180, 75]
# lime = [210, 245, 60]
# mint = [170, 255, 195]
# red = [230, 25, 75]

# brewer2
dark_blue = [31, 120, 180]
light_blue = [166, 206, 227]

light_orange = [253, 205, 172]
light_purple = [203, 213, 232]

green = [115, 163, 72]
orange = [253, 174, 97]
yellow = [227, 198, 52]

optimistic_color = light_blue
explore_color = dark_blue
truth_color = green

failure_color = orange
unviable_color = yellow


def create_set_colormap():

    colors = np.array([
        unviable_color,  # Q_V_true      0 - 0
        failure_color,  # failure 1 - 1
        truth_color,  # Q_V_true      1 - 2
        optimistic_color,  # Q_V_safe       1 - 3
        explore_color   # Q_V_explore      1 - 4
    ])/256.0
    return ListedColormap(colors)


def frame_image(img, frame_width):
    b = frame_width  # border size in pixel
    # resolution / number of pixels in x and y
    ny, nx = img.shape[0], img.shape[1]
    if img.ndim == 3:  # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2:  # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img


def plot_Q_S(Q_V_true, Q_V_explore, Q_V_safe, S_M_0, S_M_true, grids,
            samples=None, failed_samples=None, Q_F=None, variance = None,
            max_variance = None):
    # TODO change S_true, simply have S as a tuple of Ss, and add names
    extent = [grids['actions'][0][0],
              grids['actions'][0][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]

    fig = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
    if variance is not None:
        gs = fig.add_gridspec(2, 2,  width_ratios=[3, 1],hspace = 0.1)
    else:
        gs = fig.add_gridspec(1, 2,  width_ratios=[3, 1])

    ax_Q = fig.add_subplot(gs[0, 0])
    ax_S = fig.add_subplot(gs[0, 1], sharey=ax_Q)
    ax_Q.tick_params(direction='in', top=True, right=True)
    ax_S.tick_params(direction='in', left=False)
    if variance is not None:
        ax_V = fig.add_subplot(gs[1, 0], sharex=ax_Q)
        ax_V.tick_params(direction='in', top=True, right=True)
        ax_colorbar = fig.add_subplot(gs[1,1])
        ax_colorbar.axis('off')

    # ax_S.plot(S, grids['states'][0])
    # ax_S.set_ylim(ax_Q.get_ylim())
    # s_max = np.max(S)

    # if S_true is not None:
    # ax_S.plot(S_true, grids['states'][0])
    # s_max = np.max([s_max, np.max(S_true)])
    # ax_S.legend(['safe estimate', 'ground truth'])
    # print(s_max)
    # ax_S.set_xlim((0, s_max + 0.1))



    c = tuple(c/256 for c in truth_color)
    #ax_S.plot(S_M_true, grids['states'][0],
    #          color=c)
    ax_S.fill_betweenx(grids['states'][0], 0, S_M_true, facecolor="none", hatch="--",
                       edgecolor=tuple(c / 256 for c in truth_color)
)

    c = tuple(c/256 for c in optimistic_color)
    ax_S.plot(S_M_0, grids['states'][0],
              color='k')
    ax_S.plot(S_M_true, grids['states'][0],
              color='k')

    ax_S.fill_betweenx(grids['states'][0], 0, S_M_0, facecolor=c, alpha=0.7)


    # if S_labels:
    #     ax_S.legend(S_labels, loc=2, ncol=1)

    ax_S.set_xlim((0, max(S_M_true)*1.2))
    ax_S.get_yaxis().set_visible(False)
    ax_S.set_xlabel('$\Lambda$')
    # ax_S.set_ylabel('state space: height at apex')
    aspect_ratio_Q = 'auto'  # 1.5
    # aspect_ratio_S = ax_Q.get_xlim() / s_max
    # ax_S.set_aspect(aspect_ratio_S)


    X, Y = np.meshgrid(grids['actions'][0], grids['states'][0])

    ax_Q.contour(X, Y, Q_V_true, [.5], colors='k')
    # ax_Q.contour(X, Y, Q_V_explore, [.5], colors='k')
    # ax_Q.contour(X, Y, Q_V_safe, [.5], colors='k')

    # Build image from sets
    ax_Q.contour(X, Y, Q_V_safe, [.5], colors='k')
    ax_Q.contour(X, Y, Q_V_explore, [.5], colors='k')

    Q_unviable = (~Q_V_true)*1

    matplotlib.rcParams['hatch.color'] = tuple(c/256 for c in truth_color)
    ax_Q.contourf(X, Y, Q_V_true, [.5,2], colors='w', hatches=["--", None])

    if Q_F is not None:
        matplotlib.rcParams['hatch.color'] = tuple(c / 256 for c in failure_color)
        ax_Q.contourf(X, Y, Q_F, [.5,2], colors='w', hatches=["XX", None])
        Q_unviable[Q_F] = 0

    matplotlib.rcParams['hatch.color'] = tuple(c / 256 for c in unviable_color)
    ax_Q.contourf(X, Y, Q_unviable, [.5,2], colors='w', hatches=["//", None])


    ax_Q.contourf(X, Y, Q_V_safe, [.5,2],
                  colors=[tuple((c)/256 for c in optimistic_color), (0,0,0,0)],
                  alpha=0.7)

    ax_Q.contourf(X, Y, Q_V_explore, [.5,2],
                  colors=[tuple((c)/256 for c in explore_color), (0,0,0,0)],
                  alpha=0.7)

    if samples is not None and samples[0] is not None:
        action = samples[0][:, 1]
        state = samples[0][:, 0]
        if failed_samples is not None and len(failed_samples) > 0:
            ax_Q.scatter(action[failed_samples], state[failed_samples],
                         marker='x', edgecolors=[[0.9, 0.3, 0.3]], s=30,
                         facecolors=[[0.9, 0.3, 0.3]])
            failed_samples = np.logical_not(failed_samples)
            ax_Q.scatter(action[failed_samples], state[failed_samples],
                         facecolors=[[0.9, 0.3, 0.3]], s=30,
                         marker='.', edgecolors='none')
            failed_samples = np.logical_not(failed_samples)
        else:
            ax_Q.scatter(action, state,
                         facecolors=[[0.9, 0.3, 0.3]], s=30,
                         marker='.', edgecolors='none')



    ax_Q.set_xlabel('action space $A$')
    ax_Q.set_ylabel('state space $S$')

    extent = [grids['actions'][0][0],
              grids['actions'][0][-1],
              grids['states'][0][0],
              grids['states'][0][-1]]

    frame_width_x = grids['actions'][0][-1]*.03
    ax_Q.set_xlim((grids['actions'][0][0] - frame_width_x,
                   grids['actions'][0][-1] + frame_width_x))

    frame_width_y = grids['states'][0][-1]*.03
    ax_Q.set_ylim((grids['states'][0][0] - frame_width_y,
                   grids['states'][0][-1] + frame_width_y))

    if variance is not None:
        if max_variance is not None:
            variance_image = ax_V.pcolor(X,Y,variance,cmap='cividis',vmin = 0,
                                    vmax = max_variance)
        else:
            variance_image = ax_V.pcolor(X,Y,variance,cmap='cividis')

        ax_V.contour(X, Y, Q_V_safe, [.5], colors='k')

        if samples is not None and samples[0] is not None:
            action = samples[0][:, 1]
            state = samples[0][:, 0]
            if failed_samples is not None and len(failed_samples) > 0:
                ax_V.scatter(action[failed_samples], state[failed_samples],
                             marker='x', edgecolors=[[0.9, 0.3, 0.3]], s=30,
                             facecolors=[[0.9, 0.3, 0.3]])
                failed_samples = np.logical_not(failed_samples)
                ax_V.scatter(action[failed_samples], state[failed_samples],
                             facecolors=[[0.9, 0.3, 0.3]], s=30,
                             marker='.', edgecolors='none')
            else:
                ax_V.scatter(action, state,
                             facecolors=[[0.9, 0.3, 0.3]], s=30,
                             marker='.', edgecolors='none')

        ax_V.set_xlabel('action space $A$')
        ax_V.set_ylabel('state space $S$')

        frame_width_x = grids['actions'][0][-1]*.03
        ax_V.set_xlim((grids['actions'][0][0] - frame_width_x,
                       grids['actions'][0][-1] + frame_width_x))

        frame_width_y = grids['states'][0][-1]*.03
        ax_V.set_ylim((grids['states'][0][0] - frame_width_y,
                       grids['states'][0][-1] + frame_width_y))

        fig.colorbar(variance_image,ax=ax_colorbar,location='left')

    #fig.subplots_adjust(0, 0, 1, 1)
    return fig


def create_plot_callback(n_samples, experiment_name, random_string, every=50, plot_variance = False, show_flag=True, save_path='./results/',export_gif=False):
    if plot_variance:
        max_variance = 0
    def plot_callback(sampler, ndx, thresholds):
        # Plot every n-th iteration
        if ndx % every == 0 or ndx + 1 == n_samples or ndx == -1:

            Q_map_true = sampler.model_data['Q_map']
            grids = sampler.grids

            Q_M_true = sampler.model_data['Q_M']
            Q_V_true = sampler.model_data['Q_V']
            S_M_true = sampler.model_data['S_M']
            Q_F = sampler.model_data['Q_F']

            Q_V = sampler.current_estimation.safe_level_set(safety_threshold=0,
                                                            confidence_threshold=thresholds['measure_confidence'])
            S_M_0 = sampler.current_estimation.project_Q2S(Q_V)

            Q_V_exp = sampler.current_estimation.safe_level_set(safety_threshold=thresholds['safety_threshold'],
                                                                confidence_threshold=thresholds[
                                                                    'exploration_confidence'])

            if plot_variance:
                _,variance = sampler.current_estimation.Q_M()
                nonlocal max_variance
                max_variance = np.max([max_variance,np.amax(variance)])

            else:
                variance = None
                max_variance = None

            data2save = {
                'Q_V_true': Q_V_true,
                'Q_V_exp': Q_V_exp,
                'Q_V': Q_V,
                'S_M_0': S_M_0,
                'S_M_true': S_M_true,
                'grids': grids,
                'sampler.failed_samples': sampler.failed_samples,
                'ndx': ndx,
                'threshold': thresholds
            }

            fig = plot_Q_S(Q_V_true, Q_V_exp, Q_V, S_M_0, S_M_true,
                           grids, samples=(sampler.X, sampler.y),
                           failed_samples=sampler.failed_samples, Q_F=Q_F,
                           variance = variance,max_variance = max_variance)

            if save_path is not None:

                today = [datetime.date.today()]
                folder_name = str(today[0]) + '_experiment_name_' + experiment_name + '_random_' + random_string

                filename_to_format = '{:s}_samples_' + experiment_name + '_fig'
                filename = filename_to_format.format(str(ndx).zfill(4))

                path = save_path + folder_name + '/'

                file = Path(path)
                file.mkdir(parents=True, exist_ok=True)

                outfile = open(path + filename + '.pickle', 'wb')
                pickle.dump(data2save, outfile)
                outfile.close()

                plt.savefig(path + filename + '.pdf', format='pdf')

            plt.tight_layout()
            if show_flag:
                plt.show()

            plt.close('all')

            if sampler.y is not None:
                print(str(ndx) + " ACCUMULATED ERROR: "
                      + str(np.sum(np.abs(S_M_0 - S_M_true)))
                      + " Failure rate: " + str(np.mean(sampler.y < 0)))

        if ndx + 1 == n_samples:
            if export_gif and save_path is not None:
                generic_filename_minus = path + filename_to_format.format('-0*') + '.pdf'
                generic_filename_plus = path + filename_to_format.format('0*') + '.pdf'
                evolution_command = "convert -verbose -delay 50 -loop 0 -density 300 {:s} {:s} {:s}/evolution_{:s}.gif".format(generic_filename_minus,generic_filename_plus,path,random_string)
                try:
                    os.system(evolution_command)
                except Exception as e:
                    print('Error when exporting the gif concatenation of the plots:')
                    print(e)
                    print('Suggestion : make sure the command \'convert\', from \'imagemagick\', is installed.')

    return plot_callback
