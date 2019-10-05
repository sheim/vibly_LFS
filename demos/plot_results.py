import pickle
from plotting.corl_plotters import plot_Q_S
import matplotlib.pyplot as plt

def plot(filename, label_axis=True):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()

    Q_V_true = data['Q_V_true']
    Q_F = data['Q_F']
    Q_V_exp = data['Q_V_exp']
    Q_V = data['Q_V']
    S_M_0 = data['S_M_0']
    S_M_true = data['S_M_true']
    grids = data['grids']
    failed_samples = data['sampler.failed_samples']
    ndx = data['ndx']
    thresholds = data['threshold']
    X = data['sampler.X']
    y = data['sampler.y']

    fig = plot_Q_S(Q_V_true, Q_V_exp, Q_V, S_M_0, S_M_true, grids,
                   samples=(X, y),
                   failed_samples=failed_samples, Q_F=Q_F,
                   label_axis=label_axis)

    fig.show()
    return fig

path = '../results/new_example_060fail_2019-10-05_experiment_name_hovership_unviable_start/'
hover_init = path + '-001_samples_hovership_unviable_start.pickle'
hover_50 = path + '0050_samples_hovership_unviable_start.pickle'
hover_final = path +  '0249_samples_hovership_unviable_start.pickle'

fig = plot(hover_init, False)
fig.savefig(path + 'paper_init.pdf', format='pdf')

fig = plot(hover_50, False)
fig.savefig(path + 'paper_50.pdf', format='pdf')

fig = plot(hover_final, True)
fig.savefig(path + 'paper_final.pdf', format='pdf')


path = '../results/new_example_070fail_2019-10-05_experiment_name_slip_prior/'
hover_init = path + '-001_samples_slip.pickle'
hover_50 = path + '0050_samples_slip.pickle'
hover_final = path +  '0499_samples_slip.pickle'


fig = plot(hover_init, False)
fig.savefig(path + 'paper_init.pdf', format='pdf')

fig = plot(hover_50, False)
fig.savefig(path + 'paper_50.pdf', format='pdf')

fig = plot(hover_final, True)
fig.savefig(path + 'paper_final.pdf', format='pdf')