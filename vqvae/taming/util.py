import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

##################################################################################
## Computations
##################################################################################
def entropy(p):
    if p == 0:
        return 0
    else:
        return p * np.log2(p)
    
def compute_impurity(lcr, kl, lcl, mode):
    total = lcr + kl + lcl
    H = 0
    N = np.sum(total)
    for i in range(len(lcr)):
        if total[i] != 0:
            e = entropy(lcr[i]/total[i])
            e += entropy(kl[i]/total[i])
            e += entropy(lcl[i]/total[i])
            p = total[i]/N
            H += p*e
    H = -1.0*H #/ np.sum([total != 0])
    print('-------- '+ 'Entropy ' + mode + ' -----' + '--------')
    print('H = ', H)
    print('-----------------------------------')
    return H

def compute_mucov(df, idxs):
    z_dim = len(df['q_vec'][0])
    mu_ = np.zeros((idxs, z_dim))
    cov_ = np.zeros((idxs, z_dim, z_dim))
    bool_ = np.zeros((idxs), bool)
    for i in range(idxs):
        j = df['q_idx']==i
        if len(df[j]['z_vec']) == 0:
            continue
        else:
            mu = np.array(df[j]['q_vec'])[0]#.mean()
            vals = np.vstack(df[j]['z_vec'].values)
            cov = np.cov(vals, rowvar=False)
            mu_[i,:]=mu
            cov_[i, :, :] = cov
            bool_[i] = True
    df_new = pd.DataFrame({'q_idx': np.linspace(0, idxs-1, idxs, dtype=int),
                          'q_mu': mu_.tolist(),
                          'q_cov': cov_.tolist(),
                          'q_bool': bool_}) 
    return df_new


##################################################################################
## Plots
##################################################################################

def plot_future(px, py, idxs):
    plt.plot(px, py, '.')
    plt.title('Index: ' + str(idxs))
    plt.xlim(-1, 200)
    plt.ylim(-4,4)
    # plt.show()
    
def get_histinfo(df, num_idxs, path, mode):
    q_idxs = np.linspace(0, num_idxs-1, num_idxs, dtype=int)
    df.loc[df['scenario_id'].str.contains('lcr'), 'scenario_id'] = 'lcr'
    df.loc[df['scenario_id'].str.contains('kl'), 'scenario_id'] = 'kl'
    df.loc[df['scenario_id'].str.contains('lcl'), 'scenario_id'] = 'lcl'
    lcr_array = np.zeros(num_idxs)
    kl_array = np.zeros(num_idxs)
    lcl_array = np.zeros(num_idxs)
    for idx in q_idxs:
        lcr_array[idx] = len(df[(df['q_idx'] == idx) & (df['scenario_id'] == 'lcr')])
        kl_array[idx] = len(df[(df['q_idx'] == idx) & (df['scenario_id'] == 'kl')])
        lcl_array[idx] = len(df[(df['q_idx'] == idx) & (df['scenario_id'] == 'lcl')])
    idx_strings = [str(x) for x in q_idxs]
    columns = ['idx', 'lcr', 'kl', 'lcl']
    '''
    cdf = pd.DataFrame({'idx': q_idxs,
                       'lcr': lcr_array,
                       'kl': kl_array,
                       'lcl': lcl_array},)   
    cdf_log = pd.DataFrame({'idx': q_idxs,
                       'lcr': np.log(lcr_array+1),
                       'kl': np.log(kl_array+1),
                       'lcl': np.log(lcl_array+1)},)  
    cdf.to_csv(path_or_buf= os.path.join(path, mode+"_histogram.csv"))
    cdf_log.to_csv(path_or_buf= os.path.join(path, mode + "_loghistogram.csv")) 
    '''
    return lcr_array, kl_array, lcl_array
    
def plot_idx_dist(df, bins):
    df.loc[df['scenario_id'].str.contains('lcr'), 'scenario_id'] = 'lcr'
    df.loc[df['scenario_id'].str.contains('kl'), 'scenario_id'] = 'kl'
    df.loc[df['scenario_id'].str.contains('lcl'), 'scenario_id'] = 'lcl'
    sns.displot(data=df,   x="q_idx", hue="scenario_id", multiple="stack", bins=bins)
    N =  str(len(set(df['q_idx'].values)))
    plt.title('Histogram (N=' +N+ ')')
    plt.xlim(0, bins)

##################################################################################
## Other
##################################################################################
 
def get_number_params(model):
    return sum(p.numel() for p in model.parameters()) 

def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


if __name__ == "__main__":
    config = {"keya": "a",
              "keyb": "b",
              "keyc":
                  {"cc1": 1,
                   "cc2": 2,
                   }
              }
    from omegaconf import OmegaConf
    config = OmegaConf.create(config)
    print(config)
    retrieve(config, "keya")

