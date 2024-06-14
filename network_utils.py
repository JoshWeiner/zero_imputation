import pandas as pd
import numpy as np
from SERGIO.SERGIO.sergio import sergio
from GENIE3.GENIE3 import *
import MAGIC.magic as magic
import SAUCIE.SAUCIE as SAUCIE
import scScope.scscope.scscope as DeepImpute
import deepimpute.deepimpute as deepimpute
from knn_smoothing.knn_smooth import knn_smoothing

import os
import h5py
import importlib
import scprep
import scvi
import anndata
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.sparse import csr_matrix, csc_matrix

def open_flatten_file(filepath):
    with h5py.File(filepath, 'r') as h5file:
        if 'matrix' in h5file:
            shape = tuple(h5file['matrix']['shape'][:])
            print("Shape of matrix: ", shape)

            data = h5file['matrix']['data'][:]
            indices = h5file['matrix']['indices'][:]
            indptr = h5file['matrix']['indptr'][:]

            matrix = csc_matrix((data, indices, indptr), shape=shape)
            dense_matrix = matrix.toarray()
            df = pd.DataFrame(data=dense_matrix)
            df.replace(0, 1e-5, inplace=True)
    return df

def run_saucie(x_path, y_path, ind, save_path):
    #reload_modules('tensorflow.compat')
    tf = importlib.import_module('tensorflow.compat.v1')
    #importlib.reload(SAUCIE)
    tf.disable_v2_behavior()
    print("loading data")
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    print("reset graph")
    tf.reset_default_graph()
    print("Initialize saucie")
    saucie = SAUCIE.SAUCIE(y.shape[1])
    print("Load saucie")
    loadtrain = SAUCIE.Loader(y, shuffle=True)
    print("Train saucie")
    saucie.train(loadtrain, steps=1000)

    loadeval = SAUCIE.Loader(y, shuffle=False)
    # embedding = saucie.get_embedding(loadeval)
    # number_of_clusters, clusters = saucie.get_clusters(loadeval)
    rec_y = saucie.get_reconstruction(loadeval)
    save_str = '/yhat_SAUCIE'
    np.save(save_path + save_str, rec_y)

def run_deepImpute(x_path, y_path, ind, save_path):
    #reload_modules('tensorflow.compat')
    importlib.invalidate_caches()
    multinet = importlib.import_module('deepimpute.deepimpute.multinet')
    importlib.reload(multinet)
    tf = importlib.import_module('tensorflow.compat.v1')
    #tf = importlib.import_module('tensorflow')
    tf.init_scope()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    y = np.transpose(np.load(y_path))
    y = 10 ** y
    y = pd.DataFrame(y)
    x = np.transpose(np.load(x_path))
    x = pd.DataFrame(x)
    multinet = multinet.MultiNet()
    multinet.fit(y,cell_subset=1,minVMR=0.5)
    imputedData = multinet.predict(y)
    yhat_deepimpute = imputedData.to_numpy()
    save_str = '/yhat_deepImpute'
    np.save(save_path + save_str, yhat_deepimpute)

def run_magic(x_path, y_path, ind, save_path):
    print(x_path, y_path)
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    
    save_str = '/yhat_MAGIC_t_7_0'
    y_hat = scprep.filter.filter_rare_genes(y, min_cells=5)
    np.save(save_path + save_str, y_hat)

    save_str = '/yhat_MAGIC_t_7_1'
    y_norm = scprep.normalize.library_size_normalize(y_hat)
    np.save(save_path + save_str, y_norm)

    save_str = '/yhat_MAGIC_t_7_2'
    y_norm = scprep.transform.sqrt(y_norm)
    np.save(save_path + save_str, y_norm)

    for t_val in [7]:
        magic_op = magic.MAGIC(
            t=t_val,
            n_pca=20,
            n_jobs=-1,
        )
        y_hat = magic_op.fit_transform(y_norm, genes='all_genes')
        save_str = '/yhat_MAGIC_t_' + str(t_val)
        np.save(save_path + save_str, y_hat)

def run_scScope(x_path, y_path, ind, save_path):
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    DI_model = DeepImpute.train(
          y,
          15,
          use_mask=True,
          batch_size=64,
          max_epoch=1000,
          epoch_per_check=100,
          T=2,
          exp_batch_idx_input=[],
          encoder_layers=[],
          decoder_layers=[],
          learning_rate=0.0001,
          beta1=0.05,
          num_gpus=1)
    latent_code, rec_y, _ = DeepImpute.predict(y, DI_model, batch_effect=[])
    save_str = '/yhat_scScope'
    np.save(save_path + save_str, rec_y)
    
def run_scvi(x_path, y_path, ind, save_path):
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    adata = anndata.AnnData(y)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata)
    model.train()
    x = model.get_normalized_expression(return_numpy=True)
    save_str = '/y_hat_scvi'
    np.save(save_path + save_str, x)

def run_knn(x_path, y_path, ind, save_path, k=10):
    y = np.transpose(np.load(y_path))
    y[y == 0] = 1e-4
    data = np.float64(y)
    data[data == np.nan] = 1e-4
    result = knn_smoothing(data, k)
    save_str = '/y_hat_knn'
    np.save(save_path + save_str, result)


def gt_benchmark(virtual_imputation, target_file):
    # Create numpy array of same size as imputation_dataset
    gt_temp = np.zeros_like(virtual_imputation)
    f = open(target_file,'r')
    Lines = f.readlines()
    f.close()
    # For each real gene and measured gene expressions, set new array at coordinates to 1
    for j in tqdm(range(len(Lines))):
        line = Lines[j]
        line_list = line.split(',')
        target_index = int(float(line_list[0]))
        num_regs = int(float(line_list[1]))
        # skip if gene is not present in filtered dataset
        if target_index >= gt_temp.shape[1]:
            for i in range(0, target_index + 1 - gt_temp.shape[1]):
                new_column = np.zeros((gt_temp.shape[0], 1), dtype=int)
                gt_temp = np.append(gt_temp, new_column, axis=1)
                virtual_imputation = np.append(virtual_imputation, new_column, axis=1)
        for i in range(num_regs):
            reg_index = int(float(line_list[i+2]))
            gt_temp[reg_index,target_index] = 1  
    return gt_temp, virtual_imputation

def create_graph(data, cutoff=0.95, method_name='None', dataset_name=''):
    edges = pd.DataFrame(data)
    threshold = np.quantile(edges.values.flatten(), cutoff)
    
    G = nx.DiGraph()
    mask = edges >= threshold
    rows, cols = np.where(mask)
    for row, col in zip(rows, cols):
        weight = edges.iat[row, col]
        gene_1 = edges.index[row]
        gene_2 = edges.columns[col]
        G.add_edge(gene_1, gene_2, weight=weight)    
    pos = nx.spring_layout(G)
    es, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=es, edge_color=weights, width=1, edge_cmap=plt.cm.Blues)

    if method_name == 'None':
        plt.title('Top GRN Inference Edges' + '\n on ' + dataset_name)
    else:
        plt.title('Top GRN Inference Edges with Imputation Method ' + method_name + '\n on ' + dataset_name) 
    plt.axis('off')
    plt.show()
    return G

def run_GRN_and_graph(data, save_path, dataset_name='', n_genes=100):
    print("Num genes:", n_genes)
    #sampled = data.sample(n=n_genes, random_state=42)
    #np.save('./zero_imputation_experiments/sampled.npy', sampled)
    sampled = pd.DataFrame(np.load(save_path + 'sampled.npy'))
    edges_dict = {}
    for i, method in enumerate([None, 'SAUCIE', 'scScope', 'DeepImpute', 'MAGIC', 'SCVI', 'KNN']):
        print(i, method)
        if method is None:
            x = np.transpose(sampled.values)
            method = 'None'
        elif method == 'SAUCIE':
            print("Running SAUCIE")
            run_saucie(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_SAUCIE.npy')
        elif method == 'scScope':
            print("Running scScope")
            run_scScope(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_scScope.npy')
        elif method == 'DeepImpute':
            print("Running DeepImpute")
            run_deepImpute(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_deepImpute.npy')
        elif method == 'MAGIC':
            print("Running MAGIC")
            run_magic(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_MAGIC_t_7.npy')
        elif method == "SCVI":
            print("Running SCVI")
            run_scvi(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'y_hat_scvi.npy')
        elif method == "KNN":
            print("Running KNN")
            run_knn(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path, k=32)
            x = np.load(save_path + 'y_hat_knn.npy')

        x[x == 0] = 1e-5
        vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
        G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
        edges_dict[method] = set(G.edges())
    return edges_dict

def run_GRN_and_graph_with_gt(data, save_path, dataset_name='', n_genes=100):
    print("Num genes:", n_genes)
    #sampled = data.sample(n=n_genes, random_state=42)
    #np.save('./zero_imputation_experiments/sampled.npy', sampled)
    sampled = pd.DataFrame(np.load(save_path + 'sampled.npy'))
    edges_dict = {}
    for i, method in enumerate([None, 'SAUCIE', 'scScope', 'DeepImpute', 'MAGIC', 'SCVI', 'KNN']):
        print(i, method)
        if method is None:
            x = np.transpose(sampled.values)
            method = 'None'

            x[x == 0] = 1e-5
            vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
            G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
            edges_dict[method] = set(G.edges())

        elif method == 'SAUCIE':
            print("Running SAUCIE")
            run_saucie(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_SAUCIE.npy')

            x[x == 0] = 1e-5
            vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
            G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
            edges_dict[method] = set(G.edges())

        elif method == 'scScope':
            print("Running scScope")
            run_scScope(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_scScope.npy')

            x[x == 0] = 1e-5
            vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
            G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
            edges_dict[method] = set(G.edges())

        elif method == 'DeepImpute':
            print("Running DeepImpute")
            run_deepImpute(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'yhat_deepImpute.npy')

            x[x == 0] = 1e-5
            vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
            G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
            edges_dict[method] = set(G.edges())

        elif method == 'MAGIC':
            print("Running MAGIC")
            run_magic(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)

            x_final = np.load(save_path + 'yhat_MAGIC_t_7.npy')
            x_0 = np.load(save_path + 'yhat_MAGIC_t_7_0.npy')
            x_1 = np.load(save_path + 'yhat_MAGIC_t_7_1.npy')
            x_2 = np.load(save_path + 'yhat_MAGIC_t_7_2.npy')
            x = [('_final', x_final), ('_0', x_0), ('_1', x_1), ('_2', x_2)]
            for x_name, x_val in x:
                x_val[x_val == 0] = 1e-5
                vim = GENIE3(x_val, nthreads=12, ntrees=100, regulators='all')
                G = create_graph(vim, cutoff=0.95, method_name=method + x_name , dataset_name=dataset_name)
                edges_dict[method + x_name] = set(G.edges())

        elif method == "SCVI":
            print("Running SCVI")
            run_scvi(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path)
            x = np.load(save_path + 'y_hat_scvi.npy')

            x[x == 0] = 1e-5
            vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
            G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
            edges_dict[method] = set(G.edges())

        elif method == "KNN":
            print("Running KNN")
            run_knn(save_path + 'sampled.npy', save_path + 'sampled.npy', i, save_path, k=32)
            x = np.load(save_path + 'y_hat_knn.npy')

            x[x == 0] = 1e-5
            vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
            G = create_graph(vim, cutoff=0.95, method_name=method, dataset_name=dataset_name)
            edges_dict[method] = set(G.edges())

    return edges_dict