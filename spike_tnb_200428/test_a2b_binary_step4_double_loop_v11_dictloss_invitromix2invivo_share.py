
import os, shutil, log, config, sys, scipy.io
from configobj import ConfigObj
import traceback

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np

from sklearn import metrics

import pandas as pd
import glob, copy, subprocess, time, matplotlib
import matplotlib.pyplot as plt
import math
import random

from argparse import ArgumentParser
from pathlib import Path
from logging import getLogger

matplotlib.use('Agg')        
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

rootdir0 = "F:/python_home/200410_Music_Naka/home_dir_run/spike_tnb_200428"

os.chdir( rootdir0 )

os.chdir( "./" )
region_id = {}
region_date = {}
region_date_s = {}

###############################################################################
data_name_train = [None] * 17  # Create a list of 17 elements (index 0 to 16)
init_cell_index_train0 = [None] * 17  # Create a list of 17 elements (index 0 to 16)

for group in range(1, 17):  # Loop from 1 to 16

    if group == 1:
        data_name_train[group] = ['210511']
        init_cell_index_train0[group] = 0

###############################################################################

data_name_test = [None] * 10  # Create a list of 17 elements (index 0 to 16)
init_cell_index_test0 = [None] * 10  # Create a list of 17 elements (index 0 to 16)
depth_min_test = [None] * 10  # Create a list of 17 elements (index 0 to 16)
depth_max_test = [None] * 10  # Create a list of 17 elements (index 0 to 16)

for group in range(1, 10):  # Angelakilab / Loop from 1 to 16

    if group == 1: # Angelakilab /  VIS1, VIS2,~VISporらの視覚領域,
        data_name_test[group] = ['angelakilab/Subjects/NYU-40/2021-04-14/001/']  
        init_cell_index_test0[group] = 150
        depth_min_test[group] = 1800
        depth_max_test[group] = 3700
    
        
###############################################################################
region_id_train0 = {}
region_date_train0 = {}
init_cell_index_train00 = {}
init_cell_index_test00 = {}
depth_min_test0 = {}
depth_max_test0 = {}

# 10個の要素についてループを実行
for i in range(1, 2):
    for group in range(1, 17):
        if group not in region_date_train0:
            region_date_train0[group] = {}
        region_date_train0[group][i] = data_name_train[group][i-1]
        init_cell_index_train00[group] = init_cell_index_train0[group]
        
for group in range(1, 3):
    region_id_train0[group] = "InVitro"

###############################################################################
region_id_test0 = {}
region_date_test0 = {}

# 10個の要素についてループを実行
for i in range(1, 2):
  #  for group in range(1, 11):
    for group in range(1, 10):
        if group not in region_date_test0:
            region_date_test0[group] = {}
        region_date_test0[group][i] = data_name_test[group][i-1]
        init_cell_index_test00[group] = init_cell_index_test0[group]
        depth_min_test0[group] = depth_min_test[group]
        depth_max_test0[group] = depth_max_test[group]
        
for group in range(1, 3):
    region_id_test0[group] = "InVivo"
    
###############################################################################

def calculate_spike_density(data):
    # データを非負にクリップ
    data = np.clip(data, 0, None)
    return np.mean(data), np.min(data), np.max(data)

def adjust_threshold_init(output, target_density, initial_threshold=1, max_iterations=1000, tolerance=0.001):
    print("adjust_threshold: ")
    print("----_calculate_data_density、training dataの発火率を計算して、最適化している")
    # 出力データを非負にクリップ（ReLUを適用）
 #   epsilon = 1e-6
    print("Output unique values (original):", np.unique(output.cpu().numpy()))
             
    low, high = -0.1, 1.1  # low, high = 0, 1 から編集を行った
    threshold = initial_threshold
    
    print("--------------------------------------------------------------------")
    print("Init. thr.: " + str(threshold))
        
    for iteration_step in range(max_iterations):
        if iteration_step % 100 == 1:
           print("Iteration step of threshold tuning: " + str(iteration_step))
      
        if iteration_step == 1:
          print("Output unique values:", np.unique(output.cpu().numpy()))
          
        # しきい値を超えるかどうかで二値化
        binary_output = (output > threshold).float()
        if iteration_step == 1:
           print("Binary output unique values:", np.unique(binary_output.cpu().numpy()))
        
        current_density, min_dens, max_dens = calculate_spike_density(binary_output.cpu().numpy())
        
        if iteration_step % 100 == 1:
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print("Mean:", current_density, ", min:", min_dens, ", max:", max_dens, " of binary_output")
            print("1000 * current density: " + str(1000 *current_density) + ", 1000 *target density: " + str(1000 *target_density))
        
        # スパイク密度がターゲットに収束した場合
        if abs(current_density - target_density) < tolerance:
            print("---------------------------------------------------------------------------")
            print("Threshold was converged!!")
            print("Iteration step of threshold tuning: " + str(iteration_step))
            break
            
        # スパイク密度に基づいてしきい値を調整
        if current_density > target_density:
            low = threshold
        else:
            high = threshold

        # しきい値を二分法で調整
        threshold = (low + high) / 2
        
        if iteration_step % 100 == 1:
           print("Modified thr.: " + str(threshold))
     
    print("Iteration step of threshold tuning: " + str(iteration_step))
    print("1000 * current density: " + str(1000 *current_density) + ", 1000 *target density: " + str(1000 *target_density))
    if target_density > current_density:
         threshold = threshold*0.999
         print("Threshold reset:")
         print("1000 * current density: " + str(1000 *current_density) + ", 1000 *target density: " + str(1000 *target_density))
     
    print("Finral thr.: " + str(threshold))
    
    print("Binary output unique values:", np.unique(binary_output.cpu().numpy()))
    
    return threshold

###############################################################################
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def adjust_threshold_ROC_with_max_fpr(output, target, max_fpr=0.2):
    # 二次元データを一列にフラット化
    output_flat = output.ravel()
    target_flat = target.ravel()
    
    # ROC曲線を計算
    fpr, tpr, thresholds = roc_curve(target_flat, output_flat)
    roc_auc = auc(fpr, tpr)
    
    # ROC曲線のプロット
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # fprの最大値(max_fpr)を考慮した閾値の選択
    valid_indices = np.where(fpr <= max_fpr)[0]
    if len(valid_indices) == 0:
        print("Error: No threshold satisfies the maximum FPR condition.")
        return None

    # 有効な範囲内でYouden's J統計量を計算
    youden_index = tpr[valid_indices] - fpr[valid_indices]
    optimal_idx = valid_indices[np.argmax(youden_index)]
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Threshold for Low FPR (Youden's J statistic with max_fpr={max_fpr}): {optimal_threshold:.4f}")
    return optimal_threshold, roc_auc

###############################################################################

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_ROC(output, target):
    # output のフラット化処理
    if isinstance(output, np.ndarray):
        output_flat = output.ravel()
    elif isinstance(output, torch.Tensor):
        output_flat = output.contiguous().cpu().numpy().reshape(-1)
    else:
        raise TypeError("output must be either numpy.ndarray or torch.Tensor")

    # target のフラット化処理
    if isinstance(target, np.ndarray):
        target_flat = target.ravel()
    elif isinstance(target, torch.Tensor):
        target_flat = target.contiguous().cpu().numpy().reshape(-1)
    else:
        raise TypeError("target must be either numpy.ndarray or torch.Tensor")

    # target_flat が二値であることを確認し、連続値であれば二値化
    unique_values = np.unique(target_flat)
    print("Unique values in target_flat:", unique_values)
    if len(unique_values) > 2 or (unique_values != [0, 1]).any():
        print("Converting target_flat to binary with threshold 0.5...")
        target_flat = (target_flat >= 0.5).astype(int)

    # ROC曲線とAUCの計算
    fpr, tpr, thresholds = roc_curve(target_flat, output_flat)
    roc_auc = auc(fpr, tpr)

    # ROC曲線のプロット
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return roc_auc, fpr, tpr, thresholds


   
###############################################################################
def delete_directory(path):
    """指定されたパスのディレクトリを削除する。削除できない場合は、最大5回まで再試行する。"""
    max_retries = 5
    retries = 0
    while True:
        try:
            shutil.rmtree(path)
            print(f"Directory removed: {path}")
            break
        except PermissionError as e:
            if retries < max_retries:
                retries += 1
                print(f"Permission error on attempt {retries}: {e} - Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to remove directory after {max_retries} attempts.")
                raise
            
###############################################################################
def sample(data):
    data = data.to('cpu').numpy()
    values = np.random.random(data.shape)
    samples = (data >= values).astype(np.float32)
    return torch.from_numpy(samples)

###############################################################################

def custom_collate_fn(batch):
    max_length = max(x[0].shape[0] for x in batch)
    padded_tensors = [(torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.shape[0])), label) for (tensor, label) in batch]
    tensors = torch.stack([x[0] for x in padded_tensors])
    labels = torch.tensor([x[1] for x in padded_tensors])
    return tensors, labels

###############################################################################
def main_mod_func(input1, input2, input3, input4, input5):
    import pathlib
    MODEL, MODE, GENERATE_MODE, QUANTIZATION_MODE, CONFIGFILE_NAME = input1, input2, input3, input4, 'data/' + input5
   # rootdir = str(pathlib.Path(__file__).parent)
    rootdir = os.getcwd()  # カレントディレクトリを取得
    CONFIG_FILENAM = 'log_config.ini'
    modelrunners = {'transformer': TransformerStackRunner}
    assert MODEL in modelrunners, f'can not use the model {MODEL}'
    experiment_name = MODEL
    config = Config().read(os.path.join(rootdir, CONFIG_PATH))
    config.dir_result = os.path.join(config.dir_result, MODEL)
    os.makedirs(config.dir_result, exist_ok=True)
    shutil.copy(os.path.join(rootdir, CONFIG_PATH), os.path.join(config.dir_result, CONFIG_FILENAME))
    log.setlogger(config.dir_result, config.logger)
    modelrunner = modelrunners[MODEL](MODE, experiment_name, config)
    modelrunner.run(generate_mode=GENERATE_MODE, quantization_mode=QUANTIZATION_MODE)
    
###############################################################################
def load_and_convert_asdf_to_npy(data_dir, input_file, output_file):
    file_path = data_dir + input_file
    matdata = scipy.io.loadmat(file_path)
    nNeuron, nTime = len(matdata['asdf'][0])-2, matdata['asdf'][0][-1][0][1]
    
    spikes = np.zeros((nTime + 1, nNeuron), dtype=int)
    
    for i in range(nNeuron):
        times = matdata['asdf'][0][i+1][0].astype(int)  # times を整数型に変換
        spikes[times, i] = 1
    
    np.save(data_dir + output_file, spikes)
    print(f"File saved at {data_dir + output_file}")
    return spikes

from scipy.io import loadmat

######################################################################################
import time
    

def organize_and_save_spike_data_renew(infile2, outfile, num_cells, extract_index, sep_size):
    # npzファイルからデータを読み込む
    with np.load(infile2) as data:
        clusters = data['clusters']  # 細胞のインデックス
        times = data['times'] * 1000  # スパイクのタイミング  [sec]--> [ms]
        print("max time:" + str(np.max(np.max(times))))

    unique_clusters = np.unique(clusters)
    num_unique_clusters = len(unique_clusters)

    print(f"重複しない整数の個数は: {num_unique_clusters}")

    # timesの正規化（例えば、最小値を0にして最大値をsep_size未満に収める）
    times = times - np.min(times)  # 最小値を0に揃える
    max_time = np.max(times) - np.min(times)
    max_time_int = int(max_time)

    # スパイクをカウントするための行列を初期化
    spike_matrix = np.zeros((len(extract_index), max_time_int + 1), dtype=np.int32)

    print("len(times):" + str(len(times)))
    print("len(clusters):" + str(len(clusters)))
    print("FRs:" + str(len(times) / (num_unique_clusters * max_time_int)))
    print("max(times):" + str(max(times)))
    print("sep_size:" + str(sep_size))

    # スパイクデータを選択した細胞×時間ステップ形式に変換
    for i in range(0, len(times)):
        cell_id = int(clusters[i])  # 細胞ID
        time_step = int(times[i])   # スパイクの発生時間

        # 細胞IDがextract_indexに含まれている場合
        if cell_id in extract_index:
            matrix_index = extract_index.index(cell_id)
            spike_matrix[matrix_index, time_step] = 1  # スパイクをカウント

    print("FR0:" + str(np.mean(np.mean(spike_matrix))))

    # 結果をspikes0.npyとして保存
    np.save(outfile, spike_matrix.T)
    print(f"Spike data saved to {outfile}")
    time.sleep(5)

    return spike_matrix



# def organize_and_save_spike_data(infile2, outfile, num_cells=128, sep_size=1000):
def organize_and_save_spike_data(infile2, outfile, num_cells, sep_size, init_cell_index):
    # npzファイルからデータを読み込む
    with np.load(infile2) as data:
        clusters = data['clusters']  # 細胞のインデックス
        times = data['times'] * 1000      # スパイクのタイミング  [sec]--> [ms]
        print("max time:" + str(np.max(np.max(times))))
        
    
    unique_clusters = np.unique(clusters)
    num_unique_clusters = len(unique_clusters)

    print(f"重複しない整数の個数は: {num_unique_clusters}")

    # timesの正規化（例えば、最小値を0にして最大値をsep_size未満に収める）
    times = times - np.min(times)  # 最小値を0に揃える
    max_time = np.max(times) - np.min(times)
    max_time_int = int(max_time)

    # スパイクをカウントするための行列を初期化
    # spike_matrix = np.zeros((num_cells, sep_size), dtype=np.int32)
    spike_matrix = np.zeros((num_cells, max_time_int+1), dtype=np.int32)

    print("len(times):" + str(len(times)))
    print("len(clusters):" + str(len(clusters)))
    print("FRs:" + str(len(times)/(num_unique_clusters*max_time_int)))
    print("max(times):" + str(max(times)))
    print("sep_size:" + str(sep_size))

    # スパイクデータを128細胞×時間ステップ形式に変換
    for i in range(0,len(times)):
        cell_id = int(clusters[i])  # 細胞ID
        time_step = int(times[i])   # スパイクの発生時間
    
        # # 細胞IDと時間ステップが範囲内か確認
        # if init_cell_index <= cell_id < num_cells + init_cell_index and 0 <= time_step < sep_size:
        #   #  print("(" + str(cell_id) + " , " + str(time_step) + ")")
        #     spike_matrix[cell_id-init_cell_index, time_step] += 1  # スパイクをカウント

        # 細胞IDと時間ステップが範囲内か確認
        if init_cell_index <= cell_id < num_cells + init_cell_index: #  and 0 <= time_step < sep_size:
          #  print("(" + str(cell_id) + " , " + str(time_step) + ")")
            spike_matrix[cell_id-init_cell_index, time_step] = 1  # += 1  # スパイクをカウント
        
    # print("FR0:" + str(len(times)/(num_unique_clusters*max_time_int)))
    print("FR0:" + str(np.mean(np.mean(spike_matrix))))
    # 結果をspikes0.npyとして保存
    np.save(outfile, spike_matrix.T)
    print(f"Spike data saved to {outfile}")
    time.sleep(5) 

    return spike_matrix



###############################################################################

def div_weig_ryo(data_dir, div_count0, start_time_step0, segment_step0, sep_size0, init_cell_index, depth_min_test, depth_max_test ):
    infile1 = os.path.join(data_dir, 'spikes0.npy')
    infile2 = os.path.join(data_dir, 'spikes.npz')  
    
    print(infile1)
    print(infile2)
    
    extract_index = list(range(0, 1))
    
    if os.path.exists(infile2):
        print("=========depth0==============")
        spikes = np.load(infile2)
        print(spikes['depths'])
        depth = spikes['depths']
        clusters = spikes['clusters']
      #  print("      depth --- size :", len(depth))
        print("=========depth0==============")
        
        max_index = np.max(clusters)
    
        # ##########################################################
        # ##########################################################
        # depth_min_test, depth_max_test の間のindexを探す
        # 128に足りなければ、さらにカウントする
        # ##########################################################
        
        # インデックスを抽出
     #   selected_indices = np.where((depth >= depth_min_test) & (depth <= depth_max_test))[0]
        selected_indices = np.unique(clusters[np.where((depth >= depth_min_test) & (depth <= depth_max_test))[0]])
        
        print("Extracted indices (initial):", selected_indices)
        print("                   -- size :", len(selected_indices))
        print("=========depth0==============")
        # 必要な数
        required_count = 128
        
        # 結果を格納するリストスト
        
        # 条件を満たすインデックスの数が128以上の場合
        if len(selected_indices) >= required_count:
            extract_index = selected_indices[:required_count].tolist()
        else:
            # 足りない数を補う
            extract_index = selected_indices.tolist()
            remaining_count = required_count - len(extract_index)
        
            # インデックスを追加 (間を優先、次に後ろ、最後に前)
            extra_indices = []
        
            # 間の補完
            for i in range(1, len(selected_indices)):
                if len(extra_indices) >= remaining_count:
                    break
                mid_index = (selected_indices[i - 1] + selected_indices[i]) // 2
                if mid_index not in extract_index and 0 <= mid_index < len(depth) and mid_index < max_index:
                    extra_indices.append(mid_index)
        
            # 後ろの補完
            back_index = selected_indices[-1] + 1 if len(selected_indices) > 0 else 0
            while len(extra_indices) < remaining_count and back_index < len(depth) and back_index < max_index:
                if back_index not in extract_index:
                    extra_indices.append(back_index)
                back_index += 1
        
            # 前の補完
            front_index = selected_indices[0] - 1 if len(selected_indices) > 0 else len(depth) - 1
            while len(extra_indices) < remaining_count and front_index >= 0:
                if front_index not in extract_index and front_index < max_index:
                    extra_indices.append(front_index)
                front_index -= 1
        
            extract_index.extend(extra_indices[:remaining_count])
        
            # 結果のソート
            extract_index = sorted(extract_index)

        # 結果の表示
        print("Extracted indices:", extract_index)
        print("Number of indices:", len(np.unique(extract_index)))

    # ##########################################################
    # ##########################################################
    # ##########################################################
    
    
    # spikes0.npyが存在しない場合はspikes.npzをロードして変換・保存
    if os.path.exists(infile2):
        if os.path.exists(infile1):
           os.remove(infile1)
           print(f"Deleted {infile1}")
           
        if extract_index:
            # extract_index が存在する場合
            print("##########################################################")
            print("##########################################################")
            print("##########################################################")
            print("##########################################################")
            print("   Now, organize_and_save_spike_data_renew is running")
            print("##########################################################")
            print("##########################################################")
            print("##########################################################")
            print("##########################################################")
            spike_matrix = organize_and_save_spike_data_renew(infile2, infile1, 128, extract_index, sep_size0)
        else:
            # extract_index が存在しない場合
            spike_matrix = organize_and_save_spike_data(infile2, infile1, 128, sep_size0, init_cell_index)
            
            
        aa1 = np.load(infile1)
        print(f"Converted and loaded {infile2}")
    else:
        print(f" {infile2} dose not exists.")
    
    aa1 = np.load(infile1)
    print(f"Loaded {infile1}")
    print("                   -- size :", len(aa1))
    print("FR:" + str(np.mean(np.mean(aa1))))
    
    # データの時間ステップの最大値を取得
    max_time_steps = aa1.shape[0]  # 行方向が時間ステップ

    # start_pがデータ範囲を超えている場合
    if start_time_step0 >= max_time_steps:
        print(f"Warning: start_time_step {start_time_step0} exceeds available time step range {max_time_steps}. Returning empty array.")
    else: 
        print(f"OK: start_time_step {start_time_step0} does not exceed available time step range {max_time_steps}. ")
    
    start_p = int(start_time_step0)
    end_p = min(start_time_step0 + sep_size0, aa1.shape[0])  # 範囲外アクセスを防ぐためにminを使用
    aa12 = aa1[start_p:end_p, :].astype(np.int32)
    
    return aa12, extract_index  #.T


###############################################################################
def npy2npr_mod3_func(input, output):
    data = np.load(input) 
    n_time, n_neuron = data.shape[-2:] if len(data.shape) == 2 else data.shape[1:]
    lines = [f'#nNeuron\t{n_neuron}', f'#nTime\t{n_time}']
    nz = data.nonzero()
    lines.extend([f'{datum[1]}\t{datum[2]}' if len(datum) == 3 else f'{datum[0]}\t{datum[1]}' for datum in zip(*nz)])
    assert len(lines) - 2 == len(nz[0])
    with open(output, mode='w') as f:
        f.write('\n'.join(lines) + '\n')

###############################################################################
def save_first_10000_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.writelines(next(infile) for _ in range(10000))
        
###############################################################################
def print_first_60_lines(input_file):
    with open(input_file, 'r') as infile:
        lines = [line.strip() for line in infile][:60]
    max_length = max(len(line) for line in lines)
    for i in range(max_length):
        print(' '.join(line[i] if i < len(line) else ' ' for line in lines))

###############################################################################
def safe_opentail(filename, n):
    try:
        with open(filename, 'r') as file:
            return file.readlines()[-n:]
    except IOError as e:
        print(f"Unable to access {filename}: {e}")
        return []

###############################################################################
def parse_log_line(line):
    line = line.replace("nan", "0")  # 'nan' を '0' に置き換える
    ini0loss = line.find("loss:")
    ini0 = line.find("all")
    ini1 = line.find("spike")
    ini2 = line.find("nonspike")
    xx0loss = np.array(list(map(float, line[ini0loss+6:ini0-10].split("\t")))[0])
    xx0 = np.array(list(map(float, line[ini0+6:ini1-5].split("\t")))[0])
    xx1 = np.array(list(map(float, line[ini1+7:ini2-3].split("\t")))[0])
    xx2 = np.array(list(map(float, line[ini2+10:ini2+20].strip().replace('}', '').split("\t")))[0])

    return xx0loss, xx0, xx1, xx2

###############################################################################
def test_func(gen_dir2_now):
    kk = 0
    # Ensure gen_dir2_now is a Path object
    gen_dir2_now = Path(gen_dir2_now)
    # Correct path concatenation
    log_path = gen_dir2_now / "transformer" / "log.log"

    # Pass the correct log_path to the function
    lines = safe_opentail(log_path, 2 * kk + 3)

    if lines:
        line = lines[-1]
        xx0loss, xx00, xx10, xx20 = parse_log_line(line)
        loss_train_all[epoch_num-kk-2] = xx0loss
        accu_train_all[epoch_num-kk-2] = xx00
        accu_train_nonspike[epoch_num-kk-2] = xx20
        print(f"[First] all: {xx00}, spike: {xx10}, nonspike: {xx20}")

        line = safe_opentail(gen_dir2_now + "/transformer/log.log", 2 * kk + 2)[-1]
        xx0loss, xx00, xx10, xx20 = parse_log_line(line)
        loss_valid_all[epoch_num-kk-2] = xx0loss
        accu_valid_all[epoch_num-kk-2] = xx00
        accu_valid_spike[epoch_num-kk-2] = xx10
        accu_valid_nonspike[epoch_num-kk-2] = xx20

        print(f"[Second] all: {xx00}, spike: {xx10}, nonspike: {xx20}")
        kk += 1
    
    return loss_train_all, loss_valid_all, accu_valid_all, accu_valid_nonspike

###############################################################################

def update_order_with_heuristics(current_order, temperature, iteration_num, num_trials=3, max_swap_count=10):
    new_order = current_order.copy()
    
    swap_count = int(len(new_order) * (temperature / 400.0))
    swap_count = max(1, min(swap_count, max_swap_count))  # 上限を max_swap_count に設定

    best_order = new_order.copy()
    best_loss = float('inf')

    # ランダムにインデックスのペアを選ぶ（重複なし）
    swap_pairs = []
    for _ in range(swap_count):
        while True:
            a = random.randint(0, len(new_order) - 1)
            b = random.randint(0, len(new_order) - 1)
            if a != b and (a, b) not in swap_pairs and (b, a) not in swap_pairs:
                swap_pairs.append((a, b))
                break

    # 選ばれたインデックスペアを使って候補を試行
    for a, b in swap_pairs:
        # 2つの要素をスワップ
        candidate_order = new_order.copy()
        candidate_order[a], candidate_order[b] = candidate_order[b], candidate_order[a]

        # 損失を計算
        target_density = calculate_data_density(targets_it[:, candidate_order].float())
        candidate_loss = criterion(inputs_it.float(), targets_it[:, candidate_order].float(), current_density, target_density, loss_show=0)
        
        # 最良の順序を更新
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_order = candidate_order.copy()

    return best_order


def simulated_annealing_process(current_order, temperature, cooling_rate, iteration_num):
    return update_order_with_heuristics(current_order, temperature, iteration_num, num_trials=3, max_swap_count=5)

def loss_evaluation_func(spikes):
    return np.random.rand()  


def edit_data_name(data_name):
    # 'Subjects/' を除去し、最初の2つの '/' を '_' に置き換える
    new_data_name = data_name.replace('Subjects/', '').replace('/', '_', 2)
    
    # 日付部分（最後の '/' の後）を 'YYYYMMDD' 形式に変換
    date_part = new_data_name.split('_')[-1].replace('/', '')  # 最後の部分の日付から '/' を取り除く
    
    # 日付部分を置き換えた新しい名前を返す
    new_data_name = '_'.join(new_data_name.split('_')[:-1]) + '_' + date_part
    return new_data_name

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Assuming each element in batch is a tuple (data, label)
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data_padded = pad_sequence(data, batch_first=True)  # Pad the data to have the same length
    labels = torch.stack(labels, 0)
    return data_padded, labels

###############################################################################

group_IDs1 = list(range(1, 17))
group_IDs2 = list(range(1, 10)) 
## ###########################################################################
epoch_num = 200 # 60 # 2 # 150 # 3# 25 # 3 #   104 #25# 36# 71 # 35 # 3 #102 # 3 # 50 #10 # 20 #5 #350

for group_ID1 in group_IDs1:
## ###########################################################################
    region_date_train = {}
    region_date_test = {}
    init_cell_index_train = {}
    init_cell_index_test = {}
    
    for group_ID2 in group_IDs2:
            
            for i in range(1, 2):
               region_date_train[i] = region_date_train0[group_ID1][i]
               region_date_test[i]  = region_date_test0[group_ID2][i]
               
               init_cell_index_train[i] = init_cell_index_train0[group_ID1]
               init_cell_index_test[i]  = init_cell_index_test0[group_ID2]
                 
            ###############################################################################
            ###############################################################################
            class SplitDataset(Dataset):
                def __init__(self, data, *, split_size):
                    super().__init__() 
                    self.data = data.split(split_size)
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, index):
                    return self.data[index]
            
            ###############################################################################
            class transformer(nn.Module):
                def __init__(self, n_input, n_layer_list, n_hidden_list, n_output, n_heads, dropout):
                    super().__init__()
                    assert len(n_layer_list) == len(n_hidden_list), "Layer list and hidden list must have the same length."
            
                    # エンコーダーレイヤーの作成
                    self.encoder_layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=n_input if i == 0 else n_hidden_list[1],
                            nhead=n_heads,
                            dim_feedforward=n_hidden_list[i],
                            dropout=dropout,
                            batch_first=True
                        ) for i in range(len(n_layer_list))
                    ])
                    
                    # 一つの TransformerEncoder で全てのレイヤーを管理
                    self.transformer_encoder = nn.TransformerEncoder(
                        encoder_layer=self.encoder_layers[0],
                        num_layers=len(self.encoder_layers)
                    )
            
                    # 出力層
                    self.fc = nn.Linear(n_hidden_list[-1], n_output)
                    self.activation = nn.Sigmoid()
            
                def forward(self, inputs):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    inputs = inputs.to(device)
                    self.transformer_encoder = self.transformer_encoder.to(device)
                    outputs = self.transformer_encoder(inputs)
                    outputs = self.fc(outputs)
                    outputs = self.activation(outputs)  # Sigmoidを適用
                    return outputs
                
            ###############################################################################
            class Config(object):
            
                def read(self, path):
                    self.config = ConfigObj(path, encoding='utf-8')
                    self.logger = self._get('DEFAULT', 'logger')
                    self.cuda = self._getlist('DEFAULT', 'cuda', int)
                    self.use_segment = self._getint('DEFAULT', 'use_segment')
                    self.n_epoch = self._getint('DEFAULT', 'n_epoch')
                    self.split_size = self._getint('DEFAULT', 'split_size')
                    self.dim_input = self._getint('DEFAULT', 'dim_input')
                    self.train_rate = self._getfloat('DEFAULT', 'train_rate')
                    self.valid_rate = self._getfloat('DEFAULT', 'valid_rate')
                    self.dropout = self._getfloat('DEFAULT', 'dropout')
                    self.learning_rate = self._getfloat('DEFAULT', 'learning_rate')
                    self.batch_size = self._getint('DEFAULT', 'batch_size')
                    
                    self.n_layer_list = self._getlist('transformer', 'n_layer_list', int)
                    self.n_hidden_list = self._getlist('transformer', 'n_hidden_list', int)
                    self.ntoken = self._getint('transformer', 'ntoken')
                    self.n_heads = self._getint('transformer', 'n_heads')
                   # self.alpha   = self._getlist('transformer', 'alpha', float)
                   # self.gamma   = self._getlist('transformer', 'gamma', float)
                    
                    self.dir_base = self._get('path', 'base')
                    self.dir_spike = self._get('path', 'spike')
                    self.dir_result = self._get('path', 'result')
                    
                    self.threshold = self._getfloat('transformer', 'threshold_spike_counts') 
                  
                    return self
            
                def _getlist(self, section, option, func=int):
                    value = self.config[section][option]
                    if isinstance(value, str):
                        return [func(x.strip()) for x in value.split(',')]
                    else:
                        return [func(x) for x in value]
            
                def _get(self, section, option):
                    return self.config[section][option]
            
                def _getint(self, section, option):
                    return int(self.config[section][option])
            
                def _getfloat(self, section, option):
                    return float(self.config[section][option])
            
                def _getboolean(self, section, option):
                    return bool(self.config[section].as_bool(option))
                
            ###############################################################################
            
            class DiceLoss(nn.Module):
                def __init__(self, smooth=1.0, penalty_weight=0.1):
                    super(DiceLoss, self).__init__()
                    self.smooth = smooth
                    self.penalty_weight = penalty_weight
            
                def forward(self, inputs, targets, current_density, target_density,loss_show):
                    inputs = torch.sigmoid(inputs)
            
                    # 入力とターゲットの形状をフラットにする
                    inputs_flat = inputs.view(-1)
                    targets_flat = targets.view(-1)
            
                    # 密度のペナルティの計算
                    current_density_tensor = torch.tensor(current_density)
                    target_density_tensor = torch.tensor(target_density)
                    density_penalty = F.relu(current_density_tensor - target_density_tensor)
                    penalty = 10* density_penalty.mean()
                    
                    # インターセクションと損失の計算
                    intersection = (inputs_flat * targets_flat).sum()
                    dice_loss0 = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
                    dice_loss = torch.log(1 + dice_loss0)
        
                    if loss_show == 1: 
                        print("/////////////////////////////////////////////////////////////////////////////")
                        print("/////////////////////////////////////////////////////////////////////////////")
                        print("1000*DiceLoss: "+str(1000*dice_loss.item())+", 1000*Penalty: "+str(1000*penalty.item()))
                        print("Penalty/DiceLoss = "+str(penalty.item()/dice_loss.item()))
                        
                    total_loss = (1-self.penalty_weight) * dice_loss + self.penalty_weight * penalty
                        
                    if loss_show == 1: 
                        print("TotalLoss = " + str(1-self.penalty_weight) + " * DiceLoss + " + str(self.penalty_weight)+ " * Penalty = " + str(total_loss.item()))
                        print("/////////////////////////////////////////////////////////////////////////////")
                        print("/////////////////////////////////////////////////////////////////////////////")
            
                    return total_loss
            
            ###############################################################################
            NAME = 'groundtruth.npy'
            
            MODEL_FILENAME = 'spike.pt'
            MODEL_BEST_FILENAME = 'spike_best.pt'
            LOSS_IMG_FILENAME = 'loss.pdf'
            SPIKE_DENSITY_FILENAME = 'spike_density.pdf'
            ACCURACY_IMG_FILENAME = 'accuracy.pdf'
            OUTPUT_DIRNAME = 'output'
            OUTPUT_BEST_DIRNAME = 'output_best'
            
            GENERATED_TRAIN_FILENAME = 'generated_train.npy'
            GROUND_TRUTH_TRAIN_FILENAME = 'groundtruth_train.npy'
                    
            GENERATED_VALID_FILENAME = 'generated_valid.npy'
            GROUND_TRUTH_VALID_FILENAME = 'groundtruth_valid.npy'
            
            GENERATED_FILENAME = 'generated_test.npy'
            GROUND_TRUTH_FILENAME = 'groundtruth_test.npy'
            
            ###############################################################################
            class CustomStepLR:
                def __init__(self, optimizer, step_size, gamma, min_lr):
                    self.optimizer = optimizer
                    self.step_size = step_size
                    self.gamma = gamma
                    self.min_lr = min_lr
                    self.counter = 0
            
                def step(self):
                    "スケジューラーの更新ロジックを実装"
                    if self.counter % self.step_size == 0:
                        for param_group in self.optimizer.param_groups:
                            new_lr = param_group['lr'] * self.gamma
                            if new_lr < self.min_lr:
                                new_lr = self.min_lr
                            param_group['lr'] = new_lr
                    self.counter += 1
                    
            ###############################################################################
            class TransformerStackRunner:
                
                def _init_model(self):
                    return transformer(
                        n_input=self.config.dim_input,
                        n_layer_list=self.config.n_layer_list,  # ドット記法でアクセス
                        n_hidden_list=self.config.n_hidden_list,  # ドット記法でアクセス
                        n_output=self.config.dim_input,
                        n_heads=self.config.n_heads,  # ドット記法でアクセス
                        dropout=self.config.dropout,
                    )
                
                def __init__(self, mode, experiment_name, config):
                    super().__init__()
            
                    assert mode in ['train', 'eval', 'generate']
                    
                    self.mode = mode
                    self.config = config
                    self.experiment_name = experiment_name
                    self.dir_result = Path(self.config.dir_result)
                    self.model = self._init_model()
                #  self.threshold = config.threshold ################################
                    self.logger = getLogger(__name__)
                    self.device = torch.device(
                        f'cuda:{self.config.cuda[0]}'
                        if torch.cuda.is_available() else 'cpu')
                    self.modelpath = self.dir_result / MODEL_FILENAME
                    self.modelbestpath = self.dir_result / MODEL_BEST_FILENAME
                    
                    if self.modelbestpath.exists():
                        state_dict = torch.load(self.modelbestpath)
                        if state_dict is not None:
                            self.model.load_state_dict(state_dict, strict=False)
                        else:
                             print(f"Error: {self.modelbestpath} の内容が空です")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("モデル spike_best.pt が正常にロードされました 〇〇〇〇〇〇〇〇〇〇〇〇〇〇〇")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                    else:
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("モデル  spike_best.pt が見つかりません　×××××××××××××××××××××")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        print("-----------------------------------------------------------------------------")
                        
                    self.criterion = self._init_criterion()
                    self.optimizer = self._init_optimizer()
                    self.scheduler = CustomStepLR(self.optimizer, step_size=5, gamma=0.9, min_lr=0.000025)
                  
                    if len(self.config.cuda) == 1:
                        self.model = self.model.to(self.device)
                    else:
                        self.model = nn.DataParallel(
                            self.model, device_ids=self.config.cuda)
                        self.model = self.model.to(self.device)
                    
                    self.threshold_updated = False
                    
                    if mode == 'train':
                    
                        # 初期閾値を設定
                        print("閾値を初期化している at __init__ of TransformerStackRunner")
                        self.threshold = config.threshold
                        print("initial thr.:"+str(self.threshold))
                    
                        
                        # データの接続密度を計算
                        self.target_density = self._calculate_data_density()
                        print("-----------------------------------------------------------------------------")
                        
                        self.previous_loss = None
                        self.previous_threshold = 1.0
                        
                    
                def _calculate_data_density(self):
                    print("In _calculate_data_density of TransformerStackRunner")
                    dataloader_train, _, _ = self._load_data()
                    total_spikes = 0
                    total_elements = 0
                    for data in dataloader_train:
                        total_spikes += torch.sum(data)
                        total_elements += data.numel()
                    return total_spikes.item() / total_elements
                
                def _init_criterion(self):
                   # return  DiceLoss(smooth=1.0) # DiceLoss(smooth=0.1) #
                    return DiceLoss(smooth=1.0, penalty_weight=0.1)
            
                def _init_optimizer(self):
                    return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
                   # return optim.Adam(self.model.parameters(), lr=0.01)
                    
                def run(self, *args, **kwargs):
                    
                    if self.mode == 'train':
                        self._train(*args, **kwargs)
                        
                        # モデルの学習が完了したら保存
                        torch.save(self.model.state_dict(), self.modelpath)
                        
                    elif self.mode == 'eval':
                        self._eval(*args, **kwargs)
                        if self.modelbestpath.exists():
                            self.model.load_state_dict(torch.load(self.modelpath), strict=False)
                            
                            self._eval(*args, **kwargs)
                    elif self.mode == 'generate':
                        self._generate(*args, **kwargs)
            
            
            
                def _train(self, *args, generate_mode, quantization_mode, **kwargs):
                    print("-----------------------------------------------------------------------------")
                    print("In _train of TransformerStackRunner")
                    dataloader_train, dataloader_valid, dataloader_test = self._load_data()
                  #  writer = SummaryWriter(comment=self.experiment_name)
                    log_dir = os.path.join("runs", "experiment_logs")
                    writer = SummaryWriter(log_dir=log_dir, comment=self.experiment_name)
                    results_train, results_valid = ({'loss': [], 'accuracy': {'all': [], 'spike': [], 'nonspike': []}} for _ in range(2))
                    spike_density_train_all = []
                    spike_density_valid_all = []
                    best_valid = {'accuracy': float('-inf'), 'model': None}
                    
                    # #################################################
                    # #################################################
                    # 学習開始時に閾値を事前に調整している
                    # #################################################
                    sample_data = next(iter(dataloader_train))
                    print(f"Data shape: {sample_data.shape}")
                    with torch.no_grad():
                        sample_output = self.model(sample_data.to(self.device))
                        print("この上のmodelで閾値を初期化してしまっている！！")
                     
                    
                    print("----First definition of threshod----------------")
                    self.previous_threshold = adjust_threshold_init(sample_output, self.target_density, initial_threshold=self.threshold)
                    self.first_threshold = self.previous_threshold
                    self.threshold = self.previous_threshold
                    print(f"Threshold in training: {self.threshold}")
                    
                    # #################################################
                    # #################################################
                    
                    print(f"Sample data shape before transformer: {sample_output.shape}")
                    
                    
                    def run_epoch(dataloader, mode='train'):
                        # トレーニングと評価モードの切り替え
                        self.model.train() if mode == 'train' else self.model.eval()
                        
                        # 初期条件
                        loss, sample_number_sum, correct_count_sum = 0, {'all': 1, 'spike': 1, 'nonspike': 1}, {'all': 1, 'spike': 1, 'nonspike': 1}
                        spike_density = 10
                        total_loss = 0.0  # Initialize total_loss before starting the loop
                        
                        loss_diff = 1
                       # self.previous_threshold = 1.0
                        num_batches = len(dataloader)
                    
                        if self.previous_loss is None:
                            self.previous_loss = float('inf')  # もしくは、大きな値や0など、初期の比較に使える適切な値に設定
                        
                        num_batches = 0
                        for data in dataloader:
                            num_batches += 1
                            
                            print("-----------------------------------------------------------------------------")
                            print("Mode: ", mode)
                            print(f"Threshold before _process_batch: {self.threshold}")
                            
                            # バッチ処理
                            loss_value, output = self._process_batch(data, mode2=mode)
                           # total_loss += loss_value.item()
                            if isinstance(loss_value, torch.Tensor):
                                total_loss += loss_value.item()
                            else:
                                total_loss += loss_value
                            
                            print(f"Threshold after _process_batch: {self.threshold}")
                            epoch_average_loss = total_loss / num_batches
                            loss_diff = abs(loss_value - self.previous_loss)
        
                            loss_diff_thr = 1e-5 ######################################
                            
                            print("output:" + str(np.shape(output)))
                                
                            if mode == 'train':
                                
                                # 適応的にしきい値を調整
                                print(f"Current loss: {epoch_average_loss}, previous loss: {self.previous_loss}")
                                
                                print("Currently, thresholding criterion is ROC dependent method. (2)")
                                # まだ収束していない場合はROCカーブに基づくしきい値
                                aa, roc_auc = adjust_threshold_ROC_with_max_fpr(output, data[:, 1:], max_fpr=0.005)
                                
                                print(data)
                                
                                print("aa: "+str(aa))
                                print("previous_threshold: "+str( self.previous_threshold))
                            #    self.threshold = self.previous_threshold
                                if math.isinf(aa) and math.isinf(self.threshold):
                                    self.threshold = 0.5 * aa + 0.5 * self.previous_threshold
                                else:
                                    self.threshold = aa
                                self.previous_threshold = self.threshold
                                print("What!!!?   (pre-converge) : "+str(self.previous_threshold))
                                 

                                # data[:, 1:] の内容と形状の確認
                                print("Target values:", data[:, 1:])
                                print("Target shape:", data[:, 1:].shape)
                                
                                # ROCのプロットを保存
                                print("output:" + str(np.shape(output)))
                                
                                print("outpu: "+str(output))
                                print("data: "+str(data))
                            
                                roc_auc, fpr, tpr, thresholds = plot_ROC(output, data[:, 1:])
                                plt.savefig(f"{gen_dir}/ROC_train_for_thr_epoch_{epoch + 1}.pdf")
                                plt.close()
                                  
                                file_path = os.path.join(f"{gen_dir}/roc_auc_train_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{roc_auc}")
                          
                                file_path = os.path.join(f"{gen_dir}/fpr_train_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{fpr}")
                                
                                file_path = os.path.join(f"{gen_dir}/tpr_train_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{tpr}")
                                
                                file_path = os.path.join(f"{gen_dir}/hresholds_train_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{thresholds}")
                    
                    
                            elif mode =='valid':
                                if loss_diff < loss_diff_thr: # or self.threshold_updated == True:
                                   self.threshold_updated = True
                                
                                unique_values = torch.unique(data[:, 1:])
                                print("Unique values in target:", unique_values)
                                target = torch.where(data > 0, 1.0, 0.0)
                                
                                      
                                roc_auc, fpr, tpr, thresholds = plot_ROC(output, data[:, 1:])
                                
                                print("==== results for valid data ====")
                                print("roc_auc:"+str(roc_auc))
                                print("fpr:"+str(fpr))
                                print("tpr:"+str(tpr))
                                print("thresholds:"+str(thresholds))
                                print("====------------------------====")
                                
                                plt.savefig(f"{gen_dir}/ROC_valid_for_thr_epoch_{epoch + 1}.pdf")
                                plt.close()
                                
                                file_path = os.path.join(f"{gen_dir}/roc_auc_valid_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{roc_auc}")
                        
                                file_path = os.path.join(f"{gen_dir}/fpr_valid_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{fpr}")
                                
                                file_path = os.path.join(f"{gen_dir}/tpr_valid_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{tpr}")
                                
                                file_path = os.path.join(f"{gen_dir}/hresholds_valid_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{thresholds}")
                                
                            elif mode =='generate':
                            
                                roc_auc, fpr, tpr, thresholds = plot_ROC(output, data[:, 1:])
                                plt.savefig(f"{gen_dir}/ROC_test_for_thr_epoch_{epoch + 1}.pdf")
                                plt.close()
                                
                                
                                file_path = os.path.join(f"{gen_dir}/roc_auc_generate_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{roc_auc}")
                        
                                file_path = os.path.join(f"{gen_dir}/fpr_generate_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{fpr}")
                                
                                file_path = os.path.join(f"{gen_dir}/tpr_generate_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{tpr}")
                                
                                file_path = os.path.join(f"{gen_dir}/thresholds_generate_epoch_{epoch + 1}.txt")
                                with open(file_path, 'w') as f:
                                    f.write(f"{thresholds}")
                                
                            # エポックの損失を記録
                            if mode == 'train':
                                # エポック全体の平均損失を保存
                                self.previous_loss = total_loss / num_batches
                            
                            
                            #######################################################################
                            #######################################################################
                            unique_values = output.detach().cpu().numpy()
                            
                            plt.figure(figsize=(12, 10))
                            
                            # Plot original histogram
                            plt.subplot(2, 2, 1)
                            plt.hist(unique_values.ravel(), bins=50, color='blue', alpha=0.7)
                            plt.xlabel('Value', fontsize=14)
                            plt.ylabel('Frequency', fontsize=14)
                            plt.title('Histogram of Unique Values (Original Axis)', fontsize=16)
                            plt.tick_params(axis='both', which='major', labelsize=12)
                            plt.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2)
                            
                            # Plot log10 histogram
                            plt.subplot(2, 2, 2)
                            plt.hist(np.log10(unique_values.ravel() + 1e-10), bins=50, color='green', alpha=0.7)
                            plt.xlabel('Log10(Value)', fontsize=14)
                            plt.ylabel('Frequency', fontsize=14)
                            plt.title('Histogram of Unique Values (Log10 Axis)', fontsize=16)
                            plt.tick_params(axis='both', which='major', labelsize=12)
                            threshold_value = max(self.threshold, 1e-10)  # 負の値を防ぐために最小値を設定
                            plt.axvline(x=np.log10(threshold_value), color='red', linestyle='--', linewidth=2)
                            
                            # Plot original histogram with fixed xlim
                            plt.subplot(2, 2, 3)
                            if self.threshold_updated:
                                counts, bins, _ = plt.hist(unique_values.ravel(), bins=1000, color='blue', alpha=0.7)
                            else:
                                counts, bins, _ = plt.hist(unique_values.ravel(), bins=50, color='blue', alpha=0.7)
        
                            plt.xlabel('Value', fontsize=14)
                            plt.ylabel('Frequency', fontsize=14)
                            plt.title('Histogram of Unique Values (Original Axis, Limited)', fontsize=16)
                            plt.tick_params(axis='both', which='major', labelsize=12)
                            plt.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2)
                            
                            if self.threshold_updated:
                                # Find the bins that fall within the specified xlim range
                                xlim_min, xlim_max = 0.94, 1.0
                                bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
                                filtered_counts = counts[(bin_centers >= xlim_min) & (bin_centers <= xlim_max)]
                                max_count = filtered_counts.max() if filtered_counts.size > 0 else 0
                                
                                # Set ylim to 1.1 times the maximum count within the xlim range
                                plt.xlim(xlim_min, xlim_max)
                                plt.ylim(0, max_count * 1.1)
                            else:
                                plt.xlim(-0.05, 1.05)
                             
                            
                            # Plot log10 histogram with fixed xlim
                            plt.subplot(2, 2, 4)
                            if self.threshold_updated == True:
                               plt.hist(np.log10(unique_values.ravel() + 1e-10), bins=5000, color='green', alpha=0.7)
                            else:
                               plt.hist(np.log10(unique_values.ravel() + 1e-10), bins=50, color='green', alpha=0.7)
                            plt.xlabel('Log10(Value)', fontsize=14)
                            plt.ylabel('Frequency', fontsize=14)
                            plt.title('Histogram of Unique Values (Log10 Axis, Limited)', fontsize=16)
                            plt.tick_params(axis='both', which='major', labelsize=12)
                            plt.axvline(x=np.log10(self.threshold + 1e-10), color='red', linestyle='--', linewidth=2)
                            
                            if self.threshold_updated == True:
                               plt.xlim(-0.03, 0.0)
                            else:
                               plt.xlim(-8.05, 0.05)
                            
                            plt.tight_layout(pad=3.0)
                            
                            #######################################################################
                            #######################################################################
                            
                            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                            print(f"Threshold after adjust_threshold: {self.threshold}")
                            
                            output = (output > self.threshold).float()
                            
                            print("Binary output unique values:", np.unique(output.detach().cpu().numpy()))
                            print("1000 * Spike Density (Training and Validation) : " + str(1000 * output.sum().item() / output.numel()))
                            
                            loss += loss_value * data.numel()
                            spike_density = 1000 * output.sum().item() / output.numel()
                            for key, value in self._get_threshold_number(data).items():
                                sample_number_sum[key] += value
                            for key, value in self._calculate_correct_threshold_number(data[:, 1:], output).items():
                                correct_count_sum[key] += value
                                
                        return self.threshold, spike_density, loss / sample_number_sum['all'], {k: v / sample_number_sum[k] if sample_number_sum[k] != 0 else 0 for k, v in correct_count_sum.items()}
                    
                    try:
                        
                        self.threshold_updated = False
                        num_epochs = self.config.n_epoch
                        threshold_all = [0] * num_epochs
                        
                        for epoch in range(self.config.n_epoch):
                            start_time = time.time()  # エポック開始時刻を記録
                            print("=============================================================================")
                            print("Epoch #: " + str(epoch + 1))
                            print("Running for training data")  # 処理時間を表示
                            threshold, spike_density_train, loss_train, accuracy_train = run_epoch(dataloader_train, 'train')
                            
                            txt_filename = f"threshold_{epoch + 1}.txt"  # 閉じられていない引用符を修正
                            file_path = os.path.join(self.dir_result, txt_filename)
                            with open(file_path, 'w') as f:
                                f.write(f"{threshold}")
                                
                            file_path = os.path.join(self.dir_result, 'threshold.txt')
                            with open(file_path, 'w') as f:
                                f.write(f"{threshold}")
                            
                            print("Done (training data)")  # 処理時間を表示
                            
                            # Save the figure
                            plt.tight_layout()
                            plt.savefig(f"{gen_dir}/GenSigHist_Train_epoch_{epoch + 1}.pdf")
                            plt.close()
                            
                            print("-----------------------------------------------------------------------------")
                            print("Running for validation data")  # 処理時間を表示
                            threshold, spike_density_valid, loss_valid, accuracy_valid = run_epoch(dataloader_valid, 'valid')
                            print("Done (validation data)")  # 処理時間を表示
                            
                            self.threshold = threshold
                            threshold_all[epoch] = threshold
                            
                            # Save the figure
                            plt.tight_layout()
                            plt.savefig(f"{gen_dir}/GenSigHist_Valid_epoch_{epoch + 1}.pdf")
                            plt.close()
                            
                            results_train['loss'].append(loss_train)
                            results_valid['loss'].append(loss_valid)
                            
                            spike_density_train_all.append(spike_density_train)
                            spike_density_valid_all.append(spike_density_valid)
                            
                            for key in accuracy_train:
                                results_train['accuracy'][key].append(accuracy_train[key])
                                results_valid['accuracy'][key].append(accuracy_valid[key])
                            
                            writer.add_scalars(f'loss/{self.experiment_name}', {'train': loss_train, 'valid': loss_valid}, global_step=epoch)
                            writer.add_scalars(f'accuracy/{self.experiment_name}', {'train': accuracy_train['all'], 'valid': accuracy_valid['all']}, global_step=epoch)
                            
                            self.logger.info(f'train: {epoch:4}, loss: {loss_train:.6}, accu: {accuracy_train}')
                            self.logger.info(f'valid: {epoch:4}, loss: {loss_valid:.6}, accu: {accuracy_valid}')
                            
                            if best_valid['accuracy'] < accuracy_valid['all']:
                                best_valid['accuracy'], best_valid['model'] = accuracy_valid['all'], self.model.state_dict() if len(self.config.cuda) == 1 else self.model.module.state_dict()
                            
                            self.scheduler.step()  # スケジューラーを更新
                            print(f"Epoch {epoch + 1}, Learning Rate: {[group['lr'] for group in self.optimizer.param_groups]}")
                            
                            if epoch % 10 == 4:
                                self._savemodel(best_valid['model'])
                                self._savefig(results_train, results_valid)
                            #    self._savefig2(spike_density_train_all, spike_density_valid_all)
                                self._savefig2(spike_density_train_all, spike_density_valid_all)
                                
                            end_time = time.time()  # エポック終了時刻を記録
                            duration = end_time - start_time  # 処理時間を計算
                            print(f"Epoch {epoch+1} completed in {duration:.2f} seconds")  # 処理時間を表示
                            
                        #################################################################
                        print('Added the threshold plot')
        
                        # 最初のサブプロット
                        plt.subplot(2, 1, 1)  # 2行1列の最初のサブプロット
                        plt.plot(threshold_all, label='Threshold Values')
                        plt.xlabel('Epochs', fontsize=8)
                        plt.ylabel('Threshold', fontsize=8)
                        plt.title('Threshold Over Time', fontsize=12)
                        plt.legend()
                        plt.grid(True)
                        
                        # 2つ目のサブプロット
                        plt.subplot(2, 1, 2)  # 2行1列の2つ目のサブプロット
                        plt.plot(np.abs(threshold_all[-1]-threshold_all+0.0000000001), label='Threshold Values')
                        plt.xlabel('Iteration steps', fontsize=8)
                        plt.ylabel('Threshold Value', fontsize=8)
                        plt.title('Threshold Changes Over Iterations', fontsize=12)
                        plt.legend()
                        plt.grid(True)
                         # PDFで保存
                        plt.savefig(gen_dir + '/Threshold_Adjustment.pdf')
                        
                       #  plt.show()
                        plt.close()
                        
                        #################################################################
                        
                        print("Generation for training data started")  # 処理時間を表示
                        start_time0 = time.time()  # エポック開始時刻を記録
                        torch.backends.cudnn.enabled = True #  False ###########################
                        train_data, _, _ = self._load_data(raw=True)
                        train_data = train_data[None, :, :]  # add batch dimension
                        result_train = self._generate_from_data(train_data, quantization_mode).to('cpu').numpy()
                        plt.savefig(f"{gen_dir}/GenSigHist_BeforeThreshold_batch_Train.pdf")
                        plt.close()
                        
                        np.save(self.dir_result / f'{generate_mode}_{quantization_mode}_{GROUND_TRUTH_TRAIN_FILENAME}', train_data[0].numpy().astype(np.bool_))
                        np.save(self.dir_result / f'{generate_mode}_{quantization_mode}_{GENERATED_TRAIN_FILENAME}', result_train[0].astype(np.bool_))
                        
                        end_time0 = time.time()  # エポック終了時刻を記録
                        duration0 = end_time0 - start_time0  # 処理時間を計算
                        print(f"Generation for training data completed in {duration0:.2f} seconds")  # 処理時間を表示
                            
                    except KeyboardInterrupt:
                        self.logger.error(traceback.format_exc())
                        self._savemodel(best_valid['model'])
                        self._savefig(results_train, results_valid)
                        self._savefig2(spike_density_train_all, spike_density_valid_all)
                        raise
                    
                    finally:
                        # トレーニングが完了したらSummaryWriterを閉じる
                        writer.close()
                    
                    print("Saving model")  # 処理時間を表示
                    start_time1 = time.time()  # エポック開始時刻を記録
                    self._savemodel(best_valid['model'])
                    self._savefig(results_train, results_valid)
                    self._savefig2(spike_density_train_all, spike_density_valid_all)
                    
                    end_time1 = time.time()                          # エポック終了時刻を記録
                    duration1 = end_time1 - start_time1              # 処理時間を計算
                    print(f"Done in {duration1:.2f} seconds")        # 処理時間を表示
                
                    print("Generation for test data started")
                    start_time2 = time.time()                        # エポック開始時刻を記録
                    threshold, spike_density_test, _, accuracy_test = run_epoch(dataloader_test, 'generate')
                    self.threshold = threshold
                    print( "1000 * Spike density(test): " + str(spike_density_test) )
                    self.logger.info(f'test accuracy: {accuracy_test}')
                    
                    end_time2 = time.time()                          # エポック終了時刻を記録
                    duration2 = end_time2 - start_time2              # 処理時間を計算
                    print(f"Done in {duration2:.2f} seconds")        # 処理時間を表示
                        
                    print("Training completed.")
            
                def _get_threshold_number(self, data):
                    sample_number_all = data.numel()
                    sample_number_spike = data[data == 1].numel()
                    sample_number_nonspike = sample_number_all - sample_number_spike
                    return {
                        'all': sample_number_all,
                        'spike': sample_number_spike,
                        'nonspike': sample_number_nonspike,
                    }
                    
                ############################################################
                def _eval(self, *args, **kwargs):
                    print("-----------------------------------------------------------------------------")
                    print("_eval:")
                    print(f"Threshold in _eval: {self.threshold}")
                    _, _, dataloader_test = self._load_data()
                
                    self.model.eval()
                    correctsample_number_sum = {'all': 0, 'spike': 0, 'nonspike': 0}
                    count_sum = {'all': 0, 'spike': 0, 'nonspike': 0}
                    for data in dataloader_test:
                        _, output = self._process_batch(data, mode2='valid') 
                        print( "Summed value" + str(sum(sum(output))/(np.size(output)[0]*np.size(output)[0])))
                        
                        for key, value in self._get_threshold_number(data).items():
                            sample_number_sum[key] += value
                        for key, value in self._calculate_correct_threshold_number(data[:, 1:], output).items():
                            correct_count_sum[key] += value
                
                    accuracy_eval = {key: correct / sample if sample else 0 for key, (correct, sample) in zip(correct_count_sum.keys(), zip(correct_count_sum.values(), sample_number_sum.values()))}
                    self.logger.info(f'eval accuracy: {accuracy_eval}')
                    
                ############################################################
                def _generate(self, *args, generate_mode, quantization_mode, **kwargs):
                    print("-----------------------------------------------------------------------------")
                    print("generate:")
                        
                    file_path = os.path.join(self.dir_result, 'threshold.txt')
                    with open(file_path, 'r') as f:
                        threshold_value = f.read()
                    
                    # 読み込んだ値を float に変換して self.threshold に代入
                    self.threshold = float(threshold_value)
                    
                    print(f"Threshold in generate: {self.threshold}")
                    
                    _, valid_data, test_data = self._load_data(raw=True)
                    test_data = test_data[None, :, :]  # add batch dimension
                    valid_data = valid_data[None, :, :]  # add batch dimension
                
                    self.model.eval()
                    
                    torch.backends.cudnn.enabled = False ###########################
                    
                    result_valid = self._generate_from_data(valid_data, quantization_mode).to('cpu').numpy()
                    
                    #####################################################################
                    print("valid_data:" + str(np.shape(valid_data)))
                    print("result_valid:" + str(np.shape(result_valid)))
                          
                    roc_auc0, fpr0, tpr0, thresholds0 = plot_ROC(valid_data, result_valid)
                    plt.savefig(f"{gen_dir}/ROC_test_for_thr_epoch_final_befthr.pdf")
                    
                    plt.close()
                    
                    file_path = os.path.join(f"{gen_dir}/roc_auc_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{roc_auc0}")
             
                    file_path = os.path.join(f"{gen_dir}/fpr_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{fpr0}")
                     
                    file_path = os.path.join(f"{gen_dir}/tpr_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{tpr0}")
                     
                    file_path = os.path.join(f"{gen_dir}/thresholds_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{thresholds0}")
                        
                    #####################################################################
                    
                    if isinstance(result_valid, np.ndarray):
                        result_valid = torch.tensor(result_valid)
    
                    result_valid = (result_valid > self.threshold).float()
                    
                    #####################################################################
                    print("valid_data:" + str(np.shape(valid_data)))
                    print("result_valid:" + str(np.shape(result_valid)))
                          
                    plt.savefig(f"{gen_dir}/GenSigHist_BeforeThreshold_batch_Valid.pdf")
                    plt.close()
                    
                    roc_auc, fpr, tpr, thresholds = plot_ROC(valid_data, result_valid)
                    plt.savefig(f"{gen_dir}/ROC_valid_for_thr_epoch_final.pdf")
                    # plt.show()
                    plt.close()
                     
                    file_path = os.path.join(f"{gen_dir}/roc_auc_valid_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{roc_auc}")
             
                    file_path = os.path.join(f"{gen_dir}/fpr_valid_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{fpr}")
                     
                    file_path = os.path.join(f"{gen_dir}/tpr_valid_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{tpr}")
                     
                    file_path = os.path.join(f"{gen_dir}/thresholds_valid_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{thresholds}")
                                    
                    #####################################################################
                    result = self._generate_from_data(test_data, quantization_mode).to('cpu').numpy()
                    
                    #####################################################################
                    print("test_data:" + str(np.shape(test_data)))
                    print("result:" + str(np.shape(result)))
                          
                    roc_auc0, fpr0, tpr0, thresholds0 = plot_ROC(test_data, result)
                    plt.savefig(f"{gen_dir}/ROC_test_for_thr_epoch_final_befthr.pdf")
                    # plt.show()
                    plt.close()
                    
                    file_path = os.path.join(f"{gen_dir}/roc_auc_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{roc_auc0}")
             
                    file_path = os.path.join(f"{gen_dir}/fpr_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{fpr0}")
                     
                    file_path = os.path.join(f"{gen_dir}/tpr_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{tpr0}")
                     
                    file_path = os.path.join(f"{gen_dir}/thresholds_test_generate_epoch_final_befthr.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{thresholds0}")
                        
                    #####################################################################

                    # result_valid が numpy.ndarray である場合、torch.Tensor に変換
                    if isinstance(result, np.ndarray):
                        result = torch.tensor(result)
    
                    result = (result > self.threshold).float()
                    
                    #####################################################################
                    plt.savefig(f"{gen_dir}/GenSigHist_BeforeThreshold_batch_Test.pdf")
                    plt.close()
                    
                    print("test_data:" + str(np.shape(test_data)))
                    print("result:" + str(np.shape(result)))
                          
                    roc_auc, fpr, tpr, thresholds = plot_ROC(test_data, result)
                    plt.savefig(f"{gen_dir}/ROC_test_for_thr_epoch_final.pdf")
                    # plt.show()
                    plt.close()
                    
                    file_path = os.path.join(f"{gen_dir}/roc_auc_test_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{roc_auc}")
             
                    file_path = os.path.join(f"{gen_dir}/fpr_test_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{fpr}")
                     
                    file_path = os.path.join(f"{gen_dir}/tpr_test_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{tpr}")
                     
                    file_path = os.path.join(f"{gen_dir}/thresholds_test_generate_epoch_final.txt")
                    with open(file_path, 'w') as f:
                          f.write(f"{thresholds}")
                        
                    #####################################################################
                    if isinstance(result_valid, torch.Tensor):
                        result_valid = result_valid.cpu().numpy()  # numpy.ndarray に変換
                    if isinstance(result, torch.Tensor):
                        result = result.cpu().numpy()  # numpy.ndarray に変換

                    np.save(self.dir_result / f'{generate_mode}_{quantization_mode}_{GROUND_TRUTH_VALID_FILENAME}', valid_data[0].numpy().astype(np.bool_))
                    np.save(self.dir_result / f'{generate_mode}_{quantization_mode}_{GENERATED_VALID_FILENAME}', result_valid[0].astype(np.bool_))
                        
                    np.save(self.dir_result / f'{generate_mode}_{quantization_mode}_{GROUND_TRUTH_FILENAME}', test_data[0].numpy().astype(np.bool_))
                    np.save(self.dir_result / f'{generate_mode}_{quantization_mode}_{GENERATED_FILENAME}', result[0].astype(np.bool_))
                    
                ############################################################
                def _calculate_correct_threshold_number(self, data, output):
                    data = data.flatten()
                    output = output.detach().round().flatten()
                    spike_data, spike_output = data[data == 1], output[data == 1]
            
                    correct_count = metrics.accuracy_score(
                        data, output, normalize=False)
                    correct_count_spike = metrics.accuracy_score(
                        spike_data, spike_output, normalize=False)
                    correct_count_nonspike = correct_count - correct_count_spike
                    return {
                        'all': correct_count,
                        'spike': correct_count_spike,
                        'nonspike': correct_count_nonspike,
                    }
                    
                ############################################################
                
                def _savefig(self, results_train, results_valid):
                    # 損失のプロット
                    plt.figure()
                    plt.plot(results_train['loss'], label='train')
                    plt.plot(results_valid['loss'], label='valid')
                    plt.yscale('log')
                    plt.legend()
                    plt.title(f'Final Train Loss: {results_train["loss"][-1]:.4f}, Final Valid Loss: {results_valid["loss"][-1]:.4f}')
                    plt.savefig(self.dir_result / LOSS_IMG_FILENAME)
                    plt.close()
                    
                    # 精度のプロット
                    _, axes_list = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
                    panel_titles = [
                        'Overall Training and Validation Accuracy (all-all)',
                        'Spike Training and Validation Accuracy (true-positive)',
                        'Non-Spike Training and Validation Accuracy (false-negative)'
                    ]
                    
                    for i, (axes, accuracy_train, accuracy_valid, title) in enumerate(zip(axes_list, results_train['accuracy'].values(), results_valid['accuracy'].values(), panel_titles)):
                        axes.plot(accuracy_train, label='train')
                        axes.plot(accuracy_valid, label='valid')
                        axes.set_title(f'{title}\nFinal Train Accuracy: {accuracy_train[-1]:.4f}, Final Valid Accuracy: {accuracy_valid[-1]:.4f}')
                       # axes.set_yscale('log')
                        axes.legend()
                        if i < 2:  # 上の二つのパネルのx軸の値を非表示
                            axes.set_xticklabels([])
                    
                    plt.tight_layout()
                    plt.savefig(self.dir_result / ACCURACY_IMG_FILENAME)
                    plt.close()
                    
                ############################################################
                    
                def _savefig2(self, spike_density_train_all, spike_density_valid_all):
                    
                    plt.figure()
                    plt.plot(spike_density_train_all, label='train')
                    plt.plot(spike_density_valid_all, label='valid') # これがゼロになるのはなぜ？
                 #   plt.yscale('linear')
                 #   plt.yscale('log')
                    plt.legend()
                    plt.title('1000 * Spike Density (Training and Validation)')
                    plt.savefig(self.dir_result / SPIKE_DENSITY_FILENAME)
                    plt.close()
                #    print(f"Plot saved to {self.dir_result / SPIKE_DENSITY_FILENAME}")
                    
                def _savemodel(self, best_valid_model):
                    torch.save(best_valid_model, self.modelbestpath)
                    if len(self.config.cuda) == 1:
                        torch.save(self.model.state_dict(), self.modelpath)
                    else:
                        torch.save(self.model.module.state_dict(), self.modelpath)
                
                ############################################################
                def _process_batch(self, data, mode2='train'):
                    print("In _process_batch...")
                    data = data.to(self.device, dtype=torch.float32)  # デバイスへの転送とデータ型の変換
                    
                    # オプティマイザの勾配をリセット
                    if mode2=='train':
                        self.optimizer.zero_grad()
                        
                    output = self.model(data)
                    print("Output unique values:", np.unique(output.detach().cpu().numpy()))
                    
                    print(f"Mode id {mode2} - Updated Threshold: {self.threshold}")
        
                    # print(f"****************************************************************")
                    # if mode2 == 'generate':
                    #     output = (output > self.threshold).float()
                    #     print(f"Threshold of validation and test in _process_batch: {self.threshold}")
                    #     print("Currently, the thresholding process is working!!")
                    # else:
                    #     print(f"Threshold of training in _process_batch: {self.threshold}")
                    # print(f"****************************************************************")
                    
                    print("_process_batch done.")
                    # 出力データの寸法を確認し、必要に応じて調整
                    output = output[:, 1:]  # 最後の時間ステップを除外（予測値の寸法調整）
                    target = data[:, 1:]  # 次の時刻のデータを正解ラベルとして使用
                    
                    if target.dtype != torch.long:
                        target = target.long()  # 対象データの型を整数型に変換
                    
                    # 損失を計算
                    current_density, _, _ = calculate_spike_density(output.detach().cpu().numpy())
                    current_density = torch.tensor(current_density, dtype=output.dtype).to(output.device)
                    current_density = current_density.expand_as(output)
                    print("output shape:", output.shape)
                    print("target shape:", target.shape)
                    print("current_density shape:", current_density.shape if isinstance(current_density, np.ndarray) else "Not an ndarray")
                    print("current_density type:", type(current_density))
                    
                    loss = self.criterion(output, target, current_density, self.target_density,loss_show=1)
               #     loss = self.criterion(output, target)
                    
                    
                    formatted_loss = f"{loss.item():.20f}"
                    calculated_loss = loss.item()  # loss の値自体を 100,000 倍する
                    print(f"Calculated loss: {calculated_loss:.20f}")  # 結果を表示
                    
                    # 訓練時の処理
                   # if is_train:
                    if mode2=='train':
                        print("Now, updating weights by back-propagations...")
                        loss.backward()  # 誤差逆伝播
                        self.optimizer.step()  # オプティマイザの更新
                        
                    return loss.item(), output.detach().cpu()  # 損失と出力データを返す
                
                ############################################################
                def _load_data(self, raw=False):
                    data = np.load(self.config.dir_spike)  # データをロード
                    
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    print("  Data is loaded from "+ str(self.config.dir_spike))
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    print(" ############################################################# ")
                    
                    data = np.squeeze(data)  # 不要な次元を除去
                    
                    if len(data.shape) != 2:
                        raise ValueError(f"Data shape {data.shape} is not compatible. Expected 2 dimensions.")
                
                    data_length, dim = data.shape
                
                    train_size = int(data_length * self.config.train_rate)
                    valid_size = int(data_length * self.config.valid_rate)
                    test_size  = int(data_length * (1 - self.config.train_rate - self.config.valid_rate))
                    
                    used_test_order = np.random.permutation(128)
                    np.save(self.dir_result / "random_order.npy", used_test_order)
                
                    train_data = torch.as_tensor(data[: train_size, :], dtype=torch.float32)
                    test_data  = torch.as_tensor(data[train_size : train_size+test_size, :], dtype=torch.float32)[:, used_test_order]  # シャッフルされた順序でテストデータを取得
                    valid_data = torch.as_tensor(data[train_size+test_size :, :], dtype=torch.float32)
                 #   valid_data = torch.as_tensor(data[train_size+test_size : train_size+test_size+valid_size, :], dtype=torch.float32)
                    
                    # Calculate and print the mean and variance of the firing rate for each dataset
                  #  valid_mean = torch.mean(torch.mean(valid_data, dim=0).values).values
                    valid_mean = torch.mean(valid_data, dim=0).mean()
                    valid_variance = torch.mean(torch.var(valid_data, dim=0))
                    print("Validation Data - Mean of Firing Rate:", valid_mean)
                    print("Validation Data - Variance of Firing Rate:", valid_variance)
                    
                  #  train_mean = torch.mean(torch.mean(train_data, dim=0).values).values
                    train_mean = torch.mean(train_data, dim=0).mean()
                    train_variance = torch.mean(torch.var(train_data, dim=0))
                    print("Training Data - Mean of Firing Rate:", train_mean)
                    print("Training Data - Variance of Firing Rate:", train_variance)
                    
                   # test_mean = torch.mean(torch.mean(test_data, dim=0).values).values
                    test_mean = torch.mean(test_data, dim=0).mean()
                    test_variance = torch.mean(torch.var(test_data, dim=0))
                    print("Test Data - Mean of Firing Rate:", test_mean)
                    print("Test Data - Variance of Firing Rate:", test_variance)
                    
                    if raw:
                        return train_data, valid_data, test_data
                    
                    data_train = SplitDataset(train_data, split_size=self.config.split_size)
                    data_valid = SplitDataset(valid_data, split_size=self.config.split_size)
                    data_test  = SplitDataset(test_data , split_size=self.config.split_size)
                
                    dataloader_train = DataLoader(data_train, batch_size=self.config.batch_size)
                    dataloader_valid = DataLoader(data_valid, batch_size=self.config.batch_size)
                    dataloader_test  = DataLoader(data_test , batch_size=self.config.batch_size)
                    
                    return dataloader_train, dataloader_valid, dataloader_test
                    
                ############################################################
                
                def _generate_from_data(self, test_data, quantization_mode):
                    self.model.eval()
                    print("バッチサイズを変更するときには、ここの値も変更しなさい！")
                    batch_size = 32 # self.batch_size # 
                    results = []
                    with torch.no_grad():
                        for start in range(0, len(test_data), batch_size):
                            end = start + batch_size
                            data = test_data[start:end].to(self.device)
                            
                            # ここが元データから生成してゆく場所
                            #  - dataをoutputとまぜてゆくと、初期値だけから生成することになる
                            output = self.model(data)
                            
                            # Plot histogram of output unique values before applying threshold
                            plt.figure(figsize=(12, 6))
                            plt.hist(output.cpu().numpy().ravel(), bins=50, color='blue', alpha=0.7)
                            plt.xlabel('Value', fontsize=14)
                            plt.ylabel('Frequency', fontsize=14)
                            plt.title('Histogram of Unique Values (Before Threshold)', fontsize=16)
                            plt.tick_params(axis='both', which='major', labelsize=12)
                            plt.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2)  # Threshold in red dashed line
                            plt.tight_layout()
                          #  plt.savefig(f"{gen_dir}/GenSigHist_BeforeThreshold_batch_{start // batch_size + 1}.pdf")
                          #  plt.close()
                            
                            # # Apply quantization
                            # if quantization_mode == 'threshold':
                            #     output = (output > self.threshold).float()
                            #     print(f"Threshold in generation: {self.threshold}")
                            # # elif quantization_mode == 'sample':
                            # #     output = sample(output)
                            # else:
                            #     raise ValueError(f"Unsupported quantization mode: {quantization_mode}")
                            
                            results.append(output.cpu())  # Store results on CPU to save GPU memory
                    
                    # Concatenate all results from CPU to a single tensor before returning
                    return torch.cat(results, dim=0)
                    
            ###############################################################################
            ###############################################################################
            epoch_num_test = 1 # 2 # len(region_id_train0) # 最初の要素（index 0）を除外する場合
            
            data_name_train10_fortransform0 = list()
            data_name_test10_fortransform0 = list()
            
          #  for i in range(1,epoch_num_test+1):
          #     data_name_train10_fortransform0.append(region_date_train[i])
               
            for i in range(1,epoch_num_test+1):
               data_name_train10_fortransform0.append(region_date_train[i])
               data_name_test10_fortransform0.append(region_date_test[i])
                  
            ## ######################################################################
            seen = set()
            data_name_train10_fortransform = list()
            for item in data_name_train10_fortransform0:
                if item not in seen:
                    seen.add(item)
                    data_name_train10_fortransform.append(item)
                    
            print("data train list:" + str(data_name_train10_fortransform))
            
            ## ######################################################################
            seen = set()
            data_name_test10_fortransform = list()
            for item in data_name_test10_fortransform0:
                if item not in seen:
                    seen.add(item)
                    data_name_test10_fortransform.append(item)
                    
            print("data test list:" + str(data_name_test10_fortransform))
            
            ## ######################################################################
            ## ######################################################################
            # 　spykes.npy フォーマットに変換するデータを作成
            ## ######################################################################
           # data_dir0 = "/Sharing_ArtCorr_128/"
           # data_dir_output = "F:/python_home/200410_Music_Naka/home_dir_run/Sharing_ArtCorr_128/"
            
            file_list2 = "128"# + "_epo" + str(epoch_num) 
            os.chdir( "./data/" )
            
            print("Data Group ID :" + str(group_ID1) )
             
            # ###################################################################################
            # Loop for training data       
            # ###################################################################################
            
            train_or_not = 1
            for run_index2 in range(1,epoch_num_test+1,1): 
            # ###################################################################################
            # Loop for test data
            # ###################################################################################
                for run_index3 in range(1, epoch_num_test+1 ,1): 
                    
                    for run_index in range(1, 3 ,1):
                        
                        os.chdir( "F:/python_home/200410_Music_Naka/home_dir_run/spike_tnb_200428/data/" )
                        data_name_train_bf1 = "128neuron_rep" + str(epoch_num) 
                        data_name_bf2 = data_name_train_bf1 + "/Group"+str(group_ID1)+"_"+ region_id_train0[run_index2] + str(region_date_train[run_index2]) + "_" + region_id_train0[run_index2] + str(region_date_train[run_index2])        
                        
                        print("======================================================================")
                        print("======================================================================")
                        print("run_index2:" + str(run_index2) )
                        data_name_train10 = list()
                        data_name_train10.append(region_date_train[run_index2])
                        data_name_train10.append(region_date_train[run_index2])
                                
                        data_name03 = edit_data_name(region_id_test0[run_index3] + str(region_date_test[run_index3])) + "_depth_min" + str(depth_min_test[run_index3]) + "_max" + str(depth_max_test[run_index3])
                        data_name_main  = data_name_bf2 + "/" + data_name03
                       # data_name  = data_name_bf2 + "/" + region_id_test[data_index_test] + str(region_date_test[data_index_test])
                        os.makedirs("./" + data_name_main, exist_ok=True)
                
                    #   data_name_main  = data_name_bf2 + "/" + region_id_test0[run_index3] + str(region_date_test[run_index3])
                    #    os.makedirs("./" + data_name_main, exist_ok=True)
                        
                        print("run_index3:" + str(run_index3) )
                        data_name_train10.append(region_date_test[run_index3])
                        data_name_train10.append(region_date_test[run_index3])
                        
                        print("======================================================================")
                        print("======================================================================")
                        print("run_index2:" + str(run_index2) )
                        init_cell_index_all = list()
                        init_cell_index_all.append(init_cell_index_train[run_index2])
                        init_cell_index_all.append(init_cell_index_train[run_index2])
                         
                        print("run_index3:" + str(run_index3) )
                        init_cell_index_all.append(init_cell_index_test[run_index3])
                        init_cell_index_all.append(init_cell_index_test[run_index3])
                        
                        
                        print("run_index2:" + str(run_index2) )
                        depth_min_test_all = list()
                        depth_min_test_all.append(depth_min_test[run_index2])
                        depth_min_test_all.append(depth_min_test[run_index2])
                        print("run_index3:" + str(run_index3) )
                        depth_min_test_all.append(depth_min_test[run_index3])
                        depth_min_test_all.append(depth_min_test[run_index3])
                        
                        print("run_index2:" + str(run_index2) )
                        depth_max_test_all = list()
                        depth_max_test_all.append(depth_max_test[run_index2])
                        depth_max_test_all.append(depth_max_test[run_index2])
                        print("run_index3:" + str(run_index3) )
                        depth_max_test_all.append(depth_max_test[run_index3])
                        depth_max_test_all.append(depth_max_test[run_index3])
                        
                        print("======================================================================")
                            
                      #  data_name_train  = data_name_bf2 + "/" + region_id_train0[run_index3] + str(region_date_train[run_index3])
                        print(data_name_main)
                        
                        connect_list = [run_index2, run_index2, run_index3, run_index3]
                        
                        
                        def log10(input):
                            import math
                            return np.log10(input)
                            
                #    data_length = 2_500_000
                #    data_length = 2_000_000
                #    data_length = 1_600_000
                #    data_length = 1_500_000
                #    data_length = 1_400_000
                #   data_length = 1_250_000
                        data_length = 312_500
                                
                        kkm = 1
                        for nnm in connect_list:
                                
                                used_steps_LSTM = round((data_length))
                                sep_size = round((data_length)/len(connect_list))  #  (len(data_name_train10))) 
                                
                            #    data_dir_spykes = "F:\python_home\200410_Music_Naka\home_dir_run\Sharing_whole_SizeSame128e128/" + region_date_train[nnm]+ "001/"
                            #    start_time_step = 3_600_000 +nnm * sep_size 
                                if kkm <= 2:
                                   data_dir_spykes = "../../Sharing_whole_SizeSame128/" + region_date_train[1] + "001/"
                                   start_time_step = 3_600_000 +nnm * sep_size 
                                #   start_time_step = 1800_000 + nnm * sep_size
                                else:
                                   data_dir_spykes = "../../Sharing_whole_SizeSame128/" + region_date_test[1] + "/"
                                   start_time_step = 1_000_000 +nnm * sep_size
                                   
                                data_sep = []
                                init_cell_index = init_cell_index_all[kkm-1]  
                                depth_min_test0 = depth_min_test_all[kkm-1]
                                depth_max_test0 = depth_max_test_all[kkm-1]
                                data_sep, extract_index = div_weig_ryo(data_dir_spykes, len(connect_list), start_time_step  , used_steps_LSTM, sep_size,  init_cell_index , depth_min_test0, depth_max_test0 )
                                
                                # print("||||||||||||||||||||||||||||||")
                                # print("||||||||||||||||||||||||||||||")
                                # print("||||||||||||||||||||||||||||||")
                                # print("||||||||||||||||||||||||||||||")
                                # print(str(kkm) + ':')
                                # print(str(data_sep.shape))
                                # print("||||||||||||||||||||||||||||||")
                                # print("||||||||||||||||||||||||||||||")
                                # print("||||||||||||||||||||||||||||||")
                                
                                if kkm == 1:
                                    data_all = copy.copy(data_sep)
                                else:
                                    data_all = np.append(data_all, data_sep, axis = 0)
                                kkm = kkm + 1
                        
                      #  os.chdir( "../" )
                        print(os.getcwd())
                        os.makedirs("./", exist_ok=True)
                        os.makedirs("./" + data_name_main, exist_ok=True)
                
                        region_date_s[1] = region_date_train[run_index2] # +region_date_train[run_index2][-1]
                        region_date_s[2] = region_date_train[run_index2] # +region_date_train[run_index2][-1]
                        region_date_s[3] = region_date_test[run_index3] # +region_date_train[run_index3][-1]
                        data_name_main_forconfig = "tr" + str(region_date_s[1]) + "_val" + str(region_date_s[2]) + "_te" + str(region_date_s[3])
                        
                        data_name_main_forconfig = edit_data_name(data_name_main_forconfig)
                        print("======================================================================")
                    #    os.chdir("./data")
                        rootdir = os.getcwd() +"/"
                        
                        ## #########################################################
                        
                        data_name_train33  = data_name_bf2 + "/" + region_id_train0[1] + str(region_date_train[1])
                       # data_name_test44  = data_name_bf2 + "/" + region_id_test0[2] + str(region_date_test[2])
                        
                        data_name_test44  = data_name_bf2 + "/" + region_id_test0[1] + edit_data_name(str(region_date_test[1])) + "_depth_min" + str(depth_min_test[run_index3]) + "_max" + str(depth_max_test[run_index3])
                        
                        gen_dir33 = rootdir + str(data_name_train33) +"/" + file_list2 +"/transformer"+"/"
                        gen_dir44 = rootdir + str(data_name_test44) +"/" + file_list2 +"/transformer"+"/"
                 
                        gen_dir3_index1 = Path(gen_dir33)  / "threshold.txt"
                        gen_dir3_now2   = Path(gen_dir44)  / "threshold.txt"
                        if gen_dir3_index1.exists(): # run_index3 >= 2 and 
                            gen_dir3_now2.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(gen_dir3_index1, gen_dir3_now2)
                            
                        gen_dir3_index10 = Path(gen_dir33) / "spike_best.pt"
                        gen_dir3_now20   = Path(gen_dir44) / "spike_best.pt"
                        if gen_dir3_index10.exists(): # run_index3 >= 2 and 
                            gen_dir3_now20.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(gen_dir3_index10, gen_dir3_now20)
                            
                        ## #########################################################
                        gen_dir2     = rootdir + str(data_name_main)+"/"
                        gen_dir1     = rootdir + str(data_name_main) +"/" + file_list2+"/"
                        gen_dir      = rootdir + str(data_name_main) +"/" + file_list2 +"/transformer"+"/"
                        os.makedirs(gen_dir, exist_ok=True)
                        
                        for directory in [gen_dir2, gen_dir1, gen_dir]:
                            if os.path.exists(directory):
                                print("| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |")
                            else:
                                os.makedirs(directory, exist_ok=True)
                              
                        file_path = str(gen_dir) + "/extract_index.txt" #os.path.join(f"{gen_dir}/extract_index.txt")
                        with open(file_path, 'w') as f:
                              f.write(f"{extract_index}")
          
                     #   print("===========================================================")
                        os.chdir(rootdir)
                        data_all = np.nan_to_num(data_all, nan=0.0)
                        np.save( gen_dir2 + "/spikes.npy" , data_all )
                        spikes = np.load(gen_dir2+'/spikes.npy') # , spikes)
                        
                        # split_spikes = np.array([
                        #      spikes[i*data_length:(i+1)*data_length]
                        #      for i in range(0, spikes.shape[0]//data_length)])
                         
                        # np.save(gen_dir2+'/split_spikes.npy', split_spikes)
                        nNeuron = data_all.shape[1]
                        
               #         del data_all 
               #         del spikes
                        
                        
                        print("======================================================================")
                        shutil.copy("./config.ini" , "./config_" + str(data_name_main_forconfig) +".ini" )
                        
                        with open("./config_" + str(data_name_main_forconfig) + ".ini", "r+", encoding='utf-8') as config_file:
                            data_lines = config_file.read()
                            data_lines = data_lines.replace("dim_input = 101", "dim_input = "+ str(nNeuron))
                            data_lines = data_lines.replace("n_epoch = 10", "n_epoch = " + str(epoch_num ))
                            data_lines = data_lines.replace("/n/work1/gshibata/src/spike_tnb_200424","./" )
#                            data_lines = data_lines.replace("%(base)s/data/split_spikes.npy" , gen_dir2 + "/split_spikes.npy")
                            data_lines = data_lines.replace("%(base)s/data/split_spikes.npy" , gen_dir2 + "/spikes.npy")
                            data_lines = data_lines.replace("%(base)s/data/result" ,  gen_dir2 + "/128") # + "_epo" + str(epoch_num))
                            data_lines = data_lines.replace("n_layer_list = 1,2,3", "n_layer_list = 1") # ,1,1,1
                            data_lines = data_lines.replace("n_hidden_list = 128,64,256", "n_hidden_list = 128") # ,128,128,128
                            data_lines = data_lines.replace("n_hidden_list = 128,64,128", "n_hidden_list = 128") # ,128,128,128
                        #    data_lines = data_lines.replace("train_rate = 0.5", "train_rate = 0.5")
                        #    data_lines = data_lines.replace("valid_rate = 0.2", "valid_rate = 0.25")
                            data_lines = data_lines.replace("learning_rate = 0.00003", "learning_rate = 0.005")
                         #   data_lines = data_lines.replace("train_rate = 0.5", "train_rate = 0.75")
                            data_lines = data_lines.replace("train_rate = 0.5", "train_rate = 0.5")
                            data_lines = data_lines.replace("valid_rate = 0.2", "valid_rate = 0.25")
                           # data_lines = data_lines.replace("gamma = 0.0", "gamma= 4.0") 
                          #  data_lines = data_lines.replace("n_layer_list = 1,2,3", "n_layer_list = 1,1,1")
                          #  data_lines = data_lines.replace("n_hidden_list = 128,64,128", "n_hidden_list = 128,128,128")
                          
                            config_file.seek(0)
                            config_file.write(data_lines)
                            config_file.truncate()
                            
                        os.chdir(rootdir)
                        
                        CONFIG_FILENAME    =  "config_" + str(data_name_main_forconfig) + ".ini"
                        CONFIG_PATH        = os.path.join(rootdir, CONFIG_FILENAME)
                        LOGCONFIG_FILENAME = "log_config.ini"
                        
                        gen_dir2_index1 = gen_dir2 + "/128" # + "_epo" + str(epoch_num)
                        
                        if run_index == 1:
                           train_or_not = train_or_not+1
                        
                        check_file = gen_dir33 + "/raster_plot_inputs_targets.pdf"
                        
                        print("group_ID1: "+str(group_ID1))
                        print("group_ID2: "+str(group_ID2))
                        
                        if run_index == 1and not os.path.isfile(check_file): #  and group_ID1 != group_ID2 
                     #   if run_index == 1 and train_or_not == 1:# np.mod(train_or_not,2) == 0: # run_index2 ~= run_index2_prev: #  and run_index3 == 1:    # っテストデータが一つ目で、なおかつ　トレーニング期間　のときに行う
                     #   if run_index == 1:# & run_index3 == 1:   # トレーニング期間
                             #   print("run_index2_prev: " + str(run_index2_prev) + " , run_index2: " + str(run_index2))
                                print("run_index3: " + str(run_index3))
                                print("  --- --- ------> トレーニングします！")
                                main_mod_func("transformer","train","fromdata","threshold", CONFIG_FILENAME )
                                os.chdir(gen_dir)
                                os.chdir( rootdir )
                                torch.cuda.empty_cache()
                                loss_train_pdf_path = os.path.join(gen_dir2, "loss_train.pdf")
                                if not os.path.exists(loss_train_pdf_path):
                                    shutil.copy(gen_dir+"/loss.pdf",     gen_dir2+"/loss_train.pdf")
                                    shutil.copy(gen_dir+"/accuracy.pdf", gen_dir2+"/accuracy_train.pdf")
                             #   run_index2_prev = run_index2
                             
                        elif run_index == 2:#  and group_ID1 != group_ID2:
                            
                            print("======================================================================")
                            os.chdir(rootdir)
                            print("Now, copying config file")
                            
                            print("Now, editing the config file for validation phase")
                            with open("./config_" + str(data_name_main_forconfig) + ".ini", "r+", encoding='utf-8') as config_file:
                                data_lines = config_file.read()
                             #   data_lines = data_lines.replace(gen_dir2 + "/split_spikes.npy", gen_dir2 + "/split_spikes_sorted.npy")
                                data_lines = data_lines.replace(gen_dir2 + "/spikes.npy", gen_dir2 + "/spikes_sorted.npy")
                        
                                config_file.seek(0)
                                config_file.write(data_lines)
                                config_file.truncate()
                                
                            os.chdir(gen_dir)
                            if os.path.exists("./spike.pt"):
                                print("Now, erasing spike.pt")
                                os.remove("spike.pt")
                            
                            print("===========================================================")
                            initial_loss = 100.0
                            initial_temp = 100.0
                            cooling_rate = 0.9999
                          #  cooling_rate = 0.999995
                            spikes = np.load(gen_dir2 + '/spikes.npy')
                            np.save(gen_dir2+'/spikes_sorted.npy', spikes)
                           # split_spikes = np.load(gen_dir2 + '/split_spikes.npy')
                          #  np.save(gen_dir2+'/split_spikes_sorted.npy', split_spikes)
                            gen_dir2_now = Path(gen_dir2) / "128"
                            random_order_path = gen_dir2_now / "transformer" / "random_order.npy"
                          #  starting_order = np.load(random_order_path)
                            try:
                                starting_order = np.load(random_order_path)
                            except FileNotFoundError:
                                print(f"File {random_order_path} not found. Generating a random order.")
                                num_neurons = 128  # or whatever the number of items is
                                starting_order = np.random.permutation(num_neurons)
                                np.save(random_order_path, starting_order)  # Optionally save the newly generated order
                            
                            
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print(str(spikes.shape[1]))
                                  
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            
                            ideal_order = np.arange(spikes.shape[1])
                           # ideal_order = np.arange(split_spikes.shape[2])
                            current_loss = initial_loss
                            temperature = initial_temp
                            
                            current_order = starting_order.copy()
        
                           # print(" Swapping  split_spikes.npy  now. ")
                            print(" Swapping  spikes.npy  now. ")
                            
                          #  criterion = BinaryFocalLoss(gamma= 5.0, alpha=0.5, reduction='mean')
                          #  criterion = DiceLoss(smooth=1.0) # DiceLoss(smooth=0.1) # (smooth=1.0)
                            criterion = DiceLoss(smooth=1.0, penalty_weight=0.1)
                            # WeightedBinaryFocalLoss(gamma=10.0, alpha=0.5, weight=None, reduction='mean')
                                
                            if not os.path.exists(gen_dir2_now):
                                shutil.copytree(gen_dir2_index1, gen_dir2_now)
                             #   shutil.copytree(gen_dir2_index2, gen_dir2_now)
                            else:
                                print("Directory already exists")
                            
                            gen_dir2_index1 = "./config_" + str(data_name_main_forconfig) +".ini"
                            gen_dir2_now2 = gen_dir + "/log_config.ini"
                            os.chdir( "F:/python_home/200410_Music_Naka/home_dir_run/spike_tnb_200428/data/" )
                            shutil.copy(gen_dir2_index1, gen_dir2_now2)
                            
                            os.chdir(gen_dir)
                           # main_mod_func("transformer","generate","fromdata","threshold", CONFIG_FILENAME )
                            main_mod_func("transformer","generate","fromdata","threshold", CONFIG_FILENAME )
                            os.chdir(gen_dir2 + "/128/transformer")
                            
                            os.chdir(rootdir)
                            os.chdir( gen_dir )
                            inputs_it  = torch.tensor(np.load('fromdata_threshold_generated_test.npy'))
                            targets_it = torch.tensor(np.load('fromdata_threshold_groundtruth_test.npy'))
                            
                            ##########################################################################
                            ##########################################################################
                            iteration_num = 500000# 1500000 # 3# 50000 # 128*128/2 # 
                            
                            loss_all = [0] * iteration_num
                            slip_num = [0] * iteration_num
                            
                            current_order_all = np.empty((0, len(current_order)), dtype=int)
                            
                            current_loss_all = []
                            current_gap_loss_all = []
                            
                            # 初期順序との違いと、一つ前の順序との違いを記録するリスト
                            differences_from_ideal = []
                            differences_from_previous = []
                            differences_from_starting = []
                            
                            match_rates_from_ideal = []
                            match_rates_from_previous = []
                            match_rates_from_starting = []
                            
                            previous_order = current_order.copy()
                            
                            def calculate_data_density(target_tensor):
                                total_spikes = torch.sum(target_tensor)
                                total_elements = target_tensor.numel()
                                return total_spikes.item() / total_elements
        
                            gap_loss = 5e-7 # 7
                         #   gap_loss = 2e-6 # 7  # 6e-8 #  #https://www.amazon.co.jp/%E5%B0%8F%E5%AD%A64%E5%B9%B4%E7%94%9F-%E3%81%93%E3%81%9D%E3%81%82%E3%81%A9%E8%A8%80%E8%91%89%E3%83%BB%E6%96%87%E3%82%92%E3%81%A4%E3%81%AA%E3%81%90%E8%A8%80%E8%91%89-%E3%80%8C%E3%81%93%E3%82%8C%E3%83%BB%E3%81%9D%E3%82%8C%E3%83%BB%E3%81%82%E3%82%8C%E3%83%BB%E3%81%A9%E3%82%8C%E3%80%8D%E3%82%84%E3%80%8C%E3%81%A0%E3%81%8B%E3%82%89%E3%83%BB%E3%81%97%E3%81%8B%E3%81%97%E3%83%BB%E3%81%BE%E3%81%9F%E3%80%8D%E3%81%AA%E3%81%A9%E3%81%AE%E6%AD%A3%E3%81%97%E3%81%84%E4%BD%BF%E3%81%84%E6%96%B9-%E3%81%8F%E3%82%82%E3%82%93%E3%81%AE%E3%81%AB%E3%81%8C%E3%81%A6%E3%81%9F%E3%81%84%E3%81%98%E3%83%89%E3%83%AA%E3%83%AB-%E5%9B%BD%E8%AA%9E/dp/4774321605/ref=sr_1_6?crid=2CNDJVZLY0E3V&dib=eyJ2IjoiMSJ9.3vWUmNhQcQ63AwYkMSMR5H-Tfyd94KHCAklwFdNrJcGufFbzO62ONNrlPMDUOPu5dC9XIcCAK45eqawKrAKt_BVD_ZrwKgAFxNPLYXkfIN3tU3-D0PtQ11gxu4KzViEcd0DZkYZe4KwLd90dQTKR9BoQfam-C8eBM5FZ7gKshIl8MoAnBEIaQxaqDVNFsvZOYD6CLaueOQi5jHODJe32tQ.Quk-0-RgSQsc6h1w8F_86k7_3RiUGHG7Ch4P7EofTLs&dib_tag=se&keywords=%E6%8E%A5%E7%B6%9A%E8%A9%9E+%E5%B0%8F%E5%AD%A6%E7%94%9F&qid=1730277204&sprefix=%E6%8E%A5%E7%B6%9A%2Caps%2C236&sr=8-6
                                   
                            file_path = os.path.join(gen_dir, 'threshold.txt')
                            with open(file_path, 'r') as f:
                                threshold_value = f.read()
                        
                            checkpoint_path = gen_dir + "spike_best.pth"
                            from types import SimpleNamespace
                            
                            print("Configファイルを書き替えたら、ここも編集しないとだめ！")
                            config0 = SimpleNamespace(
                                        dir_result=gen_dir,
                                        dim_input=128,
                                      #  n_layer_list=[1, 1, 1],
                                      #  n_hidden_list=[128, 128, 128],
                                        n_layer_list=[1],
                                        n_hidden_list=[128],
                                        n_heads=32,
                                        dropout=0.1,
                                        cuda=(0,),
                                        learning_rate = 0.01,
                                        threshold=threshold_value,
                                       # dir_spike=gen_dir2+'split_spikes_sorted.npy',
                                        dir_spike=gen_dir2+'spikes_sorted.npy',
                                        train_rate = 0.5, 
                                        valid_rate = 0.25,
                                        split_size = 15625,
                                        batch_size = 32,
                                        checkpoint_path=checkpoint_path  # チェックポイントパスを追加
                                    )
                                    
                            runner = TransformerStackRunner(mode='train', experiment_name='example', config=config0)
                            
                            satisfied_count = 0
                            
                            current_density = calculate_data_density(inputs_it.float())
                            criterion = DiceLoss(smooth=1.0, penalty_weight=0.1)
                            
                            for iteration in range(1, iteration_num, 1):  
                                print("Rep: " + str(iteration) )
                                
                                new_order = simulated_annealing_process(current_order, temperature, cooling_rate, iteration_num)
                                assert max(new_order) < targets_it.shape[1], "Index out of bounds in targets_it."
                                print("new order"+str(new_order[:30]))
                                target_density = calculate_data_density(targets_it[:, new_order].float())
        
                               # new_loss = criterion(inputs_it.float(), targets_it[:, new_order].float(), target_density)
                                new_loss = criterion(inputs_it.float(), targets_it[:, new_order].float(), current_density, target_density,loss_show=0)
                                
                                # 損失の改善または温度に基づく確率で更新を採用
                                if new_loss < current_loss - gap_loss: # (1e-7)  current loss = 0.000501 なら変わる or random.random() < np.exp((current_loss - new_loss) / temperature):
                                    satisfied_count = 0
                                    # 連続満足条件のカウントを1増加
                                    
                                    gap_loss = gap_loss * 0.996 # 0.94 # 94
                                #    gap_loss = gap_loss * 0.997 # 0.94 # 94
                             #       gap_loss = gap_loss * 0.998 # 0.94 # 94
                                    
                                    current_loss_all.append(current_loss)
                                    current_gap_loss_all.append(current_loss-new_loss)
                                    
                                    current_order = new_order.copy()
                                    current_loss = new_loss
                                    current_order_all = np.vstack([current_order_all, current_order])
                                    
                                    import math
                                    # 前回の順序との二乗誤差を計算
                                    diff_starting_squared = sum((i - j) ** 2 for i, j in zip(current_order, starting_order))/128
                                    differences_from_starting.append( math.sqrt(diff_starting_squared) )
                                    
                                    diff_ideal_squared =  sum((i - j) ** 2 for i, j in zip(current_order, ideal_order))/128
                                    differences_from_ideal.append( math.sqrt(diff_ideal_squared) )
                                    
                                    diff_previous_squared = sum((i - j) ** 2 for i, j in zip(current_order, previous_order))/128
                                    differences_from_previous.append( math.sqrt(diff_previous_squared) )
                                    
                                    # 前回の順序との一致率を計算
                                    match_rate_starting = sum(1 for i, j in zip(current_order, starting_order) if i == j) / len(current_order)
                                    match_rates_from_starting.append(match_rate_starting)
                                    
                                    # 理想の順序との一致率を計算
                                    match_rate_ideal = sum(1 for i, j in zip(current_order, ideal_order) if i == j) / len(current_order)
                                    match_rates_from_ideal.append(match_rate_ideal)
                                    
                                    # 前の順序との一致率を計算
                                    match_rate_previous = sum(1 for i, j in zip(current_order, previous_order) if i == j) / len(current_order)
                                    match_rates_from_previous.append(match_rate_previous)
                                    
                                    # 今回の順序を前回の順序として記録
                                    previous_order = current_order.copy()
                                    
                                    current_order_all = np.vstack([current_order_all, current_order])
                                    
                                else:
                                   # 条件が満たされなかった場合、カウントをリセット
                                   satisfied_count += 1
                                   print("satisfied_count: " + str(satisfied_count))
                                   
                                   if satisfied_count >= 7500:
                                        print("条件が75000回連続で満たされたためループを終了します。")
                                        break  # ループを終了 
                    
                                loss_all[iteration - 1] = current_loss
                                
                                # 温度を更新
                                temperature *= cooling_rate
                                
                                print("Calculated loss: " +str(current_loss))
                                print("Temperature: " +str(temperature))
                                print("Gap loss: " +str(gap_loss))
                                print("Exchanged # from the beginning:" + str(len(differences_from_ideal)))
                             
                                # 終了条件の確認
                                if temperature < 1e-10:
                                    break
                                
                            os.chdir( gen_dir )
                            npy2npr_mod3_func("fromdata_threshold_generated_test.npy","gener.txt")
                            npy2npr_mod3_func("fromdata_threshold_groundtruth_test.npy","truth.txt")
                            
                            np.save(gen_dir+'/final_gen_loss.npy', current_loss)
                            np.save(gen_dir+'/final_order.npy', current_order)
                            
                            
                            with open('final_order_with_only_loss.txt', 'w') as f:
                                f.write(" ".join(map(str, current_order)))
                                    
                    ## ################################################################
                  #  print("loss_all:" + str(loss_all))
                    
                    # プロットの作成
                    plt.figure(figsize=(10, 5))
                    
                    # 非ゼロのインデックス範囲を取得
                    non_zero_indices = [i for i, val in enumerate(loss_all[1:-1]) if val != 0]
                    if non_zero_indices:
                        x_min, x_max = non_zero_indices[0] + 1, non_zero_indices[-1] + 1  # +1でインデックスの調整
                    
                        # プロットの描画
                       # plt.plot(range(1, len(loss_all)-1), loss_all[1:-1], marker='o', linestyle='-', color='blue')
                        plt.plot(range(1, x_max-1), loss_all[1:x_max-1], marker='o', linestyle='-', color='blue')
                        plt.xlabel('Iteration steps')
                        plt.ylabel('Loss')
                        plt.title('Iteration vs Loss ' + str(region_date_train[run_index2]))
                        plt.grid(True)
                    
                        # 非ゼロ範囲のみにx軸を制限
                        plt.xlim(x_min, x_max-1)
                    else:
                        print("Warning: 全てのデータがゼロです")
                    
                    # PDFで保存
                    plt.savefig('iteration_vs_loss.pdf')
                    plt.show()
                    
                    ## ################################################################
                    # プロットの作成
                    print("differences_from_ideal:" + str(differences_from_ideal))
                    print("differences_from_starting:" + str(differences_from_starting))
                    print("differences_from_previous:" + str(differences_from_previous))
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(differences_from_ideal,    label='Differences from the correct order ')
                    plt.plot(differences_from_starting, label='Difference from the swapped order')
                    plt.plot(differences_from_previous, label='Differences from the previous order')
                    plt.title('Differences from several orders over iterations')
                    plt.xlabel('Iteration steps')
                    plt.ylabel('Number of differences')
                    plt.legend()
                    plt.grid(True)
                    # PDFで保存
                    plt.savefig(gen_dir + '/Iteration_OrderChange.pdf')
                    
                   # plt.show()
                    plt.close()
                    ## ################################################################
                    plt.figure(figsize=(10, 5))
                    plt.plot(current_loss_all,    label='Current loss')
                    plt.plot(current_gap_loss_all, label='Loss change')
                    plt.title('Loss changes over iterations')
                    plt.xlabel('Iteration steps')
                    plt.ylabel('Loss or loss change')
                    plt.legend()
                    plt.grid(True)
                    # PDFで保存
                    plt.savefig(gen_dir + '/Iteration_LossChange.pdf')
                    
                    plt.close()
                    ## ################################################################
                    
                    # プロットの作成
                    print("match_rates_from_ideal:" + str(match_rates_from_ideal))
                    print("match_rates_from_starting:" + str(match_rates_from_starting))
                    print("match_rates_from_previous:" + str(match_rates_from_previous))
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(match_rates_from_ideal,    label='With the correct target order')
                    plt.plot(match_rates_from_starting, label='With the initial swapped order')
                    plt.plot(match_rates_from_previous, label='With the 1-step previous order')
                    plt.title('Matching ratios with several orders over iterations')
                    plt.xlabel('Iteration steps')
                    plt.ylabel('Matching ratio')
                    plt.legend()
                    plt.grid(True)
                    # PDFで保存
                    plt.savefig(gen_dir + '/Iteration_OrderMatchingRatio.pdf')
                    
                  #  plt.show()
                    plt.close()
                    
                    ## ################################################################
                    # Create a figure with two subplots for side-by-side comparison
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns for side-by-side plots
                    
                    # Plot the input raster on the first subplot
                    axes[0].imshow(inputs_it[:6000, :].T, cmap='binary', aspect='auto', vmin=0, vmax=1.5)
                    axes[0].set_title(f'Generated data: from {region_id_train0[run_index2]} {region_date_train[run_index2]}')
                    axes[0].set_xlabel('Time')
                    axes[0].set_ylabel('Neurons/Features')
                    
                    # Plot the target raster on the second subplot (using `current_order`)
                    axes[1].imshow(targets_it[:6000, current_order].T, cmap='binary', aspect='auto', vmin=0, vmax=1.5)
                    axes[1].set_title(f'Ground truth data: (current_order): {region_id_train0[run_index3]} {region_date_train[run_index3]}')
                    axes[1].set_xlabel('Time')
                    axes[1].set_ylabel('Neurons/Features')
                   
                    # Adjust layout and save the figure as a PDF
                    plt.tight_layout()
                    output_filename = f'raster_plot_inputs_targets.pdf'
                    plt.savefig(gen_dir + '/' + output_filename)
                    plt.close()  # Close the plot to free memory
                    
                    ## ################################################################
                    # カラーマップを表示
                    num_elements = 128
                    if len(current_order_all) != 0:
                        current_order_all = current_order_all.reshape((-1, num_elements))
                    
                    # 配列を転置
                    current_order_all_transposed = current_order_all.T
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(current_order_all_transposed, aspect='auto', cmap='viridis')
                    plt.colorbar(label='Order Index')
                    
                    # x軸とy軸のラベルを入れ替え
                    plt.xlabel('Iteration steps')
                    plt.ylabel('Element index')
                    plt.title('Order change by iterations')
                    
                    # PDFで保存
                    plt.savefig(gen_dir + '/Iteration_OrderChange_Colormap.pdf')
                    
                  #  plt.show()
                    plt.close()
                    
                    ## ################################################################
                    loss_train_all    = np.zeros((epoch_num-1))
                    accu_train_all    = np.zeros((epoch_num-1))
                    accu_train_spike    = np.zeros((epoch_num-1))
                    accu_train_nonspike = np.zeros((epoch_num-1))
                    loss_valid_all    = np.zeros((epoch_num-1))
                    accu_valid_all    = np.zeros((epoch_num-1))
                    accu_valid_spike    = np.zeros((epoch_num-1))
                    accu_valid_nonspike = np.zeros((epoch_num-1))
                    
                 #   gen_dir2_now = gen_dir2 + "/128"+ "_epo" + str(epoch_num)
                    
                    loss_train_all, loss_valid_all, accu_valid_all, accu_valid_nonspike = test_func(gen_dir2_now)
                    
                    with open('loss.txt', 'w') as f:
                        f.write(str(loss_train_all))
                        f.write(str(loss_valid_all))
                    
                    os.chdir(gen_dir)
                    
                    file_paths = [
                        os.path.join(gen_dir2_now, "transformer", "loss.txt"),
                        os.path.join(gen_dir2_now, "transformer", "gener.txt"),
                        os.path.join(gen_dir2_now, "transformer", "spike.npy"),
                        os.path.join(gen_dir2_now, "transformer", "spike_best.npy"),
                    ]
                    
                    for file_path in file_paths:
                        try:
                            os.remove(file_path)
                        except FileNotFoundError:
                            pass
                        
        ############################################################################
        ############################################################################
        ############################################################################
        ############################################################################
        
                    # Get initial shapes
                    num_elements = targets_it.shape[1]
                    
                    # Randomize the initial order of targets_it
                    starting_order = np.arange(num_elements)
                    np.random.shuffle(starting_order)
                    
                    # Apply random order to targets_it
                    targets_it = targets_it[:, starting_order]
                    
                    # Initial orders
                    current_order0 = starting_order.copy()
                    ideal_order = np.arange(num_elements)
                    previous_order0 = starting_order.copy()
                    
                    # Similarity calculation function for sparse binary data
                    def calculate_similarity(inputs, targets):
                        return np.sum((inputs == 1) & (targets == 1), axis=0).astype(float) / \
                               np.sum((inputs == 1) | (targets == 1), axis=0).astype(float)
                    
        ############################################################################
        ############################################################################
        ############################################################################
                    
                    os.chdir(gen_dir2)
                    import time
                    
                    os.chdir(rootdir)
                    torch.cuda.empty_cache()    
                    


