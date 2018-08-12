import sys
import numpy as np

def extract(filename):
    all_train, all_valid, all_test = [], [], []
    all_train_loss, all_valid_loss, all_test_loss = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            if "TRAIN_LOSS" in line:
                l = line.split(",")
                all_train_loss.append(float(l[1].replace("TRAIN_LOSS: ", "")))
                all_valid_loss.append(float(l[2].replace("VALID_LOSS: ", "")))
                all_test_loss.append(float(l[3].replace("TEST_LOSS: ", "")))
                continue
            if "BLEU: " not in line:
                continue
            bleu = float(line.strip().split("BLEU: ")[1])
            if "TRAIN_BLEU" in line:
                all_train.append(bleu)
            elif "VALID_BLEU" in line:
                all_valid.append(bleu)
            elif "TEST_BLEU" in line:
                all_test.append(bleu)

    return all_train, all_valid, all_test, all_train_loss, all_valid_loss, all_test_loss

def truncate_arrays(n,*args):
    return [a[:n] for a in args]

def get_mean_runs(*args):
    return [[np.mean(x) for x in a ] for a in args]

def get_std_runs(*args):
    return [[np.std(x) for x in a ] for a in args]

def get_runs(tvt, window):
    all_train, all_valid, all_test = [], [], []
    for i in range(len(tvt[1]) + 1 - window):
        all_train.append(tvt[0][i:i+window])
        all_valid.append(tvt[1][i:i+window])
        all_test.append(tvt[2][i:i+window])

    return all_train, all_valid, all_test

if __name__=="__main__":
    truncate = 100
    window_runs = 7

    tvt = extract(sys.argv[1])
    tvt = truncate_arrays(truncate, *tvt)
    i_min = np.argmin(tvt[4])
    j_max = np.argmax(tvt[1])

    #run_tvt = get_runs(tvt, window_runs)
    #mean_runs = get_mean_runs(*run_tvt)
    #std_runs = get_std_runs(*run_tvt)
    #r_max = np.argmax(mean_runs[1])
    print()
    print("BEST LOSS: {} {}\n    & $ {:.5f} $ & $ {:.5f} $ & \\\\".format(i_min,
        tvt[3][i_min],
        tvt[4][i_min],
        tvt[5][i_min]))
    print("BEST LOSS - BLEU {} {}\n    & $ {:.5f} $ & $ {:.5f} $ & \\\\".format(i_min,
        tvt[0][i_min],
        tvt[1][i_min],
        tvt[2][i_min]))
    print("BEST BLEU: {} {}\n    & $ {:.5f} $ & $ {:.5f} $ & \\\\".format(j_max,
        tvt[3][j_max],
        tvt[4][j_max],
        tvt[5][j_max]))
    print("BEST BLEU - BLEU {} {}\n    & $ {:.5f} $ & $ {:.5f} $ & \\\\".format(j_max,
        tvt[0][j_max],
        tvt[1][j_max],
        tvt[2][j_max]))
#    print("RUNS {} {}\pm{}\n    & $ {:.5f} \pm {:.5f} $ & $ {:.5f} \pm {:.5f} $ & \\\\".format(r_max,
#        mean_runs[0][r_max], std_runs[0][r_max],
#        mean_runs[1][r_max], std_runs[1][r_max],
#        mean_runs[2][r_max], std_runs[2][r_max]))
