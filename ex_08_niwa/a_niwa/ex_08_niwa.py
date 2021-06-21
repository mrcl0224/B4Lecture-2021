import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

def forward(outputs, PI, A, B):

    size, t = outputs.shape
    models = np.zeros(size)

    for i in range(size):

        alpha = PI[:, :, 0] * B[:, :, outputs[i, 0]]

        for j in range(1, t):

            alpha = B[:, :, outputs[i, j]] * np.sum(A.T*alpha.T, axis = 1).T

        models[i] = np.argmax(np.sum(alpha, axis=1))

    return models

def viterbi(outputs, PI, A, B):

    size, t = outputs.shape
    models = np.zeros(size)

    for i in range(size):

        alpha = PI[:, :, 0] * B[:, :, outputs[i, 0]]

        for j in range(1, t):

            alpha = B[:, :, outputs[i, j]] * np.max(A.T*alpha.T, axis = 1).T

        models[i] = np.argmax(np.max(alpha, axis=1))

    return models

def main():

    args = sys.argv

    data = pickle.load(open(args[1], "rb"))
    #print(data)

    answer = np.array(data['answer_models'])
    outputs = np.array(data['output'])
    PI = np.array(data['models']['PI'])
    A = np.array(data['models']['A'])
    B = np.array(data['models']['B'])

    #print("answer : \n", answer)

    point1 = time.perf_counter()

    models_forward = forward(outputs, PI, A, B)

    point2 = time.perf_counter()

    models_viterbi = viterbi(outputs, PI, A, B)

    point3 = time.perf_counter()

    print("#{}".format(args[1]))
    print("Forward Algorithm : {}".format(point2 - point1))
    print("Viterbi Algorithm : {}".format(point3 - point2))

    #print(models_forward)
    #print(np.count_nonzero(models_forward == answer)/models_forward.shape[0])

    acc_forward = np.count_nonzero(models_forward == answer)/models_forward.shape[0] * 100
    conf_mat_forward = np.zeros([A.shape[0], A.shape[0]])

    acc_viterbi = np.count_nonzero(models_viterbi == answer)/models_viterbi.shape[0] * 100
    conf_mat_viterbi = np.zeros([A.shape[0], A.shape[0]])

    for i in range(answer.shape[0]):

        conf_mat_forward[int(models_forward[i]), answer[i]] += 1
        conf_mat_viterbi[int(models_viterbi[i]), answer[i]] += 1

    img_spec = plt.figure(figsize=(5, 5))

    sns.heatmap(conf_mat_forward, annot=True, cbar=False, cmap="Blues")
    plt.title("Forward Algorithm\n (Acc. {}%)".format(acc_forward))
    plt.ylabel("Actual Model")
    plt.xlabel("Predict Model")
    plt.show()

    img_spec = plt.figure(figsize=(5, 5))

    sns.heatmap(conf_mat_viterbi, annot=True, cbar=False, cmap="Blues")
    plt.title("Viterbi Algorithm\n (Acc. {}%)".format(acc_viterbi))
    plt.ylabel("Actual Model")
    plt.xlabel("Predict Model")
    plt.show()

if __name__ == "__main__":

    main()
