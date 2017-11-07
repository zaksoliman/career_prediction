import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def print_dists(title_to_id, seq_length, input_seq, prediction, targets, rand=True, f_name="dist"):

    sample_idx = random.sample(range(len(prediction)), 5)
    sample_predictions = prediction[sample_idx]
    sample_targets = targets[sample_idx]
    sample_input = input_seq[sample_idx]
    y = np.argmax(sample_targets, axis=2)
    sample_top_pred = np.argsort(sample_predictions)[:, :, :3]
    with open(f_name + ".txt", "w") as f:
        i = 0
        for sorted_idx, dists in zip(sample_top_pred, sample_predictions):
            f.write("Input:\n")
            f.write(str([title_to_id.inv[ts] for ts in sample_input[i][:seq_length[i]]]) + "\n")
            f.write("Targets\n")
            f.write(str([title_to_id.inv[ts] for ts in y[i][:seq_length[i]]]) + "\n")
            f.write("Prediction:\n")
            pprint(str([[(title_to_id.inv[t_idx], dists[t, t_idx]) for t_idx in time_step] for t, time_step in
                        enumerate(sorted_idx)]) + "\n\n", f)

            i += 0

def print_output(title_to_id, seq_length, input_seq, prediction, targets, rand=True, f_name="out"):
    preds = np.argmax(prediction, axis=2)
    y = np.argmax(targets, axis=2)
    with open(f_name + ".txt", "w") as f:
        for i, ex in enumerate(preds):
            f.write("Input:\n")
            f.write(str([title_to_id.inv[ts] for ts in input_seq[i][:seq_length[i]]]) + "\n")
            f.write("Prediction:\n")
            f.write(str([title_to_id.inv[ts] for ts in ex[:seq_length[i]]]) + "\n")
            f.write("Targets\n")
            f.write(str([title_to_id.inv[ts] for ts in y[i][:seq_length[i]]]) + "\n\n")
