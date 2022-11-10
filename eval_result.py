import argparse
import os

import numpy as np
from responses import target
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

import eval_metrics
import utils


def init_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-pred", "--predicted_labels", required=True)
    parser.add_argument("-gold", "--target_labels", required=True)
    parser.add_argument(
        "--grammar_term_set", default="data/grammar_terms/grammar_terms.small.set"
    )
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("-exc", "--exclude_grammar_terms_by_count", type=int, default=0)
    parser.add_argument("-exc_p", "--exclude_preposition_tag", action="store_true")
    parser.add_argument("-out", "--output_file", default="tmp.csv")
    args = parser.parse_args()
    return args


def evals(target_binaries_list, predicted_binaries_list, label_names):
    exact_match_ratio = eval_metrics.emr(target_binaries_list, predicted_binaries_list)
    exact_match_ratio_for_each_label = eval_metrics.emr_each_label(target_binaries_list, predicted_binaries_list)
    print(f"Exact Match Ratio: {exact_match_ratio}")
    hamming_loss = eval_metrics.hamming_loss(
        target_binaries_list, predicted_binaries_list
    )
    print(f"Hamming Loss: {hamming_loss}")
    precision_of_each_class = eval_metrics.precision_of_each_class(
        target_binaries_list, predicted_binaries_list
    )
    recall_of_each_class = eval_metrics.recall_of_each_class(
        target_binaries_list, predicted_binaries_list
    )
    micro_avg_precision = eval_metrics.label_based_micro_precision(
        target_binaries_list, predicted_binaries_list
    )
    micro_avg_recall = eval_metrics.label_based_micro_recall(
        target_binaries_list, predicted_binaries_list
    )
    f_measure = f1_score(y_true=target_binaries_list, y_pred=predicted_binaries_list, average='weighted')
    print(f"Micro Average Precision: {micro_avg_precision}")
    print(f"Micro Average Recall: {micro_avg_recall}")
    print(f"Precision of each class: {precision_of_each_class}")
    print(f"Recall of each class: {recall_of_each_class}")
    print(f"F measure: {f_measure}")

    # print(
    #     classification_report(
    #         target_binaries_list,
    #         predicted_binaries_list,
    #         output_dict=False,
    #         target_names=label_names,
    #     )
    # )

    print(
        "\t".join(
            map(
                str,
                [
                    exact_match_ratio,
                    hamming_loss,
                    micro_avg_precision,
                    micro_avg_recall,
                    f_measure,
                ],
            )
        )
    )

    return (
        exact_match_ratio,
        hamming_loss,
        micro_avg_precision,
        micro_avg_recall,
        f_measure,
        precision_of_each_class,
        recall_of_each_class,
        exact_match_ratio_for_each_label
    )


def read_file(path):
    data = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            labels = line.rstrip("\n").split("\t")
            data.append(labels)
    return data


def to_binary_list(labels_list, label_num):
    data = []
    for labels in labels_list:
        binary_list = [0] * label_num
        for idx in labels:
            if idx == "":
                continue
            binary_list[idx] = 1
        data.append(binary_list)
    return data


def main():
    args = init_args()
    # Each element of <predicted / target>_labels_list has <predicted / target>_labels for corresponding development data.
    predicted_labels_list = read_file(args.predicted_labels)
    target_labels_list = read_file(args.target_labels)

    
    gts = utils.read_label(
        args.grammar_term_set
    )
    if args.top_n != 0: # if set top_n
        gts = gts[:args.top_n]
    gt2id = {gt: i for i, gt in enumerate(gts)}
    
    id2grammar_term = {id: grammar_term for grammar_term, id in gt2id.items()}
    # Grammar term labels -> ids
    def labels_to_ids(list):
        ret = []
        for labels in list:
            ids = []
            for label in labels:
                if label in gt2id:
                    ids.append(gt2id[label])
                elif label == "":
                    continue
                else:
                    raise('error')
            ret.append(ids)
        return ret

    predicted_labels_id_list = labels_to_ids(predicted_labels_list)
    target_labels_id_list = labels_to_ids(target_labels_list)
    # from sklearn.metrics import classification_report
    # classification_report
    # Ids -> binary list
    predicted_binaries_list = to_binary_list(
        predicted_labels_id_list, len(gt2id)
    )
    target_binaries_list = to_binary_list(target_labels_id_list, len(gt2id))

    while len(predicted_binaries_list) < len(target_binaries_list):
        predicted_binaries_list.append([0] * len(gt2id))

    # for idx in reversed(range(len(gt2id))):
    #     cnt = 0
    #     for jdx in range(len(target_binaries_list)):
    #         if (
    #             target_binaries_list[jdx][idx] == 1
    #             and predicted_binaries_list[jdx][idx] == 1
    #         ):
    #             cnt += 1
    #     if cnt == 0:
    #         for jdx in range(len(target_binaries_list)):
    #             predicted_binaries_list[jdx].pop(idx)
    #             target_binaries_list[jdx].pop(idx)

    (
        exact_match_ratio,
        hamming_loss,
        micro_avg_precision,
        micro_avg_recall,
        f_measure,
        precision_of_each_class,
        recall_of_each_class,
        exact_match_ratio_for_each_label
    ) = evals(
        np.array(target_binaries_list),
        np.array(predicted_binaries_list),
        list(gt2id.keys()),
    )

    with open(args.output_file, "a", encoding="utf-8") as file:
        file.write(
            "\t".join(
                map(
                    str,
                    [
                        os.path.basename(args.predicted_labels),
                        exact_match_ratio,
                        hamming_loss,
                        micro_avg_precision,
                        micro_avg_recall,
                        f_measure,
                    ],
                )
            )
            + "\n"
        )
        file.write("---exact match ratio of each label---\n")
        for idx, emr_ratio in enumerate(exact_match_ratio_for_each_label):
            grammar_term = id2grammar_term[idx]
            file.write("\t".join(map(str, [grammar_term, emr_ratio])))
            file.write("\n")



if __name__ == "__main__":
    main()
