from nltk.translate.bleu_score import sentence_bleu
import copy
import re

def compute_sentence_bleu(reference_text, pred_text):
    references = [reference_text.strip().split()]
    hypothesis = pred_text.strip().split()
    return sentence_bleu(references, hypothesis)


def bleu_reward_estimation(tokenized_reference_text_list, pred_text_list):
    tokenized_reference_list = copy.deepcopy(tokenized_reference_text_list)
    batch_size = len(tokenized_reference_list)
    sentence_bleu_list = []
    for k in range(batch_size):
        one_ref_sen = tokenized_reference_list[k]
        one_pred_sen = pred_text_list[k]
        one_sen_bleu = compute_sentence_bleu(one_ref_sen, one_pred_sen)
        sentence_bleu_list.append(one_sen_bleu)
    return sentence_bleu_list

def correct_ques_num_reward_estimation(tokenized_reference_text_list, pred_text_list):
    tokenized_reference_text_list = copy.deepcopy(tokenized_reference_text_list)
    tokenized_pred_text_list = []
    for text in pred_text_list:
        if len(re.findall("\?", ' '.join(text))) != 0:
            tokenized_pred_text_list.append(len(re.findall("\?", ' '.join(text))))
        else:
            tokenized_pred_text_list.append(1)

    # Normalised reward based on formula : 1 - (|num_of_pred_ques - gt_ques| / gt_ques)
    batch_size = len(tokenized_reference_text_list)
    correct_ques_list_reward = []
    for k in range(batch_size):
        gt_ques_num = len(re.findall("\?", tokenized_reference_text_list[k]))
        reward = 1 - (abs(gt_ques_num - tokenized_pred_text_list[k]) / max(gt_ques_num, 1))
        correct_ques_list_reward.append(reward)

    return correct_ques_list_reward