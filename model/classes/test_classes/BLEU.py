import sys

sys.path.append('./')

from ..Vocabulary import START_TOKEN, END_TOKEN

from nltk.translate.bleu_score import sentence_bleu


class BLEU:
    @staticmethod
    def get_bleu_score(predict_result, gui_name, input_path):
        result = (predict_result.replace(START_TOKEN, "")
                  .replace(END_TOKEN, "")
                  .replace("{", " ")
                  .replace(",", " ")
                  .replace("}", " "))

        # print("BLEU result: {}".format(result))

        correct_result = (open(input_path + '/' + gui_name +'.gui', 'r')
                          .read()
                          .replace(" ", "")
                          .replace("{", " ")
                          .replace(",", " ")
                          .replace("}", " "))

        prepared_result = result.split()
        prepared_correct = [correct_result.split()]

        return [
            sentence_bleu(prepared_correct, prepared_result),                       # avg score
            sentence_bleu(prepared_correct, prepared_result, weights=(1, 0, 0, 0)), # individual 1-gram
            sentence_bleu(prepared_correct, prepared_result, weights=(0, 1, 0, 0)), # individual 2-gram
            sentence_bleu(prepared_correct, prepared_result, weights=(0, 0, 1, 0)), # individual 3-gram
            sentence_bleu(prepared_correct, prepared_result, weights=(0, 0, 0, 1))  # individual 4-gram
        ]
