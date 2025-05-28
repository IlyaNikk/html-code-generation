train_model_web:
	python3 model/train.py datasets/web/training_set datasets/web/eval_set bin/web

train_model_android:
	python3 model/train.py datasets/android/training_set datasets/android/eval_set bin/android

train_model_ios:
	python3 model/train.py datasets/ios/training_set datasets/ios/eval_set bin/ios

train_autoencoder_web:
	python3 model/train.py datasets/web/training_set datasets/web/eval_set bin/web 1

train_autoencoder_android:
	python3 model/train.py datasets/android/training_set datasets/android/eval_set bin/android 1

train_autoencoder_ios:
	python3 model/train.py datasets/ios/training_set datasets/ios/eval_set bin/ios 1

bleu_for_web:
	python3 model/tests/bleu_score_test.py bin/web Main_Model.weights tests

functional_for_web:
	python3 model/tests/functional-test.py bin/web Main_Model.weights datasets/web/eval_set

functional_for_web_diff:
	python3 model/tests/functional-test.py bin/web Main_Model.weights datasets/web/eval_set 1

compile_gui:
	python3 ./compiler/web-compiler.py ${PATH}

autoencoder_predict:
	python3 model/tests/autoencoder.py datasets/web/training_set bin/web tests

create_dataset:
	python3 compiler/generate_dataset.py ${COUNT}

train_model_new_web:
	python3 model/train.py datasets/generated/web/training_set datasets/generated/web/eval_set bin/web
