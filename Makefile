# Train on the original Beltramelli pix2code web dataset (datasets/web/).
PROFILE ?= web_generated
MODE ?= greedy
LIMIT ?= 0
SEQUENCE_LENGTH ?= 150
MIN_TARGET_LENGTH ?= 0
MAX_TARGET_LENGTH ?= 0
ABLATION_SEEDS ?= 1,2,3,4,5
IMAGE_ABLATION_MANIFEST ?= datasets/generated/web/article/image_ablation/original.json
IMAGE_ABLATION_ROOT ?= bin/web/correct/new_metrics/image_ablation
IMAGE_ABLATION_OUTPUT_DIR ?= ${IMAGE_ABLATION_ROOT}
IMAGE_ABLATION_REPORT_DIR ?= ${IMAGE_ABLATION_ROOT}/report_v3
RL_STEPS ?= 20
RL_SEQUENCE_LENGTH ?= 100
RL_MODE ?= structural_rl_v4
RL_SEED ?= 1234
RL_CHECKPOINT_EVERY ?= 5
RL_GRADIENT_MICROBATCH_SIZE ?= 1
RL_OUTPUT_DIR ?= bin/web/correct/new_metrics/rl_step2/${RL_MODE}/seed_${RL_SEED}
EVAL_WEIGHTS_PATH ?= bin/web/correct/new_metrics
EVAL_OUTPUT_DIR ?= ${EVAL_WEIGHTS_PATH}/extended_metrics_step2
EVAL_SEQUENCE_LENGTH ?= 100
EVAL_RENDER_TIMEOUT_SECONDS ?= 90
EVAL_ONLY_SAMPLE ?=
EVAL_ONLY_SAMPLE_FLAG = $(if $(strip ${EVAL_ONLY_SAMPLE}),--only-sample ${EVAL_ONLY_SAMPLE})
RL_RESULTS_ROOT ?= bin/web/correct/new_metrics/rl_step2
RL_REPORT_DIR ?= ${RL_RESULTS_ROOT}/report
RL_EVALUATIONS_ROOT ?= ${RL_RESULTS_ROOT}/evaluations
RL_DIAGNOSTICS_DIR ?= ${RL_RESULTS_ROOT}/diagnostics
REPLICATION_VERSION ?= replication_v1
REPLICATION_DATASET_SEED ?= 404
REPLICATION_SPLIT_SEED ?= 404
REPLICATION_SAMPLE_COUNT ?= 1020
REPLICATION_SPLIT_DISTRIBUTION ?= 12
REPLICATION_SFT_SEED ?= 1234
REPLICATION_DATASET_ROOT ?= datasets/generated/web/${REPLICATION_VERSION}
REPLICATION_RAW_DATASET_DIR ?= ${REPLICATION_DATASET_ROOT}/all_data
REPLICATION_SPLIT_MANIFEST ?= ${REPLICATION_DATASET_ROOT}/split_manifest.json
REPLICATION_SUPERVISED_DIR ?= bin/web/correct/${REPLICATION_VERSION}/supervised
REPLICATION_RESULTS_ROOT ?= bin/web/correct/${REPLICATION_VERSION}/rl
REPLICATION_EVALUATIONS_ROOT ?= ${REPLICATION_RESULTS_ROOT}/evaluations
REPLICATION_REPORT_DIR ?= ${REPLICATION_RESULTS_ROOT}/report
REPLICATION_DIAGNOSTICS_DIR ?= ${REPLICATION_RESULTS_ROOT}/diagnostics
REPLICATION_RL_OUTPUT_DIR ?= ${REPLICATION_RESULTS_ROOT}/${RL_MODE}/seed_${RL_SEED}
REPLICATION_EVAL_WEIGHTS_PATH ?= ${REPLICATION_SUPERVISED_DIR}
REPLICATION_EVAL_OUTPUT_DIR ?= ${REPLICATION_EVALUATIONS_ROOT}/supervised_constrained
REPLICATION_IMAGE_ABLATION_MANIFEST_DIR ?= ${REPLICATION_DATASET_ROOT}/image_ablation
REPLICATION_IMAGE_ABLATION_MANIFEST ?= ${REPLICATION_IMAGE_ABLATION_MANIFEST_DIR}/original.json
REPLICATION_IMAGE_ABLATION_ROOT ?= bin/web/correct/${REPLICATION_VERSION}/image_ablation
REPLICATION_IMAGE_ABLATION_OUTPUT_DIR ?= ${REPLICATION_IMAGE_ABLATION_ROOT}
REPLICATION_IMAGE_ABLATION_REPORT_DIR ?= ${REPLICATION_IMAGE_ABLATION_ROOT}/report
EXTERNAL_EVAL_DATASET ?= datasets/generated/web/replication_v1/eval_set
EXTERNAL_EVAL_SPLIT_MANIFEST ?= datasets/generated/web/replication_v1/split_manifest.json
EXTERNAL_EVAL_SOURCE_RESULTS_ROOT ?= bin/web/correct/new_metrics/rl_step2_v2
EXTERNAL_EVAL_SOURCE_SUPERVISED ?= bin/web/correct/new_metrics
EXTERNAL_EVAL_ROOT ?= bin/web/correct/new_metrics/rl_step2_v2_external_eval_v1
EXTERNAL_EVAL_OUTPUT_DIR ?= ${EXTERNAL_EVAL_ROOT}/evaluations/supervised_constrained
EXTERNAL_EVAL_WEIGHTS_PATH ?= ${EXTERNAL_EVAL_SOURCE_SUPERVISED}
EXTERNAL_EVAL_REPORT_DIR ?= ${EXTERNAL_EVAL_ROOT}/report
EXTERNAL_EVAL_DIAGNOSTICS_DIR ?= ${EXTERNAL_EVAL_ROOT}/diagnostics
EXTERNAL_EVAL_ABLATION_MANIFEST_DIR ?= datasets/generated/web/replication_v1/external_eval_ablation
EXTERNAL_EVAL_ABLATION_MANIFEST ?= ${EXTERNAL_EVAL_ABLATION_MANIFEST_DIR}/original.json
EXTERNAL_EVAL_ABLATION_ROOT ?= ${EXTERNAL_EVAL_ROOT}/image_ablation
EXTERNAL_EVAL_ABLATION_OUTPUT_DIR ?= ${EXTERNAL_EVAL_ABLATION_ROOT}
EXTERNAL_EVAL_ABLATION_REPORT_DIR ?= ${EXTERNAL_EVAL_ABLATION_ROOT}/report

train_model_web:
	python3 model/train.py --profile web

# Train on the synthetic set produced by compiler/generate_dataset.py (datasets/generated/web/).
# Alias `train_model_new_web` below is kept for backward compatibility.
train_model_generated_web:
	python3 model/train.py --profile web_generated

train_model_generated_web_full:
	python3 model/train.py --profile web_generated --train-autoencoder --steps-fraction 1.0

# Independent replication: a new seeded synthetic dataset and a fresh SFT checkpoint.
generate_replication_dataset_web:
	python3 compiler/generate_dataset.py ${REPLICATION_SAMPLE_COUNT} --seed ${REPLICATION_DATASET_SEED} --output-dir ${REPLICATION_RAW_DATASET_DIR}

split_replication_dataset_web:
	python3 model/build_datasets.py ${REPLICATION_RAW_DATASET_DIR} ${REPLICATION_SPLIT_DISTRIBUTION} --seed ${REPLICATION_SPLIT_SEED} --clean-output --manifest ${REPLICATION_SPLIT_MANIFEST}

train_model_replication_web:
	python3 model/train.py --profile web_generated_replication_v1 --train-autoencoder --seed ${REPLICATION_SFT_SEED}

train_model_android:
	python3 model/train.py datasets/android/training_set datasets/android/eval_set bin/android

train_model_ios:
	python3 model/train.py datasets/ios/training_set datasets/ios/eval_set bin/ios

# Autoencoder pretraining for the web set (use --profile web to mirror train_model_web).
train_autoencoder_web:
	python3 model/train.py --profile web --train-autoencoder

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

bleu_for_generated_web:
	python3 model/tests/bleu_score_test.py bin/web/correct/new_metrics Main_Model.weights datasets/generated/web/article/eval_set

functional_for_generated_web:
	python3 model/tests/functional-test.py bin/web/correct/new_metrics Main_Model.weights datasets/generated/web/article/eval_set

functional_for_generated_web_diff:
	python3 model/tests/functional-test.py bin/web/correct/new_metrics Main_Model.weights datasets/generated/web/article/eval_set 1

eval_extended_web:
	python3 model/tests/evaluate_extended.py --profile ${PROFILE} --mode ${MODE} --limit ${LIMIT} --sequence-length ${SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH}

eval_constrained_web:
	python3 model/tests/evaluate_extended.py --profile ${PROFILE} --mode constrained --limit ${LIMIT} --sequence-length ${SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH}

prepare_image_ablation_web:
	python3 model/tests/prepare_image_ablation.py --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --seeds ${ABLATION_SEEDS}

eval_image_ablation_web:
	python3 model/tests/evaluate_extended.py --profile ${PROFILE} --mode constrained --sequence-length ${SEQUENCE_LENGTH} --image-ablation-manifest ${IMAGE_ABLATION_MANIFEST} --output-dir ${IMAGE_ABLATION_OUTPUT_DIR}

report_image_ablation_web:
	python3 model/tests/report_image_ablation.py --input-root ${IMAGE_ABLATION_ROOT} --output-dir ${IMAGE_ABLATION_REPORT_DIR}

rl_pilot_web:
	python3 model/rl_finetune.py --profile ${PROFILE} --steps ${RL_STEPS} --sequence-length ${RL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH}

# Fixed Step-2 experiment: continued SFT, frozen structural RL, and V3 visual RL.
test_step2_rl:
	python3 -m unittest model.tests.test_visual_score_v3 model.tests.test_rl_finetune model.tests.test_report_step2_results model.tests.test_step2_diagnostics

test_replication_protocol: test_step2_rl

rl_step2_web:
	python3 model/rl_finetune.py --profile ${PROFILE} --mode ${RL_MODE} --steps ${RL_STEPS} --seed ${RL_SEED} --sequence-length ${RL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --checkpoint-every ${RL_CHECKPOINT_EVERY} --gradient-microbatch-size ${RL_GRADIENT_MICROBATCH_SIZE} --output-dir ${RL_OUTPUT_DIR}

resume_rl_step2_web:
	python3 model/rl_finetune.py --profile ${PROFILE} --mode ${RL_MODE} --steps ${RL_STEPS} --seed ${RL_SEED} --sequence-length ${RL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --checkpoint-every ${RL_CHECKPOINT_EVERY} --gradient-microbatch-size ${RL_GRADIENT_MICROBATCH_SIZE} --output-dir ${RL_OUTPUT_DIR} --resume

eval_step2_web:
	python3 model/tests/evaluate_extended.py --profile ${PROFILE} --weights-path ${EVAL_WEIGHTS_PATH} --mode constrained --sequence-length ${EVAL_SEQUENCE_LENGTH} --render-timeout-seconds ${EVAL_RENDER_TIMEOUT_SECONDS} ${EVAL_ONLY_SAMPLE_FLAG} --output-dir ${EVAL_OUTPUT_DIR}

report_step2_results:
	python3 model/tests/report_step2_results.py --input-root ${RL_RESULTS_ROOT} --evaluations-root ${RL_EVALUATIONS_ROOT} --output-dir ${RL_REPORT_DIR}

report_step2_diagnostics:
	python3 model/tests/report_step2_diagnostics.py --input-root ${RL_RESULTS_ROOT} --evaluations-root ${RL_EVALUATIONS_ROOT} --output-dir ${RL_DIAGNOSTICS_DIR}

# Six-run independent replication: CE control and visual RL, three seeds each.
rl_replication_web:
	python3 model/rl_finetune.py --profile web_generated_replication_v1 --mode ${RL_MODE} --steps ${RL_STEPS} --seed ${RL_SEED} --sequence-length ${RL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --checkpoint-every ${RL_CHECKPOINT_EVERY} --gradient-microbatch-size ${RL_GRADIENT_MICROBATCH_SIZE} --output-dir ${REPLICATION_RL_OUTPUT_DIR}

resume_rl_replication_web:
	python3 model/rl_finetune.py --profile web_generated_replication_v1 --mode ${RL_MODE} --steps ${RL_STEPS} --seed ${RL_SEED} --sequence-length ${RL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --checkpoint-every ${RL_CHECKPOINT_EVERY} --gradient-microbatch-size ${RL_GRADIENT_MICROBATCH_SIZE} --output-dir ${REPLICATION_RL_OUTPUT_DIR} --resume

eval_replication_web:
	python3 model/tests/evaluate_extended.py --profile web_generated_replication_v1 --weights-path ${REPLICATION_EVAL_WEIGHTS_PATH} --mode constrained --sequence-length ${EVAL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --render-timeout-seconds ${EVAL_RENDER_TIMEOUT_SECONDS} ${EVAL_ONLY_SAMPLE_FLAG} --output-dir ${REPLICATION_EVAL_OUTPUT_DIR}

prepare_replication_image_ablation_web:
	python3 model/tests/prepare_image_ablation.py --input-path ${REPLICATION_DATASET_ROOT}/eval_set --output-dir ${REPLICATION_IMAGE_ABLATION_MANIFEST_DIR} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --seeds ${ABLATION_SEEDS}

eval_replication_image_ablation_web:
	python3 model/tests/evaluate_extended.py --profile web_generated_replication_v1 --weights-path ${REPLICATION_SUPERVISED_DIR} --mode constrained --sequence-length ${EVAL_SEQUENCE_LENGTH} --render-timeout-seconds ${EVAL_RENDER_TIMEOUT_SECONDS} --image-ablation-manifest ${REPLICATION_IMAGE_ABLATION_MANIFEST} --output-dir ${REPLICATION_IMAGE_ABLATION_OUTPUT_DIR}

report_replication_image_ablation_web:
	python3 model/tests/report_image_ablation.py --input-root ${REPLICATION_IMAGE_ABLATION_ROOT} --output-dir ${REPLICATION_IMAGE_ABLATION_REPORT_DIR}

report_replication_results:
	python3 model/tests/report_step2_results.py --input-root ${REPLICATION_RESULTS_ROOT} --evaluations-root ${REPLICATION_EVALUATIONS_ROOT} --output-dir ${REPLICATION_REPORT_DIR} --modes ce_control,visual_structural_rl_v6

report_replication_diagnostics:
	python3 model/tests/report_step2_diagnostics.py --input-root ${REPLICATION_RESULTS_ROOT} --evaluations-root ${REPLICATION_EVALUATIONS_ROOT} --output-dir ${REPLICATION_DIAGNOSTICS_DIR} --modes ce_control,visual_structural_rl_v6

# External evaluation replication: reuse the completed v2 checkpoints on the independently seeded eval set.
eval_external_v2_web:
	python3 model/tests/evaluate_extended.py --profile web_generated --weights-path ${EXTERNAL_EVAL_WEIGHTS_PATH} --input-path ${EXTERNAL_EVAL_DATASET} --mode constrained --sequence-length ${EVAL_SEQUENCE_LENGTH} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --render-timeout-seconds ${EVAL_RENDER_TIMEOUT_SECONDS} ${EVAL_ONLY_SAMPLE_FLAG} --output-dir ${EXTERNAL_EVAL_OUTPUT_DIR}

prepare_external_eval_ablation_web:
	python3 model/tests/prepare_image_ablation.py --input-path ${EXTERNAL_EVAL_DATASET} --output-dir ${EXTERNAL_EVAL_ABLATION_MANIFEST_DIR} --min-target-length ${MIN_TARGET_LENGTH} --max-target-length ${MAX_TARGET_LENGTH} --seeds ${ABLATION_SEEDS}

eval_external_eval_ablation_web:
	python3 model/tests/evaluate_extended.py --profile web_generated --weights-path ${EXTERNAL_EVAL_SOURCE_SUPERVISED} --input-path ${EXTERNAL_EVAL_DATASET} --mode constrained --sequence-length ${EVAL_SEQUENCE_LENGTH} --render-timeout-seconds ${EVAL_RENDER_TIMEOUT_SECONDS} --image-ablation-manifest ${EXTERNAL_EVAL_ABLATION_MANIFEST} --output-dir ${EXTERNAL_EVAL_ABLATION_OUTPUT_DIR}

report_external_eval_ablation_web:
	python3 model/tests/report_image_ablation.py --input-root ${EXTERNAL_EVAL_ABLATION_ROOT} --output-dir ${EXTERNAL_EVAL_ABLATION_REPORT_DIR}

report_external_v2_results:
	python3 model/tests/report_step2_results.py --input-root ${EXTERNAL_EVAL_SOURCE_RESULTS_ROOT} --evaluations-root ${EXTERNAL_EVAL_ROOT}/evaluations --output-dir ${EXTERNAL_EVAL_REPORT_DIR} --modes ce_control,visual_structural_rl_v6 --expected-eval-manifest ${EXTERNAL_EVAL_SPLIT_MANIFEST} --expected-eval-input-path ${EXTERNAL_EVAL_DATASET} --expected-eval-min-target-length ${MIN_TARGET_LENGTH} --expected-eval-max-target-length ${MAX_TARGET_LENGTH}

report_external_v2_diagnostics:
	python3 model/tests/report_step2_diagnostics.py --input-root ${EXTERNAL_EVAL_SOURCE_RESULTS_ROOT} --evaluations-root ${EXTERNAL_EVAL_ROOT}/evaluations --output-dir ${EXTERNAL_EVAL_DIAGNOSTICS_DIR} --modes ce_control,visual_structural_rl_v6

compile_gui:
	python3 ./compiler/web-compiler.py ${PATH}

autoencoder_predict:
	python3 model/tests/autoencoder.py datasets/web/training_set bin/web tests

create_dataset:
	python3 compiler/generate_dataset.py ${COUNT}

# Backward-compat alias for train_model_generated_web (older docs/scripts may reference this name).
train_model_new_web: train_model_generated_web

predict_one_web:
	python3 model/tests/predict_one.py bin/web Main_Model.weights ${IMAGE_PATH}
