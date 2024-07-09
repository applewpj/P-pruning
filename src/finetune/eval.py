import logging
from src.utils.data import get_dataset
from src.utils.data import get_sequence_tasks, get_qa_tasks, get_lm_tasks
import math

logger = logging.getLogger()


def eval_seq(trainer, test_datasets):
    logger.info("*** Evaluate ***")
    for test_task, test_dataset in test_datasets.items():
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        if test_task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        trainer.log_metrics("eval", metrics)
        logger.info("Evaluation metrics: {}".format(metrics))
    
def eval_qa(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    logger.info("Evaluation metrics: {}".format(metrics))
    
def eval_lm(task_name, trainer, test_datasets):
    logger.info("*** Evaluate ***")
    test_dataset = test_datasets[task_name]
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(test_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    logger.info("Validation metrics: {}".format(metrics))

    metrics = trainer.evaluate(test_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    logger.info("Test metrics: {}".format(metrics))
    
def evaluation(task_name, trainer, test_datasets):
    if task_name in get_sequence_tasks():
        eval_seq(trainer, test_datasets)
    elif task_name in get_qa_tasks():
        eval_qa(trainer)
    elif task_name in get_lm_tasks():
        eval_lm(task_name, trainer, test_datasets)
        