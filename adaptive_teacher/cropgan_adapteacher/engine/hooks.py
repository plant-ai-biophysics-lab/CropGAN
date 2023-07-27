# Borrows heavily from detectron2.engine.hooks EvalHook, Copyright (c) Facebook, Inc.

from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter

import detectron2.utils.comm as comm
from detectron2.engine.hooks import EvalHook
from detectron2.evaluation.testing import flatten_results_dict

class TBEvalHook(EvalHook):
    """
    Adds Tensorboard writer to the standard eval hook.
    """
    def __init__(self, eval_period, eval_function, writer: SummaryWriter):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self.writer = writer
        self.iteration_count = 0

    def _do_eval(self):
        self.iteration_count += self._period
        results = self._func()

        # Student tracking
        # if 'bbox_student' in results:
        #     print("STUDENT EVAL!!")
        #     for val_metric in ["AP", "AP50", "AP75"]:
        #         self.writer.add_scalar("student_"+val_metric, results["bbox_student"][val_metric], self.iteration_count+1)
        
        # # Teacher tracking
        # if "bbox_teacher" in results:
        #     print("TEACHER EVAL!!")
        #     for val_metric in ["AP", "AP50", "AP75"]:
        #         self.writer.add_scalar("teacher_"+val_metric, results["bbox_teacher"][val_metric], self.iteration_count+1)

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()