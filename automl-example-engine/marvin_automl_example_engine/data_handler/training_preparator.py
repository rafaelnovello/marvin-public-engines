#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):

        from sklearn.model_selection import train_test_split

        X = self.marvin_initial_dataset.drop("Survived", axis=1)
        y = self.marvin_initial_dataset["Survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        self.marvin_dataset = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
