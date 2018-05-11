
from flask import request
from flask import Flask
from flask import render_template

import inspect
import re
from marvin_python_toolbox.common.config import Config


app = Flask(__name__)


acquisitor_template = """
        import pandas as pd
        from marvin_python_toolbox.common.data import MarvinData

        initial_dataset = pd.read_csv(
                MarvinData.download_file("{url}"),
                sep=None, encoding='utf-8')

        self.marvin_initial_dataset = initial_dataset
"""

tpreparator_template = """
        from sklearn.model_selection import train_test_split

        X = self.marvin_initial_dataset.drop("{target_col}", axis=1)
        y = self.marvin_initial_dataset["{target_col}"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        self.marvin_dataset = {{
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }}
"""

trainer_template = """
        from tpot import TPOTClassifier

        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
        tpot.fit(self.marvin_dataset["X_train"], self.marvin_dataset["y_train"])

        self.marvin_model = {{
            "pipe": tpot.fitted_pipeline_,
        }}
"""

metrics_template = """
        score = self.marvin_model["pipe"].score(
            self.marvin_dataset["X_test"],
            self.marvin_dataset["y_test"]
        )
        self.marvin_metrics = score
"""

ppreparator_template = """
        \"""
        Return a prepared input_message compatible to the predict algorithm used by the model.
        Use the self.model and self.metrics objects if necessary.
        \"""
        return input_message
"""

predictor_template = """
        final_prediction = self.marvin_model["pipe"].predict(input_message)[0]

        return final_prediction
"""

CLAZZES = {
    "AcquisitorAndCleaner": acquisitor_template,
    "TrainingPreparator": tpreparator_template,
    "Trainer": trainer_template,
    "MetricsEvaluator": metrics_template,
    "PredictionPreparator": ppreparator_template,
    "Predictor": predictor_template,
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        form = {k:v[0] for k, v in dict(request.form).items()}

        for clazz, template in CLAZZES.items():
            if not template: continue
            #marvin_action_clazz = getattr(__import__(Config.get("package")), CLAZZES["acquisitor"])
            marvin_action_clazz = getattr(__import__('marvin_automl_example_engine'), clazz)
            source_path = inspect.getsourcefile(marvin_action_clazz)
            
            with open(source_path, 'r+') as fp:
                    lines = fp.readlines()
                    fp.seek(0)
                    for line in lines:
                        if 'def execute' in line:
                            fp.write(line)
                            fp.write(template.format(**form))
                            fp.truncate()
                            break
                        else:
                            fp.write(line)

        return render_template('index.html')