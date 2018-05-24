import time
from collections import defaultdict
from datetime import datetime

import numpy
from visdom import Visdom

from logger.plotting import plot_line


class Experiment(object):
    """
    Experiment class
    """

    def __init__(self, name, hparams, desc=None):
        """

        Args:
            name (string): the name of the experiment
            hparams (object): the hypermarameters used for this experiment
        """
        self.name = name
        self.desc = desc
        self.hparams = hparams
        self.metrics = defaultdict(Metric)

        self.timestamp_start = datetime.now()
        self.timestamp_update = datetime.now()
        self.last_update = time.time()

        self.viz = Visdom()

        self.viz.close()
        self.vis_params()

        if desc is not None:
            self.vis_desc()

    def update_plots(self):
        for exp_name, metric in self.metrics.items():
            metric.update_plot()
        # self.save_experiment()

    def vis_params(self):
        lines = []
        for param, value in self.hparams.items():
            lines.append("{}: {}".format(param, value))
        self.viz.text("<pre>{}</pre>".format("\n".join(lines)),
                      opts=dict(
                          width=max([len(x) for x in lines]) * 10,
                          height=len(lines) * 20,
                      ))

    def vis_desc(self):
        self.viz.text("<pre>{}</pre>".format(self.desc),
                      opts=dict(
                          width=max([len(x) for x in
                                     self.desc.split("\n")]) * 8.5,
                          height=len(self.desc.split("\n")) * 20,
                      ))

    def add_metric(self, metric):
        """
        Add a metric to the experiment
        Args:
            metric (Metric): a metric object

        Returns:

        """
        metric.vic_context = self.viz
        self.metrics[metric.name] = metric

    def get_score(self, metric, tag):
        return self.metrics[metric]._values[tag][-1]

    def save_experiment(self):
        """
        Implement a saving mechanism (in text, csv or a database)
        Returns:

        """
        self.timestamp_update = datetime.now()
        self.db.update({
            'name': self.name,
            'desc': self.desc,
            'hparams': self.hparams,
            'metrics': self.metrics,
            'timestamp_start': self.timestamp_start,
            'timestamp_update': self.timestamp_update,
            'last_update': self.last_update
        }, doc_ids=[self.db_record])

    def visualize_experiment(self):
        raise NotImplementedError


class Metric(object):
    """
    Metric hold the data of a value of the model that is being monitored

    A Metric object has to have a name,
    a vis_type which defines how it will be visualized
    and a dataset on which it will be attached to.
    """

    def __init__(self, name, vis_type, tags):
        """

        Args:
            name (str): the name of the metric
            vis_type (str): the visualization type
            tags (list): list of tags
        """
        self.name = name
        self.vis_type = vis_type
        self.tags = tags

        # a list that contains the values of the metric
        self.values = {tag: [] for tag in tags}

        # a helper list that contains the steps or times,
        # that each value was added to the Metric
        self.steps = {tag: [] for tag in tags}

        self.vic_context = None
        self._win = None

    def append(self, tag, value):
        """
        Add a value to the list of values of this metric
        Args:
            tag (str):
            value (int,float):

        Returns:

        """
        try:
            self.steps[tag].append(self.steps[tag][-1] + 1)
        except:
            self.steps[tag].append(1)

        self.values[tag].append(value)

    def update_plot(self):

        if self.vis_type == "line":
            plot_line(self.vic_context,
                      numpy.array([self.values[tag] for tag in self.tags]),
                      self.name, self.tags)
