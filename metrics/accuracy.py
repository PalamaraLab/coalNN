import os
import json


class AccuracyMetrics:
    def __init__(self, mode, session_name, tensorboard=None):
        self.mode = mode
        self.session_name = session_name
        self.tensorboard = tensorboard
        self.value = float('inf')
        self.n_step = 0

    def update(self, value, step):
        self.value = value
        self.n_step = step

    def update_step(self, step):
        self.n_step = step

    def evaluate(self, value, step):
        self.tensorboard.add_scalar(self.mode + '/metric', value, step)
        if self.value > value:
            print('New best ' + self.mode + ' score: {:.3f} -> {:.3f}'.format(self.value, value))
            self.update(value, step)
            self.save_json()
            return True
        else:
            return False

    def save_json(self):
        filename = os.path.join(self.session_name, self.mode + '_metrics.json')
        output = {'value': self.value, 'n_step': self.n_step}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def load_json(self, path):
        filename = os.path.join(path, self.mode + '_metrics.json')
        with open(filename, 'r', encoding='utf-8') as json_file:
            output = json.load(json_file)
        self.value = output['value']
        self.n_step = output['n_step']
