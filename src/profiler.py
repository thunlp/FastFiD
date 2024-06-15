from contextlib import contextmanager
import time
import torch

@contextmanager
def profile(enabled):
    # enter
    timer = None
    if enabled:
        timer = Timers()
    else:
        timer = EmptyTimers()
    yield timer
    # exit


    


class _Timer:
    """Timer. Code from Megatron"""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_

class Timers:
    """Group of timers. Code from Megatron"""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, logger, names=None, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        if names is None:
            names = list(self.timers.keys())
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        logger.info(string)
    
    def to_dict(self, names=None, normalizer=1.0, reset=True):
        assert normalizer > 0.0
        return_dict = {}
        if names is None:
            names = list(self.timers.keys())
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            return_dict[name] = elapsed_time
        return return_dict
    
    @contextmanager
    def record(self, label):
        # __enter__ start
        self(label).start()
        yield
        # __exit__ end
        self(label).stop()


class EmptyTimers:
    def __init__(self):
        self.timers = {}
    
    def __call__(self, name):
        return None
    
    @contextmanager
    def record(self, label):
        # __enter__ start
        yield
        # __exit__ end
    
    def log(self, logger, names=None, normalizer=1.0, reset=True):
        pass
    
    def to_dict(self, names=None, normalizer=1.0, reset=True):
        return self.timers